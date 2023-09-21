import torch
import torch.nn as nn
import pandas as pd
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import os
import numpy as np
import random
import warnings
warnings.filterwarnings(action='ignore') 

CFG = {
    'IMG_SIZE':224,
    'EPOCHS':50, #Your Epochs,
    'LR':0.001, #Your Learning Rate,
    'BATCH_SIZE':128, #Your Batch Size,
    'SEED':41
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['img_path']
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        # mos column 존재 여부에 따라 값을 설정
        mos = float(self.dataframe.iloc[idx]['mos']) if 'mos' in self.dataframe.columns else 0.0
        comment = self.dataframe.iloc[idx]['comments'] if 'comments' in self.dataframe.columns else ""
        
        return img, mos, comment
    
class BaseModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512):
        super(BaseModel, self).__init__()

        # Image feature extraction using ResNet50
        self.cnn_backbone = models.resnet50(pretrained=True)
        # Remove the last fully connected layer to get features
        modules = list(self.cnn_backbone.children())[:-1]
        self.cnn = nn.Sequential(*modules)

        # Image quality assessment head
        self.regression_head = nn.Linear(2048, 1)  # ResNet50 last layer has 2048 features

        # Captioning head
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim + 2048, hidden_dim)  # Image features and caption embeddings as input
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, images, captions=None):
        # CNN
        features = self.cnn(images)
        features_flat = features.view(features.size(0), -1)

        # Image quality regression
        mos = self.regression_head(features_flat)

        # LSTM captioning
        if captions is not None:
            embeddings = self.embedding(captions)
            # Concatenate image features and embeddings for each word in the captions
            combined = torch.cat([features_flat.unsqueeze(1).repeat(1, embeddings.size(1), 1), embeddings], dim=2)
            lstm_out, _ = self.lstm(combined)
            outputs = self.fc(lstm_out)
            return mos, outputs
        else:
            return mos, None
        
# 데이터 로드
train_data = pd.read_csv('train.csv')

# 단어 사전 생성
all_comments = ' '.join(train_data['comments']).split()
vocab = set(all_comments)
vocab = ['<PAD>', '<SOS>', '<EOS>'] + list(vocab)
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}

# 데이터셋 및 DataLoader 생성
transform = transforms.Compose([
    transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])), 
    transforms.ToTensor()
])
train_dataset = CustomDataset(train_data, transform)
train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True)

# 모델, 손실함수, 옵티마이저
model = BaseModel(len(vocab)).cuda()
criterion1 = nn.MSELoss()
criterion2 = nn.CrossEntropyLoss(ignore_index=word2idx['<PAD>'])
optimizer = torch.optim.Adam(model.parameters(), lr=CFG['LR'])

# 학습
model.train()
for epoch in range(CFG['EPOCHS']):
    total_loss = 0
    loop = tqdm(train_loader, leave=True)
    for imgs, mos, comments in loop:
        imgs, mos = imgs.float().cuda(), mos.float().cuda()
        
        # Batch Preprocessing
        comments_tensor = torch.zeros((len(comments), len(max(comments, key=len)))).long().cuda()
        for i, comment in enumerate(comments):
            tokenized = ['<SOS>'] + comment.split() + ['<EOS>']
            comments_tensor[i, :len(tokenized)] = torch.tensor([word2idx[word] for word in tokenized])

        # Forward & Loss
        predicted_mos, predicted_comments = model(imgs, comments_tensor)
        loss1 = criterion1(predicted_mos.squeeze(1), mos)
        loss2 = criterion2(predicted_comments.view(-1, len(vocab)), comments_tensor.view(-1))
        loss = loss1 + loss2

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        loop.set_description(f"Epoch {epoch + 1}")
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch + 1} finished with average loss: {total_loss / len(train_loader):.4f}")

test_data = pd.read_csv('test.csv')
test_dataset = CustomDataset(test_data, transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model.eval()
predicted_mos_list = []
predicted_comments_list = []

def greedy_decode(model, image, max_length=50):
    image = image.unsqueeze(0).cuda()
    mos, _ = model(image)
    output_sentence = []
    
    # 시작 토큰 설정
    current_token = torch.tensor([word2idx['<SOS>']]).cuda()
    hidden = None
    features = model.cnn(image).view(image.size(0), -1)

    for _ in range(max_length):
        embeddings = model.embedding(current_token).unsqueeze(0)
        combined = torch.cat([features.unsqueeze(1), embeddings], dim=2)
        out, hidden = model.lstm(combined, hidden)
        
        output = model.fc(out.squeeze(0))
        _, current_token = torch.max(output, dim=1)

        # <EOS> 토큰에 도달하면 멈춤
        if current_token.item() == word2idx['<EOS>']:
            break

        # <SOS> 또는 <PAD> 토큰은 생성한 캡션에 추가하지 않음
        if current_token.item() not in [word2idx['<SOS>'], word2idx['<PAD>']]:
            output_sentence.append(idx2word[current_token.item()])
     
    return mos.item(), ' '.join(output_sentence)

# 추론 과정
with torch.no_grad():
    for imgs, _, _ in tqdm(test_loader):
        for img in imgs:
            img = img.float().cuda()
            mos, caption = greedy_decode(model, img)
            predicted_mos_list.append(mos)
            predicted_comments_list.append(caption)

# 결과 저장
result_df = pd.DataFrame({
    'img_name': test_data['img_name'],
    'mos': predicted_mos_list,
    'comments': predicted_comments_list  # 캡션 부분은 위에서 생성한 것을 사용
})

# 예측 결과에 NaN이 있다면, 제출 시 오류가 발생하므로 후처리 진행 (sample_submission.csv과 동일하게)
result_df['comments'] = result_df['comments'].fillna('Nice Image.')
result_df.to_csv('submit.csv', index=False)

print("Inference completed and results saved to submit.csv.")