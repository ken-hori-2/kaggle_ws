import torch
import torch.nn as nn
import torchaudio
from torchaudio.transforms import MelSpectrogram, Resample
from torch.utils.data import DataLoader, Dataset
import os
import glob
import librosa
import numpy as np

class AudioPreprocessingDataset(Dataset):
    def __init__(self, data_path, sample_rate=16000, n_mels=64):
        self.data_path = data_path
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.audio_files = glob.glob(os.path.join(data_path, '*.wav'))  # .mp3も対応可能
        self.mel_spectrogram = MelSpectrogram(
            sample_rate=self.sample_rate, 
            n_mels=self.n_mels
        )

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        waveform, original_sample_rate = torchaudio.load(audio_file)

        # リサンプリング
        if original_sample_rate != self.sample_rate:
            resampler = Resample(orig_freq=original_sample_rate, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # メルスペクトログラムに変換
        mel_spec = self.mel_spectrogram(waveform)

        # 正規化
        mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()

        # ラベルの作成（例としてファイル名からラベルを取得する場合）
        label = audio_file.split('/')[-1].split('_')[0]  # ファイル名に基づくラベル

        return mel_spec, label

class AudioClassificationModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(AudioClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # フラットに変換
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# データセットとデータローダー
# data_path = './audio_data'  # データが保存されているフォルダ
data_path = './_sample_data'  # データが保存されているフォルダ
dataset = AudioPreprocessingDataset(data_path)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# モデルの初期化
input_size = 64 * 100  # メルスペクトログラムのサイズによる
num_classes = 10  # クラス数を指定
model = AudioClassificationModel(input_size=input_size, num_classes=num_classes)

# 損失関数と最適化関数
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# モデルの訓練
num_epochs = 20
for epoch in range(num_epochs):
    for mel_spec, label in dataloader:
        optimizer.zero_grad()

        # 順伝播
        outputs = model(mel_spec)

        # 損失の計算
        loss = criterion(outputs, label)
        
        # 逆伝播
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def evaluate_model(model, test_dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for mel_spec, label in test_dataloader:
            outputs = model(mel_spec)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    print(f'Accuracy: {100 * correct / total:.2f}%')

# テストデータで評価
# evaluate_model(model, test_dataloader)