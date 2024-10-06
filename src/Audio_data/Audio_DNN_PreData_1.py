import torchaudio
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchaudio
from torchaudio.transforms import MelSpectrogram, Resample
from torch.utils.data import DataLoader, Dataset
import os
import glob
import librosa
import numpy as np

# 音声ファイルの読み込み
waveform, sample_rate = torchaudio.load('./_sample_data/speech.wav')

# 音声データの確認
print(f"Shape of waveform: {waveform.shape}")
print(f"Sample rate: {sample_rate}")

# メルスペクトログラムに変換
mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=64)
mel_spec = mel_spectrogram(waveform)

# メルスペクトログラムのプロット
plt.figure(figsize=(10, 4))
plt.imshow(mel_spec.log2()[0, :, :].detach().numpy(), cmap='viridis')
plt.title("Mel Spectrogram")
plt.colorbar(format='%+2.0f dB')
plt.show()



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


# メルスペクトログラムの正規化
mel_spec_normalized = (mel_spec - mel_spec.mean()) / mel_spec.std()

# モデルに入力するためにフラット化（例として1次元に変換）
# mel_spec_flattened = mel_spec_normalized.view(mel_spec_normalized.size(0), -1)
# 変更前: mel_spec_flattened = mel_spec_normalized.view(mel_spec_normalized.size(0), -1)

# 変更後
mel_spec_flattened = mel_spec_normalized.reshape(mel_spec_normalized.size(0), -1)


# 入力データのサイズを確認
print(f"Input size for the model: {mel_spec_flattened.shape}")

# DNNモデルの定義（以前のモデルを使用）
input_size = mel_spec_flattened.size(1)
num_classes = 10  # 仮のクラス数

model = AudioClassificationModel(input_size=input_size, num_classes=num_classes)

# モデルの出力を確認
output = model(mel_spec_flattened)
print(f"Model output: {output}")

# クラスの予測
_, predicted_class = torch.max(output.data, 1)
print(f"Predicted class: {predicted_class.item()}")
