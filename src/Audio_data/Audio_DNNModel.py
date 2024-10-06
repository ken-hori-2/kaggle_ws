# 音声認識モデルの学習プロセスを組み込み、精度向上のための特徴量エンジニアリングやデータ前処理も取り入れたコードを構築します。

# 以下に、音声認識モデルのトレーニングと特徴量エンジニアリング、データ前処理を行う流れを紹介します。

# ### ステップ 1: データ前処理の改善

# 音声データの前処理を改善し、データをより扱いやすい形式に変換するために、以下の手法を導入します。

# 1. **データ正規化**: メルスペクトログラムを標準化し、入力データのばらつきを抑える。
# 2. **データ拡張**: データが少ない場合、ピッチシフトやノイズの追加などのデータ拡張を取り入れることで、モデルの汎化性能を高める。

# ```python
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random
import librosa
import os
import glob

# データセットの定義
class AudioPreprocessingDataset(Dataset):
    def __init__(self, data_path, sample_rate=16000, n_mels=64, augment=False):
        self.data_path = data_path
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.audio_files = glob.glob(os.path.join(data_path, '*.wav'))
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate, 
            n_mels=self.n_mels
        )
        self.augment = augment

    def __len__(self):
        return len(self.audio_files)

    def augment_waveform(self, waveform):
        # データ拡張: ランダムノイズの追加
        if random.random() > 0.5:
            noise = torch.randn_like(waveform) * 0.005
            waveform += noise
        return waveform

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        waveform, original_sample_rate = torchaudio.load(audio_file)

        # リサンプリング
        if original_sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # データ拡張を行う場合
        if self.augment:
            waveform = self.augment_waveform(waveform)

        # メルスペクトログラムに変換
        mel_spec = self.mel_spectrogram(waveform)

        # 正規化
        mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()

        # ラベルの作成（例としてファイル名からラベルを取得する場合）
        label = int(audio_file.split('/')[-1].split('_')[0])  # 例：ファイル名に基づく

        return mel_spec, label
# ```

# ### ステップ 2: DNNモデルの定義

# メルスペクトログラムを入力とするニューラルネットワークモデルを改良します。精度向上のために、**ドロップアウト**と**バッチ正規化**を追加します。

# ```python
class AudioClassificationModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(AudioClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.batch_norm2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # フラット化
        x = self.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
# ```

# ### ステップ 3: 学習プロセスの実装

# モデルを訓練するためのコードを記述し、学習ループにおける損失関数と最適化関数を定義します。

# ```python
# データセットとデータローダー
data_path = './_sample_data'  # 音声データのフォルダ
# 音声ファイルの読み込み
# waveform, sample_rate = torchaudio.load('./_sample_data/speech.wav')

batch_size = 32

train_dataset = AudioPreprocessingDataset(data_path, augment=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# モデルの初期化
input_size = 64 * 100  # メルスペクトログラムの次元
num_classes = 10  # 仮のクラス数
model = AudioClassificationModel(input_size=input_size, num_classes=num_classes)

# 損失関数と最適化関数
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学習プロセス
num_epochs = 20

for epoch in range(num_epochs):
    model.train()  # 訓練モード
    running_loss = 0.0
    for mel_spec, labels in train_loader:
        optimizer.zero_grad()

        # 順伝播
        outputs = model(mel_spec)

        # 損失の計算
        loss = criterion(outputs, labels)

        # 逆伝播
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

print('Finished Training')
# ```

# ### ステップ 4: モデルの評価

# 訓練後、テストデータでモデルの性能を評価します。以下はモデルの評価手順です。

# ```python
def evaluate_model(model, test_loader):
    model.eval()  # 評価モード
    correct = 0
    total = 0

    with torch.no_grad():
        for mel_spec, labels in test_loader:
            outputs = model(mel_spec)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy: {100 * correct / total:.2f}%')

# テストデータで評価
test_loader = DataLoader(train_dataset, batch_size=batch_size)  # 仮にトレーニングデータを使用
evaluate_model(model, test_loader)
# ```

# ### 特徴量エンジニアリングと前処理改善

# 1. **データ拡張**: ノイズを加えたり、音声のピッチシフトを行う。
# 2. **正規化**: 音声データを標準化することで、モデルの学習を安定化させる。
# 3. **ドロップアウト**や**バッチ正規化**を利用して、過学習を防止し、モデルの汎化性能を向上させる。

# ### 結論

# このコードは、音声データを前処理し、メルスペクトログラムとしてDNNに入力するプロセスから、データ拡張を含む学習プロセスの改善、さらには評価までの一連のフローを実装しています。より精度を高めるためには、データセットの拡張や、より深いニューラルネットワークの使用が有効です。

# # モデルの出力を確認
# output = model(mel_spec_flattened)
# print(f"Model output: {output}")

# # クラスの予測
# _, predicted_class = torch.max(output.data, 1)
# print(f"Predicted class: {predicted_class.item()}")