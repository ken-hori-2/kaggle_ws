import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ダミーデータ生成関数（32フレームの加速度データ）
def generate_single_imu_data():
    stable = np.random.normal(0, 0.05, (32, 3))   # stableなデータ
    walk = np.random.normal(1, 0.3, (32, 3))      # 歩行データ
    running = np.random.normal(2, 0.6, (32, 3))   # 走行データ
    return stable, walk, running

# DNNモデルの定義
class IMUClassifier(nn.Module):
    def __init__(self):
        super(IMUClassifier, self).__init__()
        self.fc1 = nn.Linear(32 * 3, 64)  # 入力は32フレーム分のデータ
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 3)  # 出力は3クラス（stable, walk, running）

    def forward(self, x):
        x = x.view(x.size(0), -1)  # (バッチサイズ, 32フレーム * 3軸) に変換
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# ダミーデータ生成
stable_data, walk_data, running_data = generate_single_imu_data()

# テストするための32フレームのサンプルデータを1つ選択
# ここではstableのデータを使用（適宜変更可能）
# sample_data = stable_data  # 例: stableデータ
# label_true = 0  # 正解ラベル（0: stable, 1: walk, 2: running）
sample_data = running_data  # 例: stableデータ
label_true = 0  # 正解ラベル（0: stable, 1: walk, 2: running）

# PyTorchテンソルに変換
sample_tensor = torch.tensor(sample_data, dtype=torch.float32).unsqueeze(0)  # 1サンプル分を入力

# モデルのインスタンス作成（事前に学習済みと仮定）
model = IMUClassifier()

# モデルに32フレームのデータを入力して推論
model.eval()
with torch.no_grad():
    output = model(sample_tensor)
    _, predicted_label = torch.max(output, 1)  # 最大値を持つクラスを予測

# 予測結果の表示
print(f"True Label: {label_true}, Predicted Label: {predicted_label.item()}")

# 視覚化（予測に使用したデータのプロット）
def plot_imu_data(data, label_true, label_pred):
    plt.figure(figsize=(10, 6))
    
    # 時間軸（32フレーム分の時刻）
    time_steps = np.arange(32)
    
    # x, y, z軸の加速度データをプロット
    plt.plot(time_steps, data[:, 0], label='acc_x', color='r')
    plt.plot(time_steps, data[:, 1], label='acc_y', color='g')
    plt.plot(time_steps, data[:, 2], label='acc_z', color='b')
    
    # ラベルの表示
    plt.title(f'True Label: {label_true}, Predicted Label: {label_pred}')
    plt.xlabel('Time steps (32 frames)')
    plt.ylabel('Acceleration')
    plt.legend()
    plt.show()

# データのプロットと予測結果の表示
plot_imu_data(sample_data, label_true, predicted_label.item())