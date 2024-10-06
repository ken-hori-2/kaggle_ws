import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ダミーデータ生成関数（32フレームずつ）
def generate_imu_data(num_samples):
    # 各クラスの加速度データを生成
    stable = np.random.normal(0, 0.05, (num_samples, 32, 3))
    walk = np.random.normal(1, 0.3, (num_samples, 32, 3))
    running = np.random.normal(2, 0.6, (num_samples, 32, 3))
    
    # ラベル（0 = stable, 1 = walk, 2 = running）
    labels_stable = np.zeros(num_samples)
    labels_walk = np.ones(num_samples)
    labels_running = np.full(num_samples, 2)
    
    # データとラベルを結合
    data = np.concatenate([stable, walk, running], axis=0)
    labels = np.concatenate([labels_stable, labels_walk, labels_running], axis=0)
    
    return data, labels

# モデル定義（32フレームのデータを入力する）
class IMUClassifier(nn.Module):
    def __init__(self):
        super(IMUClassifier, self).__init__()
        self.fc1 = nn.Linear(32 * 3, 64)  # 32フレーム分の加速度データ（32 x 3 = 96次元）
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 3)  # 3クラス（stable, walk, running）

    def forward(self, x):
        x = x.view(x.size(0), -1)  # (バッチサイズ, 32フレーム * 3次元) の形に変換
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# 学習済みモデルの保存ファイル名
MODEL_PATH = "imu_classifier.pth"

# # データ生成
# num_samples = 1000
# data, labels = generate_imu_data(num_samples)

# # データ分割
# X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

# # PyTorchテンソルに変換
# X_train = torch.tensor(X_train, dtype=torch.float32)
# X_test = torch.tensor(X_test, dtype=torch.float32)
# y_train = torch.tensor(y_train, dtype=torch.long)
# y_test = torch.tensor(y_test, dtype=torch.long)

# # モデルの初期化
# model = IMUClassifier()

# # 損失関数と最適化アルゴリズム
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # トレーニングループ
# num_epochs = 100
# batch_size = 32

# for epoch in range(num_epochs):
#     model.train()
#     permutation = torch.randperm(X_train.size(0))  # シャッフル
#     for i in range(0, X_train.size(0), batch_size):
#         indices = permutation[i:i+batch_size]
#         batch_X, batch_y = X_train[indices], y_train[indices]
        
#         # 順伝播
#         outputs = model(batch_X)
#         loss = criterion(outputs, batch_y)
        
#         # 逆伝播と最適化
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
    
#     if (epoch + 1) % 10 == 0:
#         print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# # モデルの保存
# torch.save(model.state_dict(), MODEL_PATH)
# print(f"Model saved to {MODEL_PATH}")

# # テスト
# model.eval()
# with torch.no_grad():
#     test_outputs = model(X_test)
#     _, predicted = torch.max(test_outputs, 1)
#     accuracy = accuracy_score(y_test, predicted)
#     print(f'Test Accuracy: {accuracy * 100:.2f}%')

# ---- 学習済みモデルを使用して予測 ---- #

# モデルのロード
model_loaded = IMUClassifier()
model_loaded.load_state_dict(torch.load(MODEL_PATH))
model_loaded.eval()
print("Model loaded and ready for prediction.")

# ダミーデータ生成（新しい32フレームのデータ）
def generate_single_imu_data():
    stable = np.random.normal(0, 0.05, (32, 3))   # stableなデータ
    walk = np.random.normal(1, 0.3, (32, 3))      # 歩行データ
    running = np.random.normal(2, 0.6, (32, 3))   # 走行データ
    return stable, walk, running

stable_data, walking_data, running_data = generate_single_imu_data()

# ここでは例として stable_data を使って推論を行う
sample_data = stable_data  # stableデータを予測
sample_data = walking_data
# sample_data = running_data
# label_true = 0  # 正解ラベルはstable（0）


# Action = 'running_data'
Action = 'walking_data'
# Action = 'stable_data'
# Numと文字列の対応付け辞書を作成
mapping_dict = {
    0: 'stable_data',
    1: 'walking_data',
    2: 'running_data'
}
# 'running'に対応する数値を取得
num = [key for key, value in mapping_dict.items() if value == Action][0]
label_true = num # mapping_dict['running_data']  # 正解ラベルはstable（0）
print("data : ", sample_data)

# PyTorchテンソルに変換
sample_tensor = torch.tensor(sample_data, dtype=torch.float32).unsqueeze(0)  # 1サンプル分

# モデルに32フレームのデータを入力して推論
with torch.no_grad():
    output = model_loaded(sample_tensor)
    _, predicted_label = torch.max(output, 1)

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