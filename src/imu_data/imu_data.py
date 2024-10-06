import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# ダミーのIMUセンサーデータを生成する関数
def generate_imu_data(num_samples):
    # 'stable' -> 安定状態の加速度データ（低振動）
    stable = np.random.normal(0, 0.05, (num_samples, 3))  # (num_samples, 3) 次元のデータ
    # 'walk' -> 歩行中の加速度データ（中程度の振動）
    walk = np.random.normal(1, 0.3, (num_samples, 3))
    # 'running' -> 走っているときの加速度データ（大きな振動）
    running = np.random.normal(2, 0.6, (num_samples, 3))
    
    # ラベル（0 = stable, 1 = walk, 2 = running）
    labels_stable = np.zeros(num_samples)
    labels_walk = np.ones(num_samples)
    labels_running = np.full(num_samples, 2)
    
    # データとラベルを結合
    data = np.concatenate([stable, walk, running], axis=0)
    labels = np.concatenate([labels_stable, labels_walk, labels_running], axis=0)
    
    return data, labels

# ハイパーパラメータ
num_samples = 1000  # 各クラスのサンプル数
num_epochs = 100    # エポック数
learning_rate = 0.001

# ダミーデータ生成
data, labels = generate_imu_data(num_samples)

# データをトレーニングセットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

# データをPyTorchテンソルに変換
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# DNNモデルの定義
class IMUClassifier(nn.Module):
    def __init__(self):
        super(IMUClassifier, self).__init__()
        self.fc1 = nn.Linear(3, 64)  # IMUの入力データ（x, y, zの3次元）
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 3)  # 出力は3クラス（stable, walk, running）

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# モデルの初期化
model = IMUClassifier()

# 損失関数と最適化アルゴリズムの設定
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# トレーニングループ
for epoch in range(num_epochs):
    model.train()
    
    # 順伝播
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # 逆伝播と最適化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# モデルのテスト
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = accuracy_score(y_test, predicted)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')





import pandas as pd

# ダミーデータをCSVファイルに保存する関数
def save_data_to_csv(data, labels, filename):
    df = pd.DataFrame(data, columns=['acc_x', 'acc_y', 'acc_z'])  # 加速度データのカラム
    df['label'] = labels  # ラベル（0: stable, 1: walk, 2: running）
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

# トレーニングとテストのデータセットをそれぞれ保存
save_data_to_csv(X_train.numpy(), y_train.numpy(), "train_data.csv")
save_data_to_csv(X_test.numpy(), y_test.numpy(), "test_data.csv")

# モデルのテストと予測結果を保存
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs, 1)

    # テストデータの予測結果と実際のラベルをCSVファイルに保存
    results_df = pd.DataFrame(X_test.numpy(), columns=['acc_x', 'acc_y', 'acc_z'])
    results_df['actual_label'] = y_test.numpy()  # 実際のラベル
    results_df['predicted_label'] = predicted.numpy()  # 予測されたラベル
    results_df.to_csv("test_results.csv", index=False)
    print("Test results saved to test_results.csv")

# 正答率も確認
accuracy = accuracy_score(y_test, predicted)
print(f'Test Accuracy: {accuracy * 100:.2f}%')