# PyTorchを使ったタイタニック生存者予測モデルを実装し、より精度を高めるために前処理とアンサンブル学習を追加します。以下は、その手順を説明した実装です。

# ### Step 1: 前処理と特徴量エンジニアリング

# まずはデータを読み込んで、特徴量エンジニアリングを行い、PyTorchで扱える形に整えます。

# ```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# データの読み込み
data_train = pd.read_csv('../input/titanic/train.csv')
data_test = pd.read_csv('../input/titanic/test.csv')
data_gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')

# 特徴量エンジニアリング
def preprocess_data(df):
    # 性別を数値に変換
    # df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    # *****
    # Sexの値を男性と女性で分ける&数値に変換
    df['Sex_male'] = df['Sex'].map({'male': 1, 'female': 0})
    df['Sex_female'] = df['Sex'].map({'male': 0, 'female': 1})
    # 性別を数値に変換
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    # # Sexの値を男性と女性で分ける
    # df['Sex_male'] = df['Sex'].replace(['male', 'female'], [1, 0]) # 男性を1, 女性を0に置き換える
    # df['Sex_female'] = df['Sex'].replace(['male', 'female'], [0, 1]) # 男性を0, 女性を1に置き換える
    # # df['Sex'] = df['Sex'].replace(['male', 'female'], [0, 1]) # 既存のデータセットも男性を0, 女性を1に置き換える
    # *****
    
    # 欠損値の補完
    # df['Age'].fillna(df['Age'].mean(), inplace=True)
    # *****
    ave_age = df['Age'].mean()
    std_age = df['Age'].std()
    df['Age'] = df['Age'].fillna(np.random.randint(ave_age - std_age, ave_age + std_age))
    # *****
    
    # 欠損値の補完
    df['Fare'].fillna(df['Fare'].mean(), inplace=True)
    # # *****
    # ave_fare = df['Embarked'].mean()
    # std_fare = df['Embarked'].std()
    df['Embarked'].fillna('S', inplace=True)
    # df['Embarked'] = df['Embarked'].fillna(np.random.randint(ave_fare - std_fare, ave_fare + std_fare))
    # # *****
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    # SibSpとParchを統合して新しい特徴量を作成
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1 # +1は単身者を反映させるため。単身者はParch, SibSPともに0なので、0になってしまうからあえて+1をすることで1人という情報を与えている
    
    # *****
    # Families=1がずば抜け鉄扉ため、別の特徴量として抜き出す
    df['Alone'] = 0 # Alone列が作成され、0で初期化される
    # locを使って各行データ(横)を書き換える
    # 書き換え対象となる「条件式、対象の名前」の順で指定する
    df.loc[df['FamilySize'] == 1, 'Alone'] = 1 # Familiesが1のデータに対してAloneに1を代入する
    # - 条件式：「data_all['Families'] == 1」　, 書き換え対象の列名：「'Alone'」
    # *****
    
    # 不要な列を削除
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    return df

data_train = preprocess_data(data_train)
data_test = preprocess_data(data_test)

# # # モデルに使う特徴量を選択
# features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Families', 'Alone']
# # # X = data_train[features].fillna(0)

# 入力とターゲットに分ける
X = data_train.drop('Survived', axis=1).values
y = data_train['Survived'].values

# 訓練データとテストデータの分割
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 標準化(スケーリング)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# # PyTorchのテンソルに変換
# X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
# y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
# X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
# y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
# ```

# ### Step 2: データセットの定義

# PyTorchの`Dataset`クラスを使ってデータを管理します。

# ```python
class TitanicDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor([self.y[idx]])

train_dataset = TitanicDataset(X_train, y_train)
val_dataset = TitanicDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# ```

# ### Step 3: PyTorchモデルの定義

# ニューラルネットワークをシンプルに定義します。

# ```python
# class TitanicModel(nn.Module):
#     def __init__(self):
#         super(TitanicModel, self).__init__()
#         self.fc1 = nn.Linear(X_train.shape[1], 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.fc3 = nn.Linear(32, 1)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.sigmoid(self.fc3(x))
#         return x


# import torch
# import torch.nn as nn

# class ImprovedTitanicModel(nn.Module):
#     def __init__(self, input_size):
#         super(ImprovedTitanicModel, self).__init__()
#         # Assuming a sequence-like structure in the input
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
#         self.fc1 = nn.Linear(32 * input_size, 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.fc3 = nn.Linear(32, 1)
        
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#         self.batchnorm1 = nn.BatchNorm1d(16)
#         self.batchnorm2 = nn.BatchNorm1d(32)
#         self.dropout = nn.Dropout(p=0.5)

#     def forward(self, x):
#         # Assuming x is [batch_size, seq_len, features], we add a channel dimension
#         x = x.unsqueeze(1)  # Convert to [batch_size, 1, seq_len]
        
#         # Convolutional layers with batch normalization and ReLU
#         x = self.conv1(x)
#         x = self.batchnorm1(x)
#         x = self.relu(x)
        
#         x = self.conv2(x)
#         x = self.batchnorm2(x)
#         x = self.relu(x)
        
#         # Flatten the output for the fully connected layers
#         x = x.view(x.size(0), -1)
        
#         # Fully connected layers with dropout and ReLU
#         x = self.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.relu(self.fc2(x))
#         x = self.dropout(x)
        
#         # Output layer with sigmoid for binary classification
#         x = self.sigmoid(self.fc3(x))
#         return x
import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedTitanicModel(nn.Module):
    def __init__(self, input_dim):
        super(ImprovedTitanicModel, self).__init__()
        # 畳み込み層（入力次元が1次元の場合）
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        
        # 全結合層
        self.fc1 = nn.Linear(32 * input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
        # Batch Normalization
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = x.unsqueeze(1)  # 1D畳み込み用に次元を追加
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x


# model = TitanicModel()
model = ImprovedTitanicModel(X_train.shape[1])
# ```

# ### Step 4: 学習と検証のループ

# 損失関数と最適化関数を定義してモデルを訓練します。

# ```python
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=100): # 20):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {running_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}')

train_model(model, train_loader, val_loader, criterion, optimizer)
# ```

# ### Step 5: アンサンブル学習の追加

# ランダムフォレストや勾配ブースティングを使ったアンサンブルを追加します。

# ```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# PyTorchモデルによる予測
model.eval()
with torch.no_grad():
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_pred_val = model(X_val_tensor).squeeze().numpy()
    # for inputs, _ in val_loader: # labels
    #     y_pred_val = model(inputs)
    y_pred_val = (y_pred_val >= 0.5).astype(int)  # 閾値 0.5
    print(f'PyTorch Model Accuracy: {accuracy_score(y_val, y_pred_val)}')

# ランダムフォレスト
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_val)
print(f'Random Forest Accuracy: {accuracy_score(y_val, rf_preds)}')

# 勾配ブースティング
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
gb_preds = gb_model.predict(X_val)
print(f'Gradient Boosting Accuracy: {accuracy_score(y_val, gb_preds)}')

# # アンサンブル（単純平均）
# ensemble_preds = (rf_preds + gb_preds) / 2
# ensemble_preds = np.where(ensemble_preds > 0.5, 1, 0)
# print(f'Ensemble Accuracy: {accuracy_score(y_val, ensemble_preds)}')

# アンサンブル (平均)
y_pred_ensemble = (y_pred_val + rf_preds + gb_preds) // 3  # 平均を取り、0か1に丸める
# ensemble_preds = np.where(y_pred_ensemble > 0.5, 1, 0)

# 評価 (accuracy)
from sklearn.metrics import accuracy_score
acc_pytorch = accuracy_score(y_val, y_pred_val)
acc_rf = accuracy_score(y_val, rf_preds)
acc_gb = accuracy_score(y_val, gb_preds)
# acc_ensemble = accuracy_score(y_val, ensemble_preds) # y_pred_ensemble)
acc_ensemble = accuracy_score(y_val, y_pred_ensemble)

print(f'PyTorch Model Accuracy: {acc_pytorch:.4f}')
print(f'Random Forest Accuracy: {acc_rf:.4f}')
print(f'Gradient Boosting Accuracy: {acc_gb:.4f}')
print(f'Ensemble Accuracy: {acc_ensemble:.4f}')

# ```

# ### まとめ

# この実装では、PyTorchのニューラルネットワークに加え、ランダムフォレストと勾配ブースティングのアンサンブル学習を使用しています。これにより、精度が向上しやすくなります。また、特徴量エンジニアリングや前処理も行っています。


# # テストデータに対してアンサンブル予測
# X_test = data_test[features].fillna(0)
# X_test_scaled = scaler.transform(X_test)
# X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
# 入力とターゲットに分ける
X_test = data_test # .drop('Survived', axis=1).values
X_test_scaled = scaler.transform(X_test)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

# PyTorchモデルの予測
model.eval()
with torch.no_grad():
    y_pred_test_pytorch = model(X_test_tensor).squeeze().numpy()
    y_pred_test_pytorch = (y_pred_test_pytorch >= 0.5).astype(int)

# ランダムフォレストの予測
y_pred_test_rf = rf_model.predict(X_test_scaled)

# 勾配ブースティングの予測
y_pred_test_gb = gb_model.predict(X_test_scaled)

# アンサンブル予測
# y_pred_test_ensemble = (y_pred_test_pytorch + y_pred_test_rf) // 2
y_pred_test_ensemble = (y_pred_test_pytorch + y_pred_test_rf + y_pred_test_gb) // 3  # 平均を取り、0か1に丸める

# 結果を保存
# submission = pd.DataFrame({'PassengerId': data_test['PassengerId'], 'Survived': y_pred_test_ensemble})
# submission.to_csv('submission.csv', index=False)

# y_pred_test_ensemble['Survived'] = (y_pred_test_ensemble['Survived'] >= 2).astype(int)
# y_pred_test_ensemble.to_csv('../output/ansemble_submit.csv', index=False)

# # Lesson6 予測
# y_pred_submit = sum(y_pred_test_ensemble) / len(y_pred_test_ensemble)
# print(y_pred_submit)
# y_pred_submit = (y_pred_submit > 0.5).astype(int)
# print(y_pred_submit)

# 予測のリストの長さがX_valと同じか確認
print(f'Length of y_pred_val_pytorch: {len(y_pred_test_ensemble)}')
print(f'Length of X_val: {len(X_val)}')


#予測結果をSubmit用のcsvとして作成
submit = data_gender_submission
submit['Survived'] = list(map(int, y_pred_test_ensemble)) # y_pred_submit))
submit.to_csv('../output/ansemble_submit.csv', index=False)
