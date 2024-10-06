# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.linear_model import LogisticRegression
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler

# # データの前処理 (ここにTitanicデータの読み込みと前処理を入れてください)
# # df = ...
# # データの読み込み
# data_train = pd.read_csv('../input/titanic/train.csv')
# data_test = pd.read_csv('../input/titanic/test.csv')
# data_gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')

# # 特徴量エンジニアリング
# def preprocess_data(df):
#     # 性別を数値に変換
#     # df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
#     # *****
#     # Sexの値を男性と女性で分ける&数値に変換
#     df['Sex_male'] = df['Sex'].map({'male': 1, 'female': 0})
#     df['Sex_female'] = df['Sex'].map({'male': 0, 'female': 1})
#     # 性別を数値に変換
#     df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
#     # # Sexの値を男性と女性で分ける
#     # df['Sex_male'] = df['Sex'].replace(['male', 'female'], [1, 0]) # 男性を1, 女性を0に置き換える
#     # df['Sex_female'] = df['Sex'].replace(['male', 'female'], [0, 1]) # 男性を0, 女性を1に置き換える
#     # # df['Sex'] = df['Sex'].replace(['male', 'female'], [0, 1]) # 既存のデータセットも男性を0, 女性を1に置き換える
#     # *****
    
#     # 欠損値の補完
#     # df['Age'].fillna(df['Age'].mean(), inplace=True)
#     # *****
#     ave_age = df['Age'].mean()
#     std_age = df['Age'].std()
#     df['Age'] = df['Age'].fillna(np.random.randint(ave_age - std_age, ave_age + std_age))
#     # *****
    
#     # 欠損値の補完
#     df['Fare'].fillna(df['Fare'].mean(), inplace=True)
#     # # *****
#     # ave_fare = df['Embarked'].mean()
#     # std_fare = df['Embarked'].std()
#     df['Embarked'].fillna('S', inplace=True)
#     # df['Embarked'] = df['Embarked'].fillna(np.random.randint(ave_fare - std_fare, ave_fare + std_fare))
#     # # *****
#     df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
#     # SibSpとParchを統合して新しい特徴量を作成
#     df['FamilySize'] = df['SibSp'] + df['Parch'] + 1 # +1は単身者を反映させるため。単身者はParch, SibSPともに0なので、0になってしまうからあえて+1をすることで1人という情報を与えている
    
#     # *****
#     # Families=1がずば抜け鉄扉ため、別の特徴量として抜き出す
#     df['Alone'] = 0 # Alone列が作成され、0で初期化される
#     # locを使って各行データ(横)を書き換える
#     # 書き換え対象となる「条件式、対象の名前」の順で指定する
#     df.loc[df['FamilySize'] == 1, 'Alone'] = 1 # Familiesが1のデータに対してAloneに1を代入する
#     # - 条件式：「data_all['Families'] == 1」　, 書き換え対象の列名：「'Alone'」
#     # *****
    
#     # 不要な列を削除
#     df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
#     return df

# data_train = preprocess_data(data_train)
# data_test = preprocess_data(data_test)

# # # # モデルに使う特徴量を選択
# # features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Families', 'Alone']
# # # # X = data_train[features].fillna(0)

# # 入力とターゲットに分ける
# X = data_train.drop('Survived', axis=1).values
# y = data_train['Survived'].values
# # # 特徴量とラベルに分ける
# # X = df.drop('Survived', axis=1).values
# # y = df['Survived'].values

# # 学習データと検証データに分割
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# # # 標準化(スケーリング)
# # scaler = StandardScaler()
# # X_train = scaler.fit_transform(X_train)
# # X_val = scaler.transform(X_val)

# # PyTorch用データローダの作成
# train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
# val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

# # 改善されたモデル
# class ImprovedTitanicModel(nn.Module):
#     def __init__(self, input_size):
#         super(ImprovedTitanicModel, self).__init__()
#         self.fc1 = nn.Linear(input_size, 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.fc3 = nn.Linear(32, 2)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.5)
#         self.batchnorm1 = nn.BatchNorm1d(64)
#         self.batchnorm2 = nn.BatchNorm1d(32)
        
#     def forward(self, x):
#         x = self.relu(self.batchnorm1(self.fc1(x)))
#         x = self.dropout(x)
#         x = self.relu(self.batchnorm2(self.fc2(x)))
#         x = self.dropout(x)
#         x = self.fc3(x)
#         return x

#     # Xavier初期化
#     def init_weights(self, layer):
#         if isinstance(layer, nn.Linear):
#             torch.nn.init.xavier_uniform_(layer.weight)
#             if layer.bias is not None:
#                 layer.bias.data.fill_(0.01)

# # モデル初期化
# model = ImprovedTitanicModel(X_train.shape[1])
# model.apply(model.init_weights)

# # 損失関数とオプティマイザの設定 (AdamWを使用)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.AdamW(model.parameters(), lr=0.001)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# # 学習ループ
# epochs = 300 # 50
# for epoch in range(epochs):
#     model.train()
#     train_loss = 0.0
#     for inputs, labels in train_loader:
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()

#     val_loss = 0.0
#     model.eval()
#     with torch.no_grad():
#         for inputs, labels in val_loader:
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             val_loss += loss.item()

#     # スケジューラで学習率を調整
#     scheduler.step(val_loss)

#     print(f'Epoch {epoch+1}/{epochs}, Training Loss: {train_loss/len(train_loader)}, Validation Loss: {val_loss/len(val_loader)}')

# # PyTorchモデルの予測
# model.eval()
# with torch.no_grad():
#     y_pred_val_pytorch = []
#     for inputs, _ in val_loader:
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs, 1)
#         y_pred_val_pytorch.extend(predicted.tolist())

# # PyTorchモデルの精度
# accuracy_pytorch = accuracy_score(y_val, y_pred_val_pytorch)
# print(f'PyTorch Model Accuracy: {accuracy_pytorch}')

# # -----------------------------
# # ランダムフォレストと勾配ブースティングの学習
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# gb = GradientBoostingClassifier(n_estimators=100, random_state=42)

# rf.fit(X_train, y_train)
# gb.fit(X_train, y_train)

# # 検証データに対する予測
# rf_preds = rf.predict(X_val)
# gb_preds = gb.predict(X_val)

# # -----------------------------
# # Stacking (メタ学習器としてロジスティック回帰を使用)
# meta_model = LogisticRegression()
# stacked_predictions = np.column_stack((y_pred_val_pytorch, rf_preds, gb_preds))
# meta_model.fit(stacked_predictions, y_val)

# # スタッキングアンサンブルの予測
# y_pred_stacked = meta_model.predict(stacked_predictions)

# # アンサンブルモデルの精度
# accuracy_stacked = accuracy_score(y_val, y_pred_stacked)
# print(f'Stacking Ensemble Accuracy: {accuracy_stacked}')








import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# データの前処理
data_train = pd.read_csv('../input/titanic/train.csv')
data_test = pd.read_csv('../input/titanic/test.csv')
data_gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')

# 特徴量エンジニアリング
def preprocess_data(df):
    df['Sex_male'] = df['Sex'].map({'male': 1, 'female': 0})
    df['Sex_female'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    ave_age = df['Age'].mean()
    std_age = df['Age'].std()
    df['Age'] = df['Age'].fillna(np.random.randint(ave_age - std_age, ave_age + std_age))
    
    df['Fare'].fillna(df['Fare'].mean(), inplace=True)
    df['Embarked'].fillna('S', inplace=True)
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['Alone'] = 0
    df.loc[df['FamilySize'] == 1, 'Alone'] = 1
    
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    return df

data_train = preprocess_data(data_train)
data_test = preprocess_data(data_test)

# 入力とターゲットに分ける
X = data_train.drop('Survived', axis=1).values
y = data_train['Survived'].values

# 学習データと検証データに分割
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 標準化(スケーリング)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# PyTorch用データローダの作成
train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

# class TitanicDataset(Dataset):
#     def __init__(self, X, y):
#         self.X = X
#         self.y = y

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, idx):
#         return torch.FloatTensor(self.X[idx]), torch.FloatTensor([self.y[idx]])

# train_dataset = TitanicDataset(X_train, y_train)
# val_dataset = TitanicDataset(X_val, y_val)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 改善されたモデル
# class ImprovedTitanicModel(nn.Module):
#     def __init__(self, input_size):
#         super(ImprovedTitanicModel, self).__init__()
#         self.fc1 = nn.Linear(input_size, 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.fc3 = nn.Linear(32, 2)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.5)
#         self.batchnorm1 = nn.BatchNorm1d(64)
#         self.batchnorm2 = nn.BatchNorm1d(32)
        
#     def forward(self, x):
#         x = self.relu(self.batchnorm1(self.fc1(x)))
#         x = self.dropout(x)
#         x = self.relu(self.batchnorm2(self.fc2(x)))
#         x = self.dropout(x)
#         x = self.fc3(x)
#         return x

#     # Xavier初期化
#     def init_weights(self, layer):
#         if isinstance(layer, nn.Linear):
#             torch.nn.init.xavier_uniform_(layer.weight)
#             if layer.bias is not None:
#                 layer.bias.data.fill_(0.01)

import torch.nn.functional as F
class ImprovedTitanicModel(nn.Module):
    def __init__(self, input_dim):
        super(ImprovedTitanicModel, self).__init__()
        # 畳み込み層（入力次元が1次元の場合）
        # self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        
        # 全結合層
        self.fc1 = nn.Linear(32 * input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        # self.fc3 = nn.Linear(64, 1)
        self.fc3 = nn.Linear(64, 2)
        
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
        # x = torch.sigmoid(self.fc3(x))
        x = self.fc3(x)
        return x
    
    # Xavier初期化
    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                layer.bias.data.fill_(0.01)

# モデル初期化
model = ImprovedTitanicModel(X_train.shape[1])
model.apply(model.init_weights)

# 損失関数とオプティマイザの設定 (AdamWを使用)
criterion = nn.CrossEntropyLoss()
# optimizer = optim.AdamW(model.parameters(), lr=0.0001) # 0.001)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# # 学習ループ
# epochs = 300
# for epoch in range(epochs):
#     model.train()
#     train_loss = 0.0
#     for inputs, labels in train_loader:
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()

#     val_loss = 0.0
#     model.eval()
#     with torch.no_grad():
#         for inputs, labels in val_loader:
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             val_loss += loss.item()

#     # スケジューラで学習率を調整
#     scheduler.step(val_loss)

#     print(f'Epoch {epoch+1}/{epochs}, Training Loss: {train_loss/len(train_loader)}, Validation Loss: {val_loss/len(val_loader)}')

best_val_loss = float('inf')
early_stopping_counter = 0
early_stopping_patience = 10
epochs = 100
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    # スケジューラで学習率を調整
    scheduler.step(val_loss)
    
    # # Early Stopping の条件
    # if val_loss < best_val_loss:
    #     best_val_loss = val_loss
    #     early_stopping_counter = 0  # Reset counter
    # else:
    #     early_stopping_counter += 1
    #     if early_stopping_counter >= early_stopping_patience:
    #         print("Early stopping")
    #         break
    
    print(f'Epoch {epoch+1}/{epochs}, Training Loss: {train_loss/len(train_loader)}, Validation Loss: {val_loss/len(val_loader)}')

# PyTorchモデルの予測
model.eval()
with torch.no_grad():
    y_pred_val_pytorch = []
    for inputs, _ in val_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        y_pred_val_pytorch.extend(predicted.tolist())

# PyTorchモデルの精度
accuracy_pytorch = accuracy_score(y_val, y_pred_val_pytorch)
print(f'PyTorch Model Accuracy: {accuracy_pytorch}')

# -----------------------------
# ランダムフォレストと勾配ブースティングの学習
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)

rf.fit(X_train, y_train)
gb.fit(X_train, y_train)

# 検証データに対する予測
rf_preds = rf.predict(X_val)
gb_preds = gb.predict(X_val)

# -----------------------------
# # Stacking (メタ学習器としてロジスティック回帰を使用)
# meta_model = LogisticRegression()
# stacked_predictions = np.column_stack((y_pred_val_pytorch, rf_preds, gb_preds))
# meta_model.fit(stacked_predictions, y_val)

# # スタッキングアンサンブルの予測
# y_pred_stacked = meta_model.predict(stacked_predictions)

# # アンサンブルモデルの精度
# accuracy_stacked = accuracy_score(y_val, y_pred_stacked)
# print(f'Stacking Ensemble Accuracy: {accuracy_stacked}')

import xgboost as xgb
import lightgbm as lgb

# -----------------------------
# XGBoostモデルの学習
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, learning_rate=0.05, max_depth=3)
xgb_model.fit(X_train, y_train)

# LightGBMモデルの学習
lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42, learning_rate=0.05, max_depth=3)
lgb_model.fit(X_train, y_train)

# -----------------------------
# 予測 (検証データに対する予測)
xgb_preds = xgb_model.predict(X_val)
lgb_preds = lgb_model.predict(X_val)

# -----------------------------
# Stacking Ensemble
meta_model = LogisticRegression()

# 既存のPyTorchモデル、Random Forest、Gradient BoostingにXGBoostとLightGBMを追加
stacked_predictions = np.column_stack((y_pred_val_pytorch, rf_preds, gb_preds)) # , xgb_preds, lgb_preds))
meta_model.fit(stacked_predictions, y_val)

# アンサンブルモデルの予測
y_pred_stacked = meta_model.predict(stacked_predictions)
print("y_pred_stacked:", y_pred_stacked)

# アンサンブルモデルの精度
accuracy_stacked = accuracy_score(y_val, y_pred_stacked)
print(f'Stacking Ensemble Accuracy with XGBoost and LightGBM: {accuracy_stacked}')






# # 入力とターゲットに分ける
# X_test = data_test # .drop('Survived', axis=1).values
# X_test_scaled = scaler.transform(X_test)
# X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

# # # PyTorchモデルの予測
# # model.eval()
# # with torch.no_grad():
# #     y_pred_test_pytorch = model(X_test_tensor).squeeze().numpy()
# #     y_pred_test_pytorch = (y_pred_test_pytorch >= 0.5).astype(int)
# # # ランダムフォレストの予測
# # y_pred_test_rf = rf_model.predict(X_test_scaled)
# # # 勾配ブースティングの予測
# # y_pred_test_gb = gb_model.predict(X_test_scaled)

# y_pred_test_ensamble = meta_model.predict(X_test_scaled)

# **********
# テストデータの前処理
# X_test = preprocess_data(data_test).values
X_test = data_test

# テストデータを標準化
X_test = scaler.transform(X_test)

# PyTorchモデルでの予測
test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

model.eval()
y_pred_test_pytorch = []
with torch.no_grad():
    for inputs in test_loader:
        inputs = inputs[0]  # テンソルの中から入力データを取得
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        y_pred_test_pytorch.extend(predicted.tolist())

# RandomForestでの予測
rf_preds_test = rf.predict(X_test)

# GradientBoostingでの予測
gb_preds_test = gb.predict(X_test)

# XGBoostでの予測
xgb_preds_test = xgb_model.predict(X_test)

# LightGBMでの予測
lgb_preds_test = lgb_model.predict(X_test)

# スタッキングのための予測結果の組み合わせ
stacked_test_predictions = np.column_stack((y_pred_test_pytorch, rf_preds_test, gb_preds_test, xgb_preds_test, lgb_preds_test))

# メタモデルで最終的な予測を行う
y_pred_test_stacked = meta_model.predict(stacked_test_predictions)

# 最終予測結果を表示
print(f'Final Test Predictions: {y_pred_test_stacked}')

# 必要に応じて予測結果を保存 (例: CSV形式で保存)
submit = pd.DataFrame({
    'PassengerId': data_test['PassengerId'],
    'Survived': y_pred_test_stacked
})

submit.to_csv('titanic_submission.csv', index=False)
# **********

# 予測のリストの長さがX_valと同じか確認
print(f'Length of y_pred_val_pytorch: {len(y_pred_test_stacked)}')
print(f'Length of X_val: {len(X_val)}')


#予測結果をSubmit用のcsvとして作成
submit = data_gender_submission
submit['Survived'] = list(map(int, y_pred_test_stacked))
submit.to_csv('../output/ansemble_submit_by_5models.csv', index=False)