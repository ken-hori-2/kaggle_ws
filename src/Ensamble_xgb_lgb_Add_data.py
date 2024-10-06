
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

# データの前処理
data_train = pd.read_csv('../input/titanic/train.csv')
data_test = pd.read_csv('../input/titanic/test.csv')
data_gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')

import pandas as pd
ex = pd.read_csv("../input/additional-input/submission.csv")

# data_test["Survived"] = ex["Survived"]
# combined_df = pd.concat([data_train, data_test], ignore_index=True)
# X = combined_df.drop('Survived', axis = 1)
# y = combined_df['Survived']

# X_test = data_test.copy() # このままだと、surviedも含まれるので、x, yに分割する
# y_test = X_test['Survived'] # y_test = X_test['Survived']
# X_test = X_test.drop('Survived', 1)

# 特徴量エンジニアリング
def preprocess_data(df):
    df['Sex_male'] = df['Sex'].map({'male': 1, 'female': 0})
    df['Sex_female'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    ave_age = df['Age'].mean()
    std_age = df['Age'].std()
    df['Age'] = df['Age'].fillna(np.random.randint(ave_age - std_age, ave_age + std_age))
    
#     df['Fare'].fillna(df['Fare'].mean(), inplace=True)
#     df['Embarked'].fillna('S', inplace=True)
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['Alone'] = 0
    df.loc[df['FamilySize'] == 1, 'Alone'] = 1
    
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    return df

data_train = preprocess_data(data_train)
data_test = preprocess_data(data_test)






# 追加データの結合
data_test["Survived"] = ex["Survived"]
combined_df = pd.concat([data_train, data_test], ignore_index=True)
X = combined_df.drop('Survived', axis = 1).values
y = combined_df['Survived'].values
# テストデータ
X_test = data_test.copy() # このままだと、surviedも含まれるので、x, yに分割する
y_test = X_test['Survived'].values # y_test = X_test['Survived']
X_test = X_test.drop('Survived', axis = 1).values

# # 入力とターゲットに分ける
# X = data_train.drop('Survived', axis=1).values
# y = data_train['Survived'].values






# 学習データと検証データに分割
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.008, random_state=0)

# 標準化(スケーリング)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# -----------------------------
# # ランダムフォレストと勾配ブースティングの学習
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# gb = GradientBoostingClassifier(n_estimators=100, random_state=42)

# rf.fit(X_train, y_train)
# gb.fit(X_train, y_train)

# # 検証データに対する予測
# rf_preds = rf.predict(X_val)
# gb_preds = gb.predict(X_val)

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
# stacked_predictions = np.column_stack((rf_preds, gb_preds, xgb_preds, lgb_preds))
stacked_predictions = np.column_stack((xgb_preds, lgb_preds))
meta_model.fit(stacked_predictions, y_val)

# アンサンブルモデルの予測
y_pred_stacked = meta_model.predict(stacked_predictions)

# アンサンブルモデルの精度
accuracy_stacked = accuracy_score(y_val, y_pred_stacked)
print(f'Stacking Ensemble Accuracy with XGBoost and LightGBM: {accuracy_stacked}')


# **********
# テストデータの前処理
# X_test = preprocess_data(data_test).values
# X_test = data_test

# テストデータを標準化
X_test = scaler.transform(X_test)

# # PyTorchモデルでの予測
# test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32))
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# # RandomForestでの予測
# rf_preds_test = rf.predict(X_test)

# # GradientBoostingでの予測
# gb_preds_test = gb.predict(X_test)




# XGBoostでの予測
xgb_preds_test = xgb_model.predict(X_train)
# LightGBMでの予測
lgb_preds_test = lgb_model.predict(X_train)
accuracy_xgb = accuracy_score(y_train, xgb_preds_test)
print(f'xgb Accuracy : {accuracy_xgb}')
accuracy_lgb = accuracy_score(y_train, lgb_preds_test)
print(f'lgb Accuracy : {accuracy_lgb}')


xgb_pred = xgb_model.predict(X_test)
accuracy_xgb = accuracy_score(y_test, xgb_pred)
print(f'XGB Accuracy (test data): {accuracy_xgb}')

lgb_pred = lgb_model.predict(X_test)
accuracy_lgb= accuracy_score(y_test, lgb_pred)
print(f'LightGBM Accuracy (test data): {accuracy_lgb}')

# XGBoostでの予測
xgb_preds_test = xgb_model.predict(X_test)

# LightGBMでの予測
lgb_preds_test = lgb_model.predict(X_test)

# スタッキングのための予測結果の組み合わせ
# stacked_test_predictions = np.column_stack((rf_preds_test, gb_preds_test, xgb_preds_test, lgb_preds_test))
stacked_test_predictions = np.column_stack((xgb_preds_test, lgb_preds_test))

# メタモデルで最終的な予測を行う
y_pred_test_stacked = meta_model.predict(stacked_test_predictions)

# 最終予測結果を表示
print(f'Final Test Predictions: {y_pred_test_stacked}')

# # 予測のリストの長さがX_valと同じか確認
# print(f'Length of y_pred_val_pytorch: {len(y_pred_test_stacked)}')
# print(f'Length of X_val: {len(X_val)}')

Ensamble_pred = meta_model.predict(stacked_test_predictions)
accuracy_Ensamble = accuracy_score(y_test, Ensamble_pred)
print(f'Ensamble Accuracy (test data): {accuracy_Ensamble}')


#予測結果をSubmit用のcsvとして作成
submit = data_gender_submission
submit['Survived'] = list(map(int, y_pred_test_stacked)) # y_pred_submit))
submit.to_csv('ansemble_submit_by_Add_train_data.csv', index=False)