import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# データの読み込み
data_train = pd.read_csv('../input/titanic/train.csv')
data_test = pd.read_csv('../input/titanic/test.csv')
data_gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')

# 教師データとテストデータを結合
data_all = pd.concat([data_train, data_test], sort=False)

# 性別を数値に変換
data_all['Sex'] = data_all['Sex'].replace(['male', 'female'], [0, 1])

# 家族数の特徴量を作成
data_all['Families'] = data_all['Parch'] + data_all['SibSp'] + 1

# 一人かどうかを判定する特徴量を作成
data_all['Alone'] = 0
data_all.loc[data_all['Families'] == 1, 'Alone'] = 1

# 年齢の欠損値を補完
ave = data_all['Age'].mean()
std = data_all['Age'].std()
data_all['Age'] = data_all['Age'].fillna(np.random.randint(ave - std, ave + std))

# 乗船場所の欠損値を補完
data_all['Embarked'] = data_all['Embarked'].fillna('S')
data_all['Embarked'] = data_all['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

# 不要な列を削除
drop_columns = ['PassengerId', 'Name', 'Ticket', 'Cabin']
data_all.drop(drop_columns, axis=1, inplace=True)

# 運賃の欠損値を補完
data_all['Fare'] = data_all['Fare'].fillna(np.mean(data_all['Fare']))

# データを再度分割
data_train = data_all[:len(data_train)]
data_test = data_all[len(data_train):]

# 特徴量と目的変数を分離
y_data_train = data_train['Survived']
X_data_train = data_train.drop('Survived', axis=1)
X_data_test = data_test.drop('Survived', axis=1)

# 学習用データと検証用データに分割
from sklearn.model_selection import train_test_split
X_data_train, X_data_valid, y_data_train, y_data_valid = train_test_split(X_data_train, y_data_train, test_size=0.3, random_state=0, stratify=y_data_train)

# カテゴリ変数を category 型に変換
categorical_features = ['Sex', 'Embarked', 'Pclass']
for feature in categorical_features:
    X_data_train[feature] = X_data_train[feature].astype('category')
    X_data_valid[feature] = X_data_valid[feature].astype('category')
    X_data_test[feature] = X_data_test[feature].astype('category')

# # LightGBMをインポートしてモデル作成
# import lightgbm as lgb
import os
os.environ['LIGHTGBM_RANKING'] = '0'  # 互換モジュールを無効化
import lightgbm as lgb


# lgb_data_train = lgb.Dataset(X_data_train, y_data_train, categorical_feature=categorical_features)
# lgb_data_eval = lgb.Dataset(X_data_valid, y_data_valid, reference=lgb_data_train, categorical_feature=categorical_features)

# params = {
#     'objective': 'binary'
# }

# model = lgb.train(
#     params,
#     lgb_data_train,
#     valid_sets=lgb_data_eval,
#     verbose_eval=10,
#     num_boost_round=1000,
#     early_stopping_rounds=10
# )
