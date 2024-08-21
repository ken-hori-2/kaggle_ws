# Part A
# 1.データの加工せずに、そのままモデルにつっこむ →エラー

# 2.最低限の加工をして、モデルを構築しパフォーマンス評価 パラメータはデフォルト

# Part B
# 3.使うデータは２と同様で、パラメータチューニングしパフォーマンス向上を狙う

# add Codeadd Markdown
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgbm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
# %matplotlib inline
# add Codeadd Markdown
# 1.データの加工を一切せずに、モデルにつっこむ¶
# add Codeadd Markdown
# データを読み込む

# train_df = トレーニングデータ、これをもとに、モデルを構築

# test_df = これもとに、予測した結果を提出

# add Codeadd Markdown
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
# add Codeadd Markdown
# トレーニングデータを分割する

# X_train,y_train = 実際にモデル構築に用いるデータ

# X_test,y_test = モデル構築には用いず、パフォーマンスを見るためにとっておくデータ（正解データがある）

# add Codeadd Markdown
# Separate the label
y = train_df.pop('Survived')

# Take a hold out set randomly
X_train, X_test, y_train, y_test = train_test_split(train_df, y, test_size=0.2, random_state=42)

#LighGBMのために、データを加工
lgb_train = lgbm.Dataset(X_train, y_train)
lgb_eval = lgbm.Dataset(X_test, y_test)
# add Codeadd Markdown
# ハイパーパラメータを設定

# ハイパーパラメータ＝人の手で決めるパラメータ

# 今回は、モデルをGBDTに指定して、ほかは全部デフォルト値

# add Codeadd Markdown
#パラメータはGBDTの指定のみ
lgbm_params = {
    'boosting': 'gbdt',          # GBDTを使う
              }
# add Codeadd Markdown
#モデル構築
#model = lgbm.train(params=lgbm_params,train_set=lgb_train)
# add Codeadd Markdown
# エラーが出る

# ValueError: DataFrame.dtypes for data must be int, float or bool.

# Did not expect the data types in the following fields: Name, Sex, Ticket, Cabin, Embarked

# つまり、int,float,boolでないといけない。名前などstring型のデータは変更しないと加工しないといけない

# add Codeadd Markdown
# 2. 最低限のデータ加工を行う
# add Codeadd Markdown
train_df = pd.read_csv("/..input/titanic/train.csv")
test_df = pd.read_csv('/../input/titanic/test.csv')
# train_df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 891 entries, 0 to 890
# Data columns (total 12 columns):
#  #   Column       Non-Null Count  Dtype  
# ---  ------       --------------  -----  
#  0   PassengerId  891 non-null    int64  
#  1   Survived     891 non-null    int64  
#  2   Pclass       891 non-null    int64  
#  3   Name         891 non-null    object 
#  4   Sex          891 non-null    object 
#  5   Age          714 non-null    float64
#  6   SibSp        891 non-null    int64  
#  7   Parch        891 non-null    int64  
#  8   Ticket       891 non-null    object 
#  9   Fare         891 non-null    float64
#  10  Cabin        204 non-null    object 
#  11  Embarked     889 non-null    object 
# dtypes: float64(2), int64(5), object(5)
# memory usage: 83.7+ KB
# add Codeadd Markdown
# ユニークな値の数をチェック

# 種類が多いものは、意味がないので削除

# add Codeadd Markdown
for name in train_df.columns:
    print(name,end=" ")
    print(len(train_df[name].unique()))
# PassengerId 891
# Survived 2
# Pclass 3
# Name 891
# Sex 2
# Age 89
# SibSp 7
# Parch 7
# Ticket 681
# Fare 248
# Cabin 148
# Embarked 4
# add Codeadd Markdown
#名前、チケット、Cabinは、関係なさそうなので、とりあえず、削除
df_list=[train_df,test_df]
for df in df_list:
    df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
# Passenger ID も削除、ただ、提出用にtest_dfのPassenger IDは保存
test_passenger_ids = test_df.pop('PassengerId')
train_df.drop(['PassengerId'], axis=1, inplace=True)
# add Codeadd Markdown
# 先ほどのエラー原因を解決するために、カテゴリ変数を変換

# add Codeadd Markdown
#Sexを数字に変換
for df in df_list:
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le = le.fit(df["Sex"])
    df["Sex"]=le.transform(df["Sex"])
# add Codeadd Markdown
#Embarkedを数字に変換
le = LabelEncoder()
#le = le.fit(train_df["Embarked"])
#train_df["Embarked"]=le.transform(train_df["Embarked"])
# add Codeadd Markdown
# エラーが出る

# TypeError: Encoders require their input to be uniformly strings or numbers. Got ['float', 'str']

# 複数のデータ型があると、encoderが使えない

# 今回は、欠損値があるため、データ型が複数になっている

# add Codeadd Markdown
#Encoderを使うには、NaNを埋め、データの型を揃える必要がある
#Embarkedを数字に変換
print(train_df["Embarked"].unique())
#NaNを埋める必要がある
for df in df_list:
    df["Embarked"]=df["Embarked"].fillna("N")
    le = LabelEncoder()
    le = le.fit(df["Embarked"])
    df["Embarked"]=le.transform(df["Embarked"])
print(train_df["Embarked"].unique())
# ['S' 'C' 'Q' nan]
# [3 0 2 1]
# add Codeadd Markdown
# トレーニングデータを分割する

# y = 正解データ

# X_train,y_train = 実際にモデル構築に用いるデータ

# X_test,y_test = モデル構築には用いず、パフォーマンスを見るためにとっておくデータ（正解データがある）

# add Codeadd Markdown
# Separate the label
y = train_df.pop('Survived')

# Take a hold out set randomly
X_train, X_test, y_train, y_test = train_test_split(train_df, y, test_size=0.2, random_state=42)
# add Codeadd Markdown
# LGBM用のデータセットに加工

# add Codeadd Markdown
# Create an LGBM dataset for training
categorical_features = ['Sex', 'Pclass', 'Embarked']
train_data = lgbm.Dataset(data=X_train, label=y_train, categorical_feature=categorical_features, free_raw_data=False)
# Create an LGBM dataset from the test
test_data = lgbm.Dataset(data=X_test, label=y_test, categorical_feature=categorical_features, free_raw_data=False)
# Finally, create a dataset for the FULL training data to give us maximum amount of data to train on after 
# performance has been calibrate
final_train_set = lgbm.Dataset(data=train_df, label=y, 
                              categorical_feature=categorical_features, free_raw_data=False)

# add Codeadd Markdown
#モデル構築
model = lgbm.train(params=lgbm_params,train_set=final_train_set)
#パラメータはGBDTの指定のみ
lgbm_params = {
    'boosting': 'gbdt',          # GBDTを使う
              }

#モデル構築
model = lgbm.train(params=lgbm_params,train_set=final_train_set)
# [LightGBM] [Warning] Auto-choosing row-wise multi-threading, the overhead of testing was 0.000213 seconds.
# You can set `force_row_wise=true` to remove the overhead.
# And if memory is not enough, you can set `force_col_wise=true`.
# [LightGBM] [Info] Total Bins 224
# [LightGBM] [Info] Number of data points in the train set: 891, number of used features: 7
# [LightGBM] [Info] Start training from score 0.383838
# add Codeadd Markdown
# トレーニングデータから取っておいたテスト用のデータで、パフォーマンス評価

# X_test,y_testで評価

# add Codeadd Markdown
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
preds = np.round(model.predict(X_test))
print('Accuracy score = \t {}'.format(accuracy_score(y_test, preds)))
print('Precision score = \t {}'.format(precision_score(y_test, preds)))
print('Recall score =   \t {}'.format(recall_score(y_test, preds)))
print('F1 score =      \t {}'.format(f1_score(y_test, preds)))
# Accuracy score = 	 0.9273743016759777
# Precision score = 	 0.9420289855072463
# Recall score =   	 0.8783783783783784
# F1 score =      	 0.9090909090909092
# add Codeadd Markdown
# 最終結果

# testデータを用い、予測しPassengerIDと結合し、提出用ファイルを出力

# add Codeadd Markdown
y_pred = np.round(model.predict(test_df)).astype(int)

output_df = pd.DataFrame({'PassengerId': test_passenger_ids, 'Survived': y_pred})

output_df.to_csv("20240817_lgmb_submit.csv",index=False)