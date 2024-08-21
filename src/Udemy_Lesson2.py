import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

for dirname, _, filenames in os.walk('/kaggle_ws/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#データの読み込みはPandasの「read_csv()」
data_train = pd.read_csv('../input/titanic/train.csv')
data_test = pd.read_csv('../input/titanic/test.csv')
data_gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')


#教師データとテストデータ、双方にエンジニアリングを行うため、一旦ひとつに結合
data_all = pd.concat([data_train, data_test], sort=False)

#Sexの値を数値に置き換え
# data_all['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
data_all['Sex'] = data_all['Sex'].replace(['male', 'female'], [0, 1])

#Embarkedの欠損値を補完し、数値に置き換える
# data_all['Embarked'].fillna('S', inplace=True)
# data_all['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2], inplace=True)

data_all['Embarked'] = data_all['Embarked'].fillna('S')
data_all['Embarked'] = data_all['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2])
# どのように補完するかは、本来ならいろいろ試して上で精度を元に確認する必要がある
# 今回は欠損値は二件しかない.
# なので最もデータ件数の多い'S'の値で補完
#   - なぜ？: データが多いほど一見あたりのデータの重みは小さいから？
# 欠損の補完はfillna()
# 補完が完了したら、0, 1, 2の値に変更する
# /tmp/ipykernel_18/1626234391.py:6: FutureWarning: Downcasting behavior in `replace` is deprecated and will be removed in a future version. To retain the old behavior, explicitly call `result.infer_objects(copy=False)`. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
#   data_all['Embarked'] = data_all['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2])

data_all['Fare'] = data_all['Fare'].fillna(np.mean(data_all['Fare']))

# 2. 列全体に対して処理を行う方法:
# data_all.fillna({'Fare': np.mean(data_all['Fare'])}, inplace=True)
#Ageの欠損値を平均値で補完する
# data_all['Age'].fillna(np.mean(data_all['Age']), inplace=True)
data_all['Age'] = data_all['Age'].fillna(np.mean(data_all['Age']))

# Ageは欠損値も多いので使わないという手もある.(Agenの列自体を削除)
# 欠損値を含む行のみ削除する
# もしくは今回のように平均値や中央値野や最頻値で埋める
#影響の少なそうな特徴量は一旦削除する
drop_columns = ['PassengerId', 'Name', 'Parch', 'SibSp', 'Ticket', 'Cabin']
data_all.drop(drop_columns, axis=1, inplace=True)

# 削除対象をリストにまとめる
# axis=1は列方向に削除
#再確認
print(data_all.isnull().sum())


#################
# データセット作成 #
#################
#結合したデータを、再度教師データとテストデータに分離
data_train = data_all[:len(data_train)]
data_test = data_all[len(data_train):]



############
# モデル作成 #
############

#アルゴリズムに投入するため、特徴量と目的変数を分離
y_data_train = data_train['Survived']
X_data_train = data_train.drop('Survived', axis=1)
X_data_test = data_test.drop('Survived', axis=1)



#################################
# Udemy_Lesson1.pyとの違いのみ記載 #
#################################


# Lesson2ではロジスティック回帰を使用
# Logistic回帰アルゴリズムをインポート
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(penalty='l2', solver='sag', random_state=0)

# ロジスティック回帰:線形回帰の出力を0~1の確率に変換したもの
# 2値分類に特化したアルゴリズム


#教師データの学習
clf.fit(X_data_train, y_data_train)

# ポイントは説明変数と目的変数を別の引数として指定すること

#作成したモデルで予測
#predictの閾値は0.5がデフォルトなので、0.5以上を1、未満を0として返す
y_data_pred = clf.predict(X_data_test)

#予測結果をSubmit用のcsvとして作成
submit = data_gender_submission
submit['Survived'] = list(map(int, y_data_pred))
submit.to_csv('logisticRegression_submit.csv', index=False)

# map(): 指定した要素に対して、1つ1つ関数を適用できる関数
# 1つ目の引数が適用したい関数、2つ目の引数が要素を持つ変数
# map(int, y_data_pred): 全ての予測結果に対してint()をしている




#################
# モデルの選定方法 #
#################

"""
多くの場合において、アルゴリズムを選ぶ判断軸は以下

数値による回帰や2値分類問題:

まずロジスティック回帰を行い、今後の改善に向けた指針を立てる 多値分類(3つ以上の分類):
決定木やランダムフォレスト
なぜこれらを最初に試すか？

それなりの精度が出ることが経験則で判明しており、結果の解釈もしやすいから
"""