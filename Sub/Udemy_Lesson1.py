# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle_ws/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#パッケージの読み込みは「import」
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#データの読み込みはPandasの「read_csv()」
data_train = pd.read_csv('../input/titanic/train.csv')
data_test = pd.read_csv('../input/titanic/test.csv')
data_gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')


"""
*******************
EDA（探索的データ分析）
*******************
"""
#変数名で実行すれば中身を表示してくれる
#ただし、数が多いと自動で省略される
print(data_train)


#出力行数の確認
print(pd.get_option('display.max_rows'))
#全行数の出力
print(pd.set_option('display.max_rows', 900))

print(data_test.head(10))

print(data_gender_submission.head())

#describe()を使えば、主要な統計指標をまとめて確認できる
print(data_train.describe())



# #データの概要をざっと確認するには「pandas_profiling」が便利
# import pandas_profiling

# #実行はpandas.DataFrameの「profile_report()」
# print(data_train.profile_report())

"""
*******************
各特徴量と目的変数の関係性を確認
*******************
"""
#PclassとSurvivedの件数を棒グラフで可視化
sns.countplot(x='Pclass', hue='Survived', data=data_train)
plt.show()

#SexとSurvivedの件数を棒グラフで可視化
sns.countplot(x='Sex', hue='Survived', data=data_train)
plt.show()

#AgeとSurvivedの件数を棒グラフで可視化
#データ数が多いと非常に見づらい
sns.countplot(x='Age', hue='Survived', data=data_train) # 見づらい
plt.show()

#ヒストグラムで幅を取って確認
plt.hist(data_train.loc[data_train['Survived'] == 0, 'Age'].dropna(), bins=30, alpha=0.7, label='0')
plt.hist(data_train.loc[data_train['Survived'] == 1, 'Age'].dropna(), bins=30, alpha=0.7, label='1')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend(title='Survived')
plt.show()

#SibSpとSurvivedの件数を棒グラフで可視化
sns.countplot(x='SibSp', hue='Survived', data=data_train)
plt.show()

#ParchとSurvivedの件数を棒グラフで可視化
sns.countplot(x='Parch', hue='Survived', data=data_train)
#凡例の位置固定
plt.legend(loc='upper right', title='Survived')
plt.show()

#ヒストグラムで幅を取って確認
#特定の範囲に集まっているときはビンの範囲を指定したほうが見やすい
plt.hist(data_train.loc[data_train['Survived'] == 0, 'Fare'].dropna(), bins=30, alpha=0.7, label='0')
plt.hist(data_train.loc[data_train['Survived'] == 1, 'Fare'].dropna(), bins=30, alpha=0.7, label='1')
plt.xlabel('Fare')
plt.ylabel('Count')
plt.legend(title='Survived')
plt.show()

#再確認
#特定の範囲に集まっているときはビンの範囲を指定したほうが見やすい
plt.hist(data_train.loc[data_train['Survived'] == 0, 'Fare'].dropna(), range=(0,250), bins=20, alpha=0.7, label='0')
plt.hist(data_train.loc[data_train['Survived'] == 1, 'Fare'].dropna(), range=(0,250), bins=20, alpha=0.7, label='1')
plt.xlabel('Fare')
plt.ylabel('Count')
plt.legend(title='Survived')
plt.show()

#EmbarkedとSurvivedの件数を棒グラフで可視化
sns.countplot(x='Embarked', hue='Survived', data=data_train)
plt.show()



# """
# *******************
# 特徴量エンジニアリング
# *******************
# """

# #教師データとテストデータ、双方にエンジニアリングを行うため、一旦ひとつに結合
# data_all = pd.concat([data_train, data_test], sort=False)
# print(data_all)

# #各特徴量の欠損値の数を確認
# #テストデータは「Survived」がないため欠損値としてカウントされている
# print(data_all.isnull().sum())

# #Sexの値を数値に置き換え
# data_all['Sex'].replace(['male', 'female'], [0, 1], inplace=True)

# print(data_all.head())

# #Embarkedの欠損値を補完し、数値に置き換える
# data_all['Embarked'].fillna('S', inplace=True)
# data_all['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2], inplace=True)

# print(data_all.head(900))


# #Fareの欠損値を平均値で補完する
# data_all['Fare'].fillna(np.mean(data_all['Fare']), inplace=True)

# #Ageの欠損値を平均値で補完する
# data_all['Age'].fillna(np.mean(data_all['Age']), inplace=True)

# #影響の少なそうな特徴量は一旦削除する
# drop_columns = ['PassengerId', 'Name', 'Parch', 'SibSp', 'Ticket', 'Cabin']
# data_all.drop(drop_columns, axis=1, inplace=True)

# #再確認
# print(data_all.isnull().sum())

# #結合したデータを、再度教師データとテストデータに分離
# data_train = data_all[:len(data_train)]
# data_test = data_all[len(data_train):]