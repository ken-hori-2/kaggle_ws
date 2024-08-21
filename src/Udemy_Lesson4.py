# memo
"""
LughtGBM:
    勾配ブースティング法で、大量に決定木を作成しながら学習していくアルゴリズム
    前回の決定木の結果を反映しながら新たな決定木を次々学習していく
    
    強み：データ量が十分であれば精度が高くなりやすい
    弱み：ロジスティック回帰やランダムフォレストよりも処理が重い、過学習になりやすい


過学習について：
    外れ値まで予測モデルに組み込まれてしまうと、実際のデータで予測する際に、外れ値周辺の予測結果が大きく間違ってしまうことがある
    
    発生原因：
        今回のような外れ値を取り除かずにモデルを作成すると、外れ値も学習されてしまい、教師データにのみフィットするようになってしまう
        LightGBMは決定機を順番に学習するので、教師データにフィットして行きやすい...つまり外れ値を含めばそれにもフィットしやすい
    
"""


# Lesson3からの変更点
# 特徴量エンジニアリングの追加部分(categorical_features = ['Sex', 'Embarked', 'Pclass'])
# と
# モデルの作成を大きく変更





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
# data_all
# pd.set_option('future.no_silent_downcasting', True)

# Sexの値を男性と女性で分ける
data_all['Sex_male'] = data_all['Sex'].replace(['male', 'female'], [1, 0]) # 男性を1, 女性を0に置き換える
data_all['Sex_female'] = data_all['Sex'].replace(['male', 'female'], [0, 1]) # 男性を0, 女性を1に置き換える

data_all['Sex'] = data_all['Sex'].replace(['male', 'female'], [0, 1]) # 既存のデータセットも男性を0, 女性を1に置き換える
# data_all.head()
# ParchとSibSpを合わせた特徴量を作成
data_all['Families'] = data_all['Parch'] + data_all['SibSp'] + 1 # +1は単身者を反映させるため。単身者はParch, SibSPともに0なので、0になってしまうからあえて+1をすることで1人という情報を与えている
# スライスと教師データの件数を使ってデータを分離する
data_train['Families'] = data_all['Families'][:len(data_train)] # リストのスライスを用いて、前から教師データの件数分まで入れる
data_test['Families'] = data_all['Families'][len(data_train):] # もしくは教師データの件数文以降を入れる＝テストデータ

data_all['Alone'] = 0 # Alone列が作成され、0で初期化される
data_all.loc[data_all['Families'] == 1, 'Alone'] = 1 # Familiesが1のデータに対してAloneに1を代入する
ave = data_all['Age'].mean()
std = data_all['Age'].std()
data_all['Age'] = data_all['Age'].fillna(np.random.randint(ave - std, ave + std))

data_all['Embarked'] = data_all['Embarked'].fillna('S')
data_all['Embarked'] = data_all['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

################
# Lesson4で追加 #
################
# LightGBMはカテゴリ変数を数値に置き換えなくても処理してくれる
categorical_features = ['Sex', 'Embarked', 'Pclass']

drop_columns = ['PassengerId', 'Name', 'Ticket', 'Cabin']
# 'Parch', 'SibSp' はこれらを使って新たな特徴量を作成したので一旦削除対象外にする

data_all.drop(drop_columns, axis=1, inplace=True)

data_all['Fare'] = data_all['Fare'].fillna(np.mean(data_all['Fare']))

#結合したデータを、再度教師データとテストデータに分離
data_train = data_all[:len(data_train)]
data_test = data_all[len(data_train):]

#アルゴリズムに投入するため、特徴量と目的変数を分離
y_data_train = data_train['Survived']
X_data_train = data_train.drop('Survived', axis=1)
X_data_test = data_test.drop('Survived', axis=1)

# 学習用データと検証用データに分割する
from sklearn.model_selection import train_test_split
X_data_train, X_data_valid, y_data_train, y_data_valid = train_test_split(X_data_train, y_data_train, test_size=0.3, random_state=0, stratify=y_data_train)

# LightGBMをインポートしてモデル作成
import lightgbm as lgb

categorical_features = ['Sex', 'Embarked', 'Pclass']

lgb_data_train = lgb.Dataset(X_data_train, y_data_train, categorical_feature=categorical_features)
lgb_data_eval = lgb.Dataset(X_data_valid, y_data_valid, reference=lgb_data_train, categorical_feature=categorical_features)

params = {
'objective': 'binary' # 0 or 1なので
}
# 以下のやり方は以前の方法なので使えない(参考:https://zenn.dev/local/articles/e2e6de3959e96d)
model = lgb.train(params, lgb_data_train, valid_sets=lgb_data_eval,
                 verbose_eval=10,
                 num_boost_round=1000,
                 early_stopping_rounds=10)