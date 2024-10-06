# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

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


###############################
# 各特徴量と目的変数の関係性を確認 #
###############################
"""
視覚的にデータを分析する
"""
# #PclassとSurvivedの件数を棒グラフで可視化
# sns.countplot(x='Pclass', hue='Survived', data=data_train)

# """
# データを見ながら仮説を立てる
# 今回の場合は、客室のグレードが1等室〜3等室あり、3等室だけ以上にSurvived=0が多いので、優先的に救助されなかったのではないかという仮説が立てられる。
# データ分析は仮説を立てて、検証の繰り返し。
# どんな視点でもいいので、何か気づきかあればメモすることを心がける。
# """

# #SexとSurvivedの件数を棒グラフで可視化
# sns.countplot(x='Sex', hue='Survived', data=data_train)
# # plt.show()

# # 女性と子供は優先的に救助されたのかもしれない。
# #AgeとSurvivedの件数を棒グラフで可視化
# #データ数が多いと非常に見づらい
# sns.countplot(x='Age', hue='Survived', data=data_train) # 見づらい
# # plt.show()
# #ヒストグラムで幅を取って確認
# # loc: 条件を持たす行データのみ抜き出す
# # 今回はAgeごとにSurvivedの値を取得している
# # Survived が　0, 1で条件指定して取得している
# # 抜き出した結果に対して、dropna(): データがない行を削除する
# # binsは棒の数.今回は30本を指定.この本数に合わせて横軸の幅が自動で設定される.
# # legendはlabel(凡例)を表示させるかどうか
# plt.hist(data_train.loc[data_train['Survived'] == 0, 'Age'].dropna(), bins=30, alpha=0.7, label='0')
# plt.hist(data_train.loc[data_train['Survived'] == 1, 'Age'].dropna(), bins=30, alpha=0.7, label='1')
# plt.xlabel('Age')
# plt.ylabel('Count')
# plt.legend(title='Survived')
# # plt.show()

# # 赤ん坊から幼児までは優先的に救助されたのではないかと考えられる.
# #SibSpとSurvivedの件数を棒グラフで可視化
# sns.countplot(x='SibSp', hue='Survived', data=data_train)

# # 兄弟や配偶者のいない方(SibSp=0)は死亡数が生存数の2倍くらいある
# # 1人の場合は生存数を上回っている
# #ParchとSurvivedの件数を棒グラフで可視化
# sns.countplot(x='Parch', hue='Survived', data=data_train)
# #凡例の位置固定
# # nocをしてすることで凡例の位置を指定できる
# plt.legend(loc='upper right', title='Survived')

# # SibSpと似た様な傾向にある.
# # 人数が増えると生存率が下がる.
# # Parchは親子の数という定義だが、SibSpとデータの傾向が非常に似ているのでデータを二つに分ける必要はないかも

# # 例えば二つを組み合わせた何か別の特徴量を作れば何か変わるかも？
# #ヒストグラムで幅を取って確認
# #特定の範囲に集まっているときはビンの範囲を指定したほうが見やすい
# plt.hist(data_train.loc[data_train['Survived'] == 0, 'Fare'].dropna(), bins=30, alpha=0.7, label='0')
# plt.hist(data_train.loc[data_train['Survived'] == 1, 'Fare'].dropna(), bins=30, alpha=0.7, label='1')
# plt.xlabel('Fare')
# plt.ylabel('Count')
# plt.legend(title='Survived')

# # 大半のデータが0~100で収まっている
# # rangeで表示範囲を指定してみたい部分を見た方がいい

#########
# 再確認 #
#########
# #特定の範囲に集まっているときはビンの範囲を指定したほうが見やすい
# plt.hist(data_train.loc[data_train['Survived'] == 0, 'Fare'].dropna(), range=(0,250), bins=20, alpha=0.7, label='0')
# plt.hist(data_train.loc[data_train['Survived'] == 1, 'Fare'].dropna(), range=(0,250), bins=20, alpha=0.7, label='1')
# plt.xlabel('Fare')
# plt.ylabel('Count')
# plt.legend(title='Survived')

# # 運賃の低い人は死亡率が高く、運賃の高い客は低い傾向がある
# # 高い運賃を払った方から救助されたのかもしれない
# #EmbarkedとSurvivedの件数を棒グラフで可視化
# sns.countplot(x='Embarked', hue='Survived', data=data_train)

# # Sの港で乗った人は多くの方が亡くなっている
# # Cの港で乗った人は逆の傾向が見られる
# # 乗船した港によって客層が多少異なることが考えられる


# **************************************************************************************************************

######################
# 特徴量エンジニアリング #
######################
"""
文字から数値への変換が特徴量エンジニアリングの役割

そもそもモデル作成に向けての必須の作業
同時によく行われるのが、欠損値の補完(欠損値のある行を削除したり、平均で埋める)

モデルの精度を向上させるための作業
ケンタッキーのチキンはクリスマスの売り上げが年間売り上げの1/3を占めるそう. 働いている人たちはわかるが、機械学習で予測する際はその情報を与えないと、その情報を反映したモデルを作ってくれない. 特徴量の作成は業務そのものの情報をデータとして用意するとても大切な作業.
"""

#教師データとテストデータ、双方にエンジニアリングを行うため、一旦ひとつに結合
data_all = pd.concat([data_train, data_test], sort=False)
print(data_all)
# オプションでsortを指定: Falseだと列は並び替えないように指定

# このあと教師データに特徴量エンジニアリングの一環でデータを変更していく.
# 教師データのみに変更を反映させると予測させるテストデータにモデルが対応できないので、二つを結合してから変更を加える.
"""
PassengerId	Survived	Pclass	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked
0	1	0.0	3	Braund, Mr. Owen Harris	male	22.0	1	0	A/5 21171	7.2500	NaN	S
1	2	1.0	1	Cumings, Mrs. John Bradley (Florence Briggs Th...	female	38.0	1	0	PC 17599	71.2833	C85	C
2	3	1.0	3	Heikkinen, Miss. Laina	female	26.0	0	0	STON/O2. 3101282	7.9250	NaN	S
3	4	1.0	1	Futrelle, Mrs. Jacques Heath (Lily May Peel)	female	35.0	1	0	113803	53.1000	C123	S
4	5	0.0	3	Allen, Mr. William Henry	male	35.0	0	0	373450	8.0500	NaN	S
...	...	...	...	...	...	...	...	...	...	...	...	...
413	1305	NaN	3	Spector, Mr. Woolf	male	NaN	0	0	A.5. 3236	8.0500	NaN	S
414	1306	NaN	1	Oliva y Ocana, Dona. Fermina	female	39.0	0	0	PC 17758	108.9000	C105	C
415	1307	NaN	3	Saether, Mr. Simon Sivertsen	male	38.5	0	0	SOTON/O.Q. 3101262	7.2500	NaN	S
416	1308	NaN	3	Ware, Mr. Frederick	male	NaN	0	0	359309	8.0500	NaN	S
417	1309	NaN	3	Peter, Master. Michael J	male	NaN	1	1	2668	22.3583	NaN	C
1309 rows × 12 columns
"""

#各特徴量の欠損値の数を確認
#テストデータは「Survived」がないため欠損値としてカウントされている
data_all.isnull().sum()
"""
PassengerId       0
Survived        418
Pclass            0
Name              0
Sex               0
Age             263
SibSp             0
Parch             0
Ticket            0
Fare              1
Cabin          1014
Embarked          2
dtype: int64
"""

################################
# Udemy_Lesson3 メモ　(精度向上） #
################################
"""
指針

①Sexは女性が圧倒的に生存率が高いため、男性と女性別々の特徴量に分けてみる

②ParchとSibSpはどちらも家族構成の特徴量であり、人数に応じて生存率が変わることから、2つを組み合わせた特徴量を作ってみる

③Ageは年代差が大きいため、平均に標準偏差を加えてばらつきを考慮する

----- 詳細 -----
①Sexは女性が圧倒的に生存率が高いため、男性と女性別々の特徴量に分けてみる

データに偏りがある場合は特徴量を分けることで精度が上がることがある。
　例：Sexは男女でデータに違いがある = データの分布が全く異なる = 同じ特徴量の中に傾向の異なるデータがまとめられているとも言える。
 ・こういう場合は特徴量時代を別々にを分けてあげることで　、それぞれの傾向をより正確に反映できるという考え方。

②ParchとSibSpはどちらも家族構成の特徴量であり、人数に応じて生存率が変わることから、2つを組み合わせた特徴量を作ってみる
　　ParchとSibSpは兄弟や配偶者、親子の人数を表す特徴量（家族構成）。意味合いとしても非常に似ており、ヒストグラムを見ても一定の人数において似たような生存率の特徴を持っている。
 ・このように似たような特徴量を持つ場合は、一つの特徴量にまとめることでよりその情報を精度に反映させられる場合がある。

③Ageは年代差が大きいため、平均に標準偏差を加えてばらつきを考慮する 　　　　 Ageの欠損値に対して、これまでは平均値を使っていたが、「欠損値の件数がそもそも多い点」、「データの分布に幅がある点」から平均値が最適かどうかわからない。
　・明確な答えはない。
 　　- 中央値や最頻値でも試してみるといったtry&errorが本来は必要
・今回は別の視点で、平均値に対して標準偏差を加味する
　　　　- これによりデータのばらつきに対する影響を多少考慮できると考えられる
"""

########################
# relaceの警告の回避方法 #
########################
# # 新しい挙動を受け入れる場合、Pandasの設定を変更することで警告を無視できます。
# pd.set_option('future.no_silent_downcasting', True)
# # 現在の動作を維持するには、replaceの後にinfer_objectsメソッドを呼び出します。
# # data_all['Sex_male'] = data_all['Sex'].replace(['male', 'female'], [1, 0]).infer_objects(copy=False)
# # data_all['Sex_female'] = data_all['Sex'].replace(['male', 'female'], [0, 1]).infer_objects(copy=False)
# # data_all['Sex'] = data_all['Sex'].replace(['male', 'female'], [0, 1]).infer_objects(copy=False)



# Sexの値を男性と女性で分ける
# ***** test *****
# data_all['Sex_male'] = data_all['Sex'].replace(['male', 'female'], [1, 0]) # 男性を1, 女性を0に置き換える
# data_all['Sex_female'] = data_all['Sex'].replace(['male', 'female'], [0, 1]) # 男性を0, 女性を1に置き換える
# data_all['Sex'] = data_all['Sex'].replace(['male', 'female'], [0, 1]) # 既存のデータセットも男性を0, 女性を1に置き換える
# これでエラー回避
data_all['Sex_male'] = data_all['Sex'].replace(['male', 'female'], [1, 0]).astype(int) # 男性を1, 女性を0に置き換える
data_all['Sex_female'] = data_all['Sex'].replace(['male', 'female'], [0, 1]).astype(int) # 男性を0, 女性を1に置き換える
data_all['Sex'] = data_all['Sex'].replace(['male', 'female'], [0, 1]).astype(int)
# data_all['Embarked'] = data_all['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
# ***** test *****
print(data_all.head())
"""
PassengerId	Survived	Pclass	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked	Sex_male	Sex_female
0	1	0.0	3	Braund, Mr. Owen Harris	0	22.0	1	0	A/5 21171	7.2500	NaN	S	1	0
1	2	1.0	1	Cumings, Mrs. John Bradley (Florence Briggs Th...	1	38.0	1	0	PC 17599	71.2833	C85	C	0	1
2	3	1.0	3	Heikkinen, Miss. Laina	1	26.0	0	0	STON/O2. 3101282	7.9250	NaN	S	0	1
3	4	1.0	1	Futrelle, Mrs. Jacques Heath (Lily May Peel)	1	35.0	1	0	113803	53.1000	C123	S	0	1
4	5	0.0	3	Allen, Mr. William Henry	0	35.0	0	0	373450	8.0500	NaN	S	1	0
"""
# ParchとSibSpを合わせた特徴量を作成
data_all['Families'] = data_all['Parch'] + data_all['SibSp'] + 1 # +1は単身者を反映させるため。単身者はParch, SibSPともに0なので、0になってしまうからあえて+1をすることで1人という情報を与えている
# スライスと教師データの件数を使ってデータを分離する
data_train['Families'] = data_all['Families'][:len(data_train)] # リストのスライスを用いて、前から教師データの件数分まで入れる
data_test['Families'] = data_all['Families'][len(data_train):] # もしくは教師データの件数文以降を入れる＝テストデータ
sns.countplot(x='Families', data=data_train, hue='Survived')
# 可視化の喧嘩、Families=1...つまり、単身者が非常に多いことがわかる
# これは指針①のように別の特徴量として抜き出しておく。

# Families=1がずば抜け鉄扉ため、別の特徴量として抜き出す
data_all['Alone'] = 0 # Alone列が作成され、0で初期化される

# locを使って各行データ(横)を書き換える
# 書き換え対象となる「条件式、対象の名前」の順で指定する
data_all.loc[data_all['Families'] == 1, 'Alone'] = 1 # Familiesが1のデータに対してAloneに1を代入する
# - 条件式：「data_all['Families'] == 1」　, 書き換え対象の列名：「'Alone'」

print(data_all.head())
"""
PassengerId	Survived	Pclass	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked	Sex_male	Sex_female	Families	Alone
0	1	0.0	3	Braund, Mr. Owen Harris	0	22.0	1	0	A/5 21171	7.2500	NaN	S	1	0	2	0
1	2	1.0	1	Cumings, Mrs. John Bradley (Florence Briggs Th...	1	38.0	1	0	PC 17599	71.2833	C85	C	0	1	2	0
2	3	1.0	3	Heikkinen, Miss. Laina	1	26.0	0	0	STON/O2. 3101282	7.9250	NaN	S	0	1	1	1
3	4	1.0	1	Futrelle, Mrs. Jacques Heath (Lily May Peel)	1	35.0	1	0	113803	53.1000	C123	S	0	1	2	0
4	5	0.0	3	Allen, Mr. William Henry	0	35.0	0	0	373450	8.0500	NaN	S	1	0	1	1
"""

# まず平均と標準偏差を求める
ave = data_all['Age'].mean()
std = data_all['Age'].std()
# 「平均±標準偏差」間の値からランダムで選ばれた値で補完
# ave-std ~ ave+std の中から整数型の値がランダムに生成される
# data_all['Age'].fillna(np.random.randint(ave - std, ave + std), inplace = True)
data_all['Age'] = data_all['Age'].fillna(np.random.randint(ave - std, ave + std))
# - 具体的には、aveから標準偏差を引いた値と足した値のいずれかの値を加える

# 乱数なので、再現性がないという欠点がある　≒ 実行結果も毎回多少の差が出る

# 年齢は若い人から高齢者まで幅広いため、中央値で補完するのも手
# - 平均だと外れ値に引っ張られてしまう
# data_all['Age'].fillna(data_all['Age'].median(), inplace = True)
# data_all['Age'] = data_all['Age'].fillna(data_all['Age'].median())
# Embarkedはこれまでと同じように補完


########################################
# Embarkedの欠損値を補完し、数値に置き換える #
########################################
# # Lesson1のやり方
# data_all['Embarked'].fillna('S', inplace=True)
# data_all['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2], inplace=True)

data_all['Embarked'] = data_all['Embarked'].fillna('S')
# どのように補完するかは、本来ならいろいろ試して上で精度を元に確認する必要がある
# 今回は欠損値は二件しかない.
# なので最もデータ件数の多い'S'の値で補完
#   - なぜ？: データが多いほど一見あたりのデータの重みは小さいから？
# 欠損の補完はfillna()
# 補完が完了したら、0, 1, 2の値に変更する


########################
# Udemy_Lesson3でのメモ #
########################
# data_all['Embarked'] = data_all['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2])
# おそらく、以下のmapでは、以下の処理をしている(mapは)
# - Embarledの 'S' を全部 0 に, 'C'を全部 1 に, 'Q'　を全部 2 に置き換える
data_all['Embarked'] = data_all['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

# 以下GPT-4o-miniの回答
# .map({'S': 0, 'C': 1, 'Q': 2}):
# map関数を使用して、'Embarked'列の各値を、指定した辞書に基づいて変換します。
# この辞書 {'S': 0, 'C': 1, 'Q': 2} では、'S'が0に、'C'が1に、'Q'が2に変換されることを示しています。
# 例えば、'S'という値が含まれている場合、それが0に変換されます。
# .astype(int):
# astype(int) は、変換後のデータ型を整数型に変更します。
# map関数によって生成された値は元々浮動小数点数型（float）である可能性があるため、整数型（int）に変換します。
# まとめ
# このコードは、'Embarked'列の値（'S', 'C', 'Q'）を対応する数値（0, 1, 2）に変換し、その後、整数型に変換しています。これにより、カテゴリカルなデータが数値データに変換され、機械学習モデルなどで使いやすくなります。



################
# Lesson4で追加 #
################
# # 対象のカラムをcategory型に変換
# data_all['Sex'] = data_all['Sex'].astype('category')
# data_all['Embarked'] = data_all['Embarked'].astype('category')
# data_all['Pclass'] = data_all['Pclass'].astype('category')

# LightGBMはカテゴリ変数を数値に置き換えなくても処理してくれる
categorical_features = ['Sex', 'Embarked', 'Pclass']
print(data_all)
"""
PassengerId	Survived	Pclass	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked	Sex_male	Sex_female	Families	Alone
0	1	0.0	3	Braund, Mr. Owen Harris	0	22.0	1	0	A/5 21171	7.2500	NaN	0	1	0	2	0
1	2	1.0	1	Cumings, Mrs. John Bradley (Florence Briggs Th...	1	38.0	1	0	PC 17599	71.2833	C85	1	0	1	2	0
2	3	1.0	3	Heikkinen, Miss. Laina	1	26.0	0	0	STON/O2. 3101282	7.9250	NaN	0	0	1	1	1
3	4	1.0	1	Futrelle, Mrs. Jacques Heath (Lily May Peel)	1	35.0	1	0	113803	53.1000	C123	0	0	1	2	0
4	5	0.0	3	Allen, Mr. William Henry	0	35.0	0	0	373450	8.0500	NaN	0	1	0	1	1
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
413	1305	NaN	3	Spector, Mr. Woolf	0	15.0	0	0	A.5. 3236	8.0500	NaN	0	1	0	1	1
414	1306	NaN	1	Oliva y Ocana, Dona. Fermina	1	39.0	0	0	PC 17758	108.9000	C105	1	0	1	1	1
415	1307	NaN	3	Saether, Mr. Simon Sivertsen	0	38.5	0	0	SOTON/O.Q. 3101262	7.2500	NaN	0	1	0	1	1
416	1308	NaN	3	Ware, Mr. Frederick	0	15.0	0	0	359309	8.0500	NaN	0	1	0	1	1
417	1309	NaN	3	Peter, Master. Michael J	0	15.0	1	1	2668	22.3583	NaN	1	1	0	3	0
1309 rows × 16 columns
"""

#影響の少なそうな特徴量は一旦削除する
# drop_columns = ['PassengerId', 'Name', 'Parch', 'SibSp', 'Ticket', 'Cabin']
drop_columns = ['PassengerId', 'Name', 'Ticket', 'Cabin']
# 'Parch', 'SibSp' はこれらを使って新たな特徴量を作成したので一旦削除対象外にする

data_all.drop(drop_columns, axis=1, inplace=True)

# 削除対象をリストにまとめる
# axis=1は列方向に削除

##############################
# Fareの欠損値を平均値で補完する #
##############################
# data_all['Fare'].fillna(np.mean(data_all['Fare']), inplace=True)
# # data_all['Fare'] = data_all['Fare'].fillna(np.mean(data_all['Fare']))
# # 元々数値の値なので数値変換は不要
# # 欠損値は平均で補完する
# data_all

####################################
# GPTによる回答(inplaceに関するエラー) #
####################################
# このエラーは、Pandasの将来のバージョンでの挙動の変化を警告しています。具体的には、チェインされた代入（chained assignment）を行う際に、inplace パラメータが期待通りに動作しない可能性があることを示しています。
# この警告を回避するために、以下のようにコードを修正することができます
# 1. inplaceパラメータを使用しない方法:
data_all['Fare'] = data_all['Fare'].fillna(np.mean(data_all['Fare']))

# 2. 列全体に対して処理を行う方法:
# data_all.fillna({'Fare': np.mean(data_all['Fare'])}, inplace=True)

# Lesson1ではここでAgeの欠損を平均で補完していたが、すでに「指標③ 平均に標準偏差を加えてばらつきを考慮」にて実施済み

# **************************************************************************************************************

#################
# データセット作成 #
#################
#結合したデータを、再度教師データとテストデータに分離
data_train = data_all[:len(data_train)]
data_test = data_all[len(data_train):]

# 今回は教師データとテストデータは単純に縦連結している
# 上側が教師データ、下側がテストデータ

# スライスを使って戦闘業から教師データの件数分抜き出す
# その後、教師データの1個下の行から最後までを抜き出す

# **************************************************************************************************************
# add
# y_data_test = data_test['Survived']
# print("y_data_test : ", y_data_test)
# print(data_test)

##############################
# モデル作成(Lesson6での変更点) #
##############################
#アルゴリズムに投入するため、特徴量と目的変数を分離
y_data_train = data_train['Survived']
X_data_train = data_train.drop('Survived', axis=1)
X_data_test = data_test.drop('Survived', axis=1)

# print("y_data_train : ", y_data_train)
# print(X_data_train)

# Cross Validationでモデル作成
from sklearn.model_selection import KFold
# os.environ['LIGHTGBM_RANKING'] = '0'  # 互換モジュールを無効化
# import lightgbm as lgb
"""
Local環境ではエラーになるので、コメントアウト
lgbmではなく、ランダムフォレストを使う
"""
# **************************************************************************************************************
############
# モデル作成 #
############
#ランダムフォレストアルゴリズムをインポート
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
# **************************************************************************************************************






y_data_preds = [] # 各分割でのテストデータに対する予測値を格納するリスト
models = [] # 各分割で学習したモデルを格納するリスト
oof_data_train = np.zeros((len(X_data_train),)) # 各分割での検証用データに対する予測値を格納するリスト
cv = KFold(n_splits=5, shuffle=True, random_state=0) # データをどのように分割するかの指定
# n_splits=5は分割数の指定 : 教師データを学習用データと検証用データに分割したペアを5つ用意する
# shuffle=True : 分割前に一旦データをシャッフルする(データの偏り軽減)
# random_state=0 : 乱数のシードを固定

# params = {
#     'objective': 'binary', # 0 or 1なので
#     'max_bin': 300, # 各特徴量の最大の分割数(大きすることでよりアルゴリズムの動作の幅が広がる可能性がある)
#     'learning_rate': 0.05, # 小さくすることで丁寧に学習
#     'num_leaves': 40 # 一つの決定木に対する分岐の末端の最大数(大きくすることでアルゴリズムの表現の幅が広がる)
# }

# add
from sklearn.metrics import accuracy_score
# accuracy_score関数で簡単に正解率を求められる
# y_data_pred_oof = (oof_data_train > 0.5).astype(int)

# print("oof_data_train : ", oof_data_train)

# 今回は分割数が5なので、5回繰り返す
for fold_id, (train_index, valid_index) in enumerate(cv.split(X_data_train)):
    X_tr = X_data_train.loc[train_index, :] # 説明変数の学習データ
    X_val = X_data_train.loc[valid_index, :] # 説明変数の検証データ
    y_tr = y_data_train.loc[train_index] # 目的変数の学習データ
    y_val = y_data_train.loc[valid_index] # 目的変数の検証データ
    
    # # lgb
    # lgb_data_train = lgb.Dataset(X_tr, y_tr, categorical_feature=categorical_features)
    # lgb_data_eval = lgb.Dataset(X_val, y_val, reference=lgb_data_train, categorical_feature=categorical_features)
    
    # model = lgb.train(params,lgb_data_train, valid_sets=lgb_data_eval,
    # #                       valid_names=['train', 'valid'],
    #                   num_boost_round=1000,
    #                   callbacks=[lgb.log_evaluation(period=10), lgb.early_stopping(10)],
    #                  )
    
    
    
    # # lgb
    # oof_data_train[valid_index] = model.predict(X_val, num_iteration=model.best_iteration)
    # y_data_pred = model.predict(X_data_test, num_iteration=model.best_iteration)
    
    #################
    # 教師データの学習 #
    #################
    model = clf.fit(X_tr, y_tr) # X_data_train, y_data_train)
    #####################
    # 作成したモデルで予測 #
    #####################
    #predictの閾値は0.5がデフォルトなので、0.5以上を1、未満を0として返す
    # y_data_pred = clf.predict(X_data_test)
    oof_data_train[valid_index] = model.predict(X_val) # 教師データのうちのValidation
    
    y_data_pred_oof = (oof_data_train > 0.5).astype(int)
    # print("PRED:", y_data_pred)
    # print("OOF:", y_data_pred_oof)
    print("acc score : ", accuracy_score(y_data_train, y_data_pred_oof)) # 教師データうちのValidationの正解率を評価
    # print(len(y_data_pred_oof), len(y_data_pred))
    
    
    y_data_pred = model.predict(X_data_test) # 実際にテストデータを予測
    
    
    y_data_preds.append(y_data_pred)
    models.append(model)
    # models.append(clf)
    # print(model)

"""
~中略~
Early stopping, best iteration is:
[53]	valid_0's binary_logloss: 0.428577
"""

# 各モデルのスコアと平均を計算

# # 各モデルのベストスコアを格納する
# scores = [
#     # 引数のキーワードは、先ほどの実行結果のものを入れる(<>の部分)
#     # [59]	<valid_0>'s <binary_logloss>: 0.440187
#     m.best_score['valid_0']['binary_logloss'] for m in models # forの内包表記をする場合、[繰り返しの処理(処理の中身), for]の条件の順に書く
# ]

# score = sum(scores) / len(scores)
# print(scores)
# print(score)
"""
[0.37631477357721776, 0.4379802759624778, 0.36763497698030134, 0.4243885050194767, 0.4285774935989604]
0.40697920502768675
"""
#oof_data_trainに対する正解率を計算
# from sklearn.metrics import accuracy_score
# accuracy_score関数で簡単に正解率を求められる

# y_data_pred_oof = (oof_data_train > 0.5).astype(int)
# print(len(y_data_train), len(y_data_pred_oof))
# print("acc score : ", accuracy_score(y_data_train, y_data_pred_oof))
"""
0.8294051627384961
"""
# print("y_data_pred:", y_data_pred)


# # Lesson7で追加(分割されたデータの割合が本当に一定になるのか確かめる. 目的変数yが1の割合を出力)
# # yは0or1なので、その合計を件数で割れば1の割合が求められる
# print("y_train data rate (0, 1): ", sum(y_tr)/len(y_tr))
# print("y_val data rate (0, 1): ", sum(y_val)/len(y_val))
# # 0と1の割合が一定ならうまく学習データと検証データが均等に分けられていることになる
# # 今回はどれも0.38くらいなどでほぼ均等にデータが分けられていることになる
# #################
# # 教師データの学習 #
# #################
# clf.fit(X_data_train, y_data_train)
# #####################
# # 作成したモデルで予測 #
# #####################
# #predictの閾値は0.5がデフォルトなので、0.5以上を1、未満を0として返す
# y_data_pred = clf.predict(X_data_test)
# # print("y_data_pred : ", y_data_pred)
# # y_data_pred_oof = (oof_data_train > 0.5).astype(int)
# print(len(y_data_test), len(y_data_pred))
# # print("acc score : ", accuracy_score(y_data_test, y_data_pred))
# print("*****")
# print("y_data_test:", y_data_test)
# print("y_data_pred:", y_data_pred)





 
# #アルゴリズムに投入するため、特徴量と目的変数を分離
# y_data_train = data_train['Survived']
# X_data_train = data_train.drop('Survived', axis=1)
# X_data_test = data_test.drop('Survived', axis=1)

# # 説明変数と目的変数をそれぞれ　x, yとして分ける
# # x:説明変数はdrop()でSurvived=答えのみ削除する
# # axis=1は列を削除

# # 教師データだけでなく、テストデータも同様にSurvived列のみ削除する
# # 特徴量エンジニアリングで結合した際に、テストデータになかったSurvived列が欠損値(NaN)として作成されてしまったので、合わせて削除する

# # y_train:答え(ラベル)
# # x_train:学習に使うデータ
# # x_test: 予測精度評価に使うデータ
# # 学習用データと検証用データに分割する
# from sklearn.model_selection import train_test_split

# # test
# # Separate the label
# # y = train_df.pop('Survived')

# # 検証とは
# # モデル作成後にテストデータで検証する前にモデルを作成の時点で現状のモデルの精度を検証データを使って確認しながら学習を進める手法
# # 後で過学習などが判明する前にモデル作成中にモデル作成中に検証データを使って精度を確認する

# X_data_train, X_data_valid, y_data_train, y_data_valid = train_test_split(X_data_train, y_data_train, test_size=0.3, random_state=0, stratify=y_data_train)
# # 成果物
# # 説明変数と目的変数が学習用と検証用でそれぞれできるので4つ

# # train_test_splitの引数
# # 説明変数, 目的変数, 検証データの割合, 乱数のシード, 学習用と検証用でデータの割合を保つという指定(目的変数Survivedが学習データでは1, 検証データでは0といったラベルが偏るといったことがないように、均一になるようにする)
# # LightGBMをインポートしてモデル作成
# import lightgbm as lgbm

# categorical_features = ['Sex', 'Embarked', 'Pclass']

# lgb_data_train = lgbm.Dataset(X_data_train, y_data_train, categorical_feature=categorical_features)
# lgb_data_eval = lgbm.Dataset(X_data_valid, y_data_valid, reference=lgb_data_train, categorical_feature=categorical_features)
# # LightGBMではカテゴリ型(文字列型)のデータに対して独自に解釈を加えてくれる(数字に置き換えた際の懸念となる大きさや順序などの影響の考慮)
# # categori_feature...文字型の変数 : カテゴリ型のデータがあれば変数としてまとめて指定できる
# # refernceは学習データを指定する : 学習データと区別するための指定(教師データに対してどのくらい精度に差があるとか？)






# #　 ***** Lesson5での変更点 *****

# # Lesson4
# # params = {
# # 'objective': 'binary' # 0 or 1なので
# # }

# # acc : 0.76
# params = {
#     'objective': 'binary', # 0 or 1なので
#     'max_bin': 300, # 各特徴量の最大の分割数(大きすることでよりアルゴリズムの動作の幅が広がる可能性がある)
#     'learning_rate': 0.05, # 小さくすることで丁寧に学習
#     'num_leaves': 40 # 一つの決定木に対する分岐の末端の最大数(大きくすることでアルゴリズムの表現の幅が広がる)
# }
# # acc: 0.75
# # params = {
# #     'boosting': 'gbdt',
# #     'max_bin': 300, # 各特徴量の最大の分割数(大きすることでよりアルゴリズムの動作の幅が広がる可能性がある)
# #     'learning_rate': 0.05, # 小さくすることで丁寧に学習
# #     'num_leaves': 40 # 一つの決定木に対する分岐の末端の最大数(大きくすることでアルゴリズムの表現の幅が広がる)
# # }

# #　 ***** Lesson5での変更点 *****

# # MEMO
# # モデルの精度を上げるためにパラメータをチューニングすることも大切だが、（最後のひと押しくらい）
# # それ以上に精度に貢献するデータそのものを作り出す特徴量エンジニアリングの方が大事（時間をかけるならこっち）




# # # Create an LGBM dataset for training
# # categorical_features = ['Sex', 'Pclass', 'Embarked']
# # train_data = lgbm.Dataset(data=X_data_train, label=y_data_train, categorical_feature=categorical_features, free_raw_data=False)
# # # Create an LGBM dataset from the test
# # test_data = lgbm.Dataset(data=X_data_test, label=y_data_valid, categorical_feature=categorical_features, free_raw_data=False)
# # # Finally, create a dataset for the FULL training data to give us maximum amount of data to train on after

# # # performance has been calibrate
# # final_train_set = lgbm.Dataset(data=data_all, label=y_data_train, 
# # #                               categorical_feature=categorical_features, free_raw_data=False)
# #パラメータはGBDTの指定のみ
# lgbm_params = {
#     'boosting': 'gbdt',          # GBDTを使う
#               }
# #モデル構築
# # model = lgbm.train(params=lgbm_params,train_set=final_train_set)
# model = lgbm.train(
#     params,
# #     lgbm_params,
# #     train_data,
# #     train_set=final_train_set, # main
# #     valid_sets=[train_data, valid_data], # main
#     lgb_data_train, 
#     valid_sets=lgb_data_eval,
# #     valid_sets=[lgb_data_train, lgb_data_eval], # test
    
#     valid_names=['train', 'valid'],
#     num_boost_round=1000,
# #     callbacks=[lgb.early_stopping(100)],
#     callbacks=[lgbm.log_evaluation(period=10), lgbm.early_stopping(10)],
# #     verbose_eval=100, # log_evaluationに変更
# )

# # モデル作成におけるベースの情報を目的関数として指定(回帰や分類において機械学習を使う目的や精度を測る手法は何を使うかなど)
# # 最低限目的を表すobjectiveのみ指定すればあとはobjectiveに合わせてデフォルトで設定してくれる

# # 以下のやり方は以前の方法なので使えない(参考:https://zenn.dev/local/articles/e2e6de3959e96d)
# # model = lgbm.train(params, lgb_data_train, valid_sets=lgb_data_eval) # ,
# # #                  verbose_eval=10,
# # #                  num_boost_round=1000,
# # #                  early_stopping_rounds=10)
# # # Before
# # lgb.train(
# #     params,
# #     train_set,
# #     valid_sets=[train_set, valid_set],
# #     valid_names=['train', 'valid'],
# #     num_boost_round=1000,
# #     early_stopping_rounds=100,　# ここが変わった
# #     verbose_eval=100,
# # )
# # 新しいやり方
# # After
# # lgb.train(
# #     params,
# #     train_set,
# #     valid_sets=[train_set, valid_set],
# #     valid_names=['train', 'valid'],
# #     num_boost_round=1000,
# #     callbacks=[lgb.early_stopping(100)],　# ここが変わった
# #     verbose_eval=100,
# # )

# # 検証データの指定はvalid_setsを指定する
# # verbose_eval : 学習を10回ごとに表示する
# # num_boost_round : 学習を何回繰り返すか(勾配ブースティング決定木を順々に作成することを1000回繰り返す)
# # early_stopping : 過学習になりそうなタイミングで学習を終了する(今回は誤差が一定以上小さくならなくて、10回様子見してそれでも変わらなければ終了するという意味)

# # LightGBMを含める勾配ブースティング系のアルゴリズムはハイパーパラメータはとても多く、全てを適切に設定できるデータサイエンティストは少ない
# # 少しずつ覚える


"""
今回のエラーの原因はラベルがintやfloat型ではなく、おそらく文字列型だったから(.astype(int)でint型にしてあげる必要がある)
"""
'\n今回のエラーの原因はラベルがintやfloat型ではなく、おそらく文字列型だったから(.astype(int)でint型にしてあげる必要がある)\n'


# **************************************************************************************************************

#########################
# 予測(Lesson6での変更点) #
#########################
# # 作成したモデルで予測(ベストな学習回数のモデルで実施)
# y_data_pred = model.predict(X_data_test, num_iteration=model.best_iteration)
# # 出力結果は0~1の連続値となる
# y_data_pred[:5]
# # 0.5を上回れば1と予測したとみなす
# y_data_pred = (y_data_pred > 0.5).astype(int)
# y_data_pred[:5]

# **************************************************************************************************************

#################################
# 予測結果をSubmit用のcsvとして作成 #
#################################
# submit = data_gender_submission
# submit['Survived'] = list(map(int, y_data_pred))
# submit.to_csv('tuning_submit_boosting_gbdt.csv', index=False)
# Lesson6 予測
y_pred_submit = sum(y_data_preds) / len(y_data_preds)
print(y_pred_submit)
y_pred_submit = (y_pred_submit > 0.5).astype(int)
print(y_pred_submit)
"""
array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0,
       1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
       1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1,
       1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0,
       1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
       0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
       0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
       1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1,
       0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0,
       1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
       0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1,
       0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
       0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0,
       1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0,
       0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0,
       1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1,
       0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0])
"""

# **************************************************************************************************************
##########
# Submit #
##########
#予測結果をSubmit用のcsvとして作成
submit = data_gender_submission
submit['Survived'] = list(map(int, y_pred_submit))
submit.to_csv('kfold_submit.csv', index=False)