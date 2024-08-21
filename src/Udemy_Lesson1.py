# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
for dirname, _, filenames in os.walk('/kaggle_ws/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
"""
/kaggle/input/titanic/train.csv
/kaggle/input/titanic/test.csv
/kaggle/input/titanic/gender_submission.csv
"""

# パッケージの読み込み
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# data_train = pd.read_csv('/kaggle/input/titanic/train.csv') # /kaggle/input/titanic/train.csv
# data_test = pd.read_csv('/kaggle/input/titanic/test.csv') # /kaggle/input/titanic/test.csv
# data_gender_submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv') # /kaggle/input/titanic/gender_submission.csv
#データの読み込みはPandasの「read_csv()」
data_train = pd.read_csv('../input/titanic/train.csv')
data_test = pd.read_csv('../input/titanic/test.csv')
data_gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')


#######################
# EDA（探索的データ分析）#
#######################
print(data_train)
"""
PassengerId	Survived	Pclass	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked
0	1	0	3	Braund, Mr. Owen Harris	male	22.0	1	0	A/5 21171	7.2500	NaN	S
1	2	1	1	Cumings, Mrs. John Bradley (Florence Briggs Th...	female	38.0	1	0	PC 17599	71.2833	C85	C
2	3	1	3	Heikkinen, Miss. Laina	female	26.0	0	0	STON/O2. 3101282	7.9250	NaN	S
3	4	1	1	Futrelle, Mrs. Jacques Heath (Lily May Peel)	female	35.0	1	0	113803	53.1000	C123	S
4	5	0	3	Allen, Mr. William Henry	male	35.0	0	0	373450	8.0500	NaN	S
...	...	...	...	...	...	...	...	...	...	...	...	...
886	887	0	2	Montvila, Rev. Juozas	male	27.0	0	0	211536	13.0000	NaN	S
887	888	1	1	Graham, Miss. Margaret Edith	female	19.0	0	0	112053	30.0000	B42	S
888	889	0	3	Johnston, Miss. Catherine Helen "Carrie"	female	NaN	1	2	W./C. 6607	23.4500	NaN	S
889	890	1	1	Behr, Mr. Karl Howell	male	26.0	0	0	111369	30.0000	C148	C
890	891	0	3	Dooley, Mr. Patrick	male	32.0	0	0	370376	7.7500	NaN	Q
891 rows × 12 columns
"""

# 出力行数の確認
print(pd.get_option('display.max_rows'))
"""
60
"""
# 出力行数の変更
pd.set_option('display.max_rows', 10) # 900)

# データ数が多い場合は全件表示は時間がかかる
# head()で表示行数をしてい(デフォルトは5行、括弧の中に行数指定可能)
print(data_train.head())
"""
PassengerId	Survived	Pclass	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked
0	1	0	3	Braund, Mr. Owen Harris	male	22.0	1	0	A/5 21171	7.2500	NaN	S
1	2	1	1	Cumings, Mrs. John Bradley (Florence Briggs Th...	female	38.0	1	0	PC 17599	71.2833	C85	C
2	3	1	3	Heikkinen, Miss. Laina	female	26.0	0	0	STON/O2. 3101282	7.9250	NaN	S
3	4	1	1	Futrelle, Mrs. Jacques Heath (Lily May Peel)	female	35.0	1	0	113803	53.1000	C123	S
4	5	0	3	Allen, Mr. William Henry	male	35.0	0	0	373450	8.0500	NaN	S
"""
print(data_train.head(10))
"""
PassengerId	Survived	Pclass	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked
0	1	0	3	Braund, Mr. Owen Harris	male	22.0	1	0	A/5 21171	7.2500	NaN	S
1	2	1	1	Cumings, Mrs. John Bradley (Florence Briggs Th...	female	38.0	1	0	PC 17599	71.2833	C85	C
2	3	1	3	Heikkinen, Miss. Laina	female	26.0	0	0	STON/O2. 3101282	7.9250	NaN	S
3	4	1	1	Futrelle, Mrs. Jacques Heath (Lily May Peel)	female	35.0	1	0	113803	53.1000	C123	S
4	5	0	3	Allen, Mr. William Henry	male	35.0	0	0	373450	8.0500	NaN	S
5	6	0	3	Moran, Mr. James	male	NaN	0	0	330877	8.4583	NaN	Q
6	7	0	1	McCarthy, Mr. Timothy J	male	54.0	0	0	17463	51.8625	E46	S
7	8	0	3	Palsson, Master. Gosta Leonard	male	2.0	3	1	349909	21.0750	NaN	S
8	9	1	3	Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)	female	27.0	0	2	347742	11.1333	NaN	S
9	10	1	2	Nasser, Mrs. Nicholas (Adele Achem)	female	14.0	1	0	237736	30.0708	NaN	C
"""
print(data_test.head(10))
"""
PassengerId	Pclass	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked
0	892	3	Kelly, Mr. James	male	34.5	0	0	330911	7.8292	NaN	Q
1	893	3	Wilkes, Mrs. James (Ellen Needs)	female	47.0	1	0	363272	7.0000	NaN	S
2	894	2	Myles, Mr. Thomas Francis	male	62.0	0	0	240276	9.6875	NaN	Q
3	895	3	Wirz, Mr. Albert	male	27.0	0	0	315154	8.6625	NaN	S
4	896	3	Hirvonen, Mrs. Alexander (Helga E Lindqvist)	female	22.0	1	1	3101298	12.2875	NaN	S
5	897	3	Svensson, Mr. Johan Cervin	male	14.0	0	0	7538	9.2250	NaN	S
6	898	3	Connolly, Miss. Kate	female	30.0	0	0	330972	7.6292	NaN	Q
7	899	2	Caldwell, Mr. Albert Francis	male	26.0	1	1	248738	29.0000	NaN	S
8	900	3	Abrahim, Mrs. Joseph (Sophie Halaut Easu)	female	18.0	0	0	2657	7.2292	NaN	C
9	901	3	Davies, Mr. John Samuel	male	21.0	2	0	A/4 48871	24.1500	NaN	S
"""
print(data_gender_submission.head())
"""
PassengerId	Survived
0	892	0
1	893	1
2	894	0
3	895	0
4	896	1
"""
#describe()を使えば、主要な統計指標をまとめて確認できる
print(data_train.describe())
"""
PassengerId	Survived	Pclass	Age	SibSp	Parch	Fare
count	891.000000	891.000000	891.000000	714.000000	891.000000	891.000000	891.000000
mean	446.000000	0.383838	2.308642	29.699118	0.523008	0.381594	32.204208
std	257.353842	0.486592	0.836071	14.526497	1.102743	0.806057	49.693429
min	1.000000	0.000000	1.000000	0.420000	0.000000	0.000000	0.000000
25%	223.500000	0.000000	2.000000	20.125000	0.000000	0.000000	7.910400
50%	446.000000	0.000000	3.000000	28.000000	0.000000	0.000000	14.454200
75%	668.500000	1.000000	3.000000	38.000000	1.000000	0.000000	31.000000
max	891.000000	1.000000	3.000000	80.000000	8.000000	6.000000	512.329200
"""
# #データの概要をざっと確認するには「pandas_profiling」が便利
# import pandas_profiling
# #実行はpandas.DataFrameの「profile_report()」
# data_train.profile_report()





###############################
# 各特徴量と目的変数の関係性を確認 #
###############################

#PclassとSurvivedの件数を棒グラフで可視化
sns.countplot(x='Pclass', hue='Survived', data=data_train)

"""
データを見ながら仮説を立てる
今回の場合は、客室のグレードが1等室〜3等室あり、3等室だけ以上にSurvived=0が多いので、優先的に救助されなかったのではないかという仮説が立てられる。

データ分析は仮説を立てて、検証の繰り返し。
どんな視点でもいいので、何か気づきかあればメモすることを心がける。
"""
'\nデータ分析は仮説を立てて、検証の繰り返し。\nどんな視点でもいいので、何か気づきかあればメモすることを心がける。\n'

# SexとSurvivedの件数を棒グラフで可視化
sns.countplot(x='Sex', hue='Survived', data=data_train)
# plt.show()
# 女性と子供は優先的に救助されたのかもしれない。

# AgeとSurvivedの件数を棒グラフで可視化
# データ数が多いと非常に見づらい
sns.countplot(x='Age', hue='Survived', data=data_train) # 見づらい
# plt.show()
# なので…⇩
# ヒストグラムで幅を取って確認
# loc: 条件を持たす行データのみ抜き出す
# 今回はAgeごとにSurvivedの値を取得している
# Survived が　0, 1で条件指定して取得している
# 抜き出した結果に対して、dropna(): データがない行を削除する
# binsは棒の数.今回は30本を指定.この本数に合わせて横軸の幅が自動で設定される.
# legendはlabel(凡例)を表示させるかどうか
plt.hist(data_train.loc[data_train['Survived'] == 0, 'Age'].dropna(), bins=30, alpha=0.7, label='0')
plt.hist(data_train.loc[data_train['Survived'] == 1, 'Age'].dropna(), bins=30, alpha=0.7, label='1')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend(title='Survived')
# plt.show()
# 赤ん坊から幼児までは優先的に救助されたのではないかと考えられる.

#SibSpとSurvivedの件数を棒グラフで可視化
sns.countplot(x='SibSp', hue='Survived', data=data_train)
# 兄弟や配偶者のいない方(SibSp=0)は死亡数が生存数の2倍くらいある
# 1人の場合は生存数を上回っている

#ParchとSurvivedの件数を棒グラフで可視化
sns.countplot(x='Parch', hue='Survived', data=data_train)
#凡例の位置固定
# nocをしてすることで凡例の位置を指定できる
plt.legend(loc='upper right', title='Survived')

# SibSpと似た様な傾向にある.
# 人数が増えると生存率が下がる.
# Parchは親子の数という定義だが、SibSpとデータの傾向が非常に似ているのでデータを二つに分ける必要はないかも

# 例えば二つを組み合わせた何か別の特徴量を作れば何か変わるかも？

#ヒストグラムで幅を取って確認
#特定の範囲に集まっているときはビンの範囲を指定したほうが見やすい
plt.hist(data_train.loc[data_train['Survived'] == 0, 'Fare'].dropna(), bins=30, alpha=0.7, label='0')
plt.hist(data_train.loc[data_train['Survived'] == 1, 'Fare'].dropna(), bins=30, alpha=0.7, label='1')
plt.xlabel('Fare')
plt.ylabel('Count')
plt.legend(title='Survived')

# 大半のデータが0~100で収まっている
# rangeで表示範囲を指定してみたい部分を見た方がいい

#再確認
#特定の範囲に集まっているときはビンの範囲を指定したほうが見やすい
plt.hist(data_train.loc[data_train['Survived'] == 0, 'Fare'].dropna(), range=(0,250), bins=20, alpha=0.7, label='0')
plt.hist(data_train.loc[data_train['Survived'] == 1, 'Fare'].dropna(), range=(0,250), bins=20, alpha=0.7, label='1')
plt.xlabel('Fare')
plt.ylabel('Count')
plt.legend(title='Survived')

# 運賃の低い人は死亡率が高く、運賃の高い客は低い傾向がある
# 高い運賃を払った方から救助されたのかもしれない

#EmbarkedとSurvivedの件数を棒グラフで可視化
sns.countplot(x='Embarked', hue='Survived', data=data_train)

# Sの港で乗った人は多くの方が亡くなっている
# Cの港で乗った人は逆の傾向が見られる
# 乗船した港によって客層が多少異なることが考えられる

# 一気に全部表示
plt.show()






######################
# 特徴量エンジニアリング #
######################

"""
# 文字から数値への変換が特徴量エンジニアリングの役割

# そもそもモデル作成に向けての必須の作業
# 同時によく行われるのが、欠損値の補完(欠損値のある行を削除したり、平均で埋める)

# モデルの精度を向上させるための作業
# ケンタッキーのチキンはクリスマスの売り上げが年間売り上げの1/3を占めるそう. 働いている人たちはわかるが、機械学習で予測する際はその情報を与えないと、その情報を反映したモデルを作ってくれない. 特徴量の作成は業務そのものの情報をデータとして用意するとても大切な作業.
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
print(data_all.isnull().sum())
"""
PassengerId       0
Survived        418
Pclass            0
Name              0
Sex               0
               ... 
Parch             0
Ticket            0
Fare              1
Cabin          1014
Embarked          2
Length: 12, dtype: int64
"""
#Sexの値を数値に置き換え
# data_all['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
data_all['Sex'] = data_all['Sex'].replace(['male', 'female'], [0, 1])

# replace関数は一つ目の引数に開き変え前の値、二つ目の引数に置き換え後の値を指定する
# inplace=True: 元のdfに変更結果を反映させる
# /tmp/ipykernel_18/1610560096.py:3: FutureWarning: Downcasting behavior in `replace` is deprecated and will be removed in a future version. To retain the old behavior, explicitly call `result.infer_objects(copy=False)`. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
#   data_all['Sex'] = data_all['Sex'].replace(['male', 'female'], [0, 1])

print(data_all.head())
"""
PassengerId	Survived	Pclass	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked
0	1	0.0	3	Braund, Mr. Owen Harris	0	22.0	1	0	A/5 21171	7.2500	NaN	S
1	2	1.0	1	Cumings, Mrs. John Bradley (Florence Briggs Th...	1	38.0	1	0	PC 17599	71.2833	C85	C
2	3	1.0	3	Heikkinen, Miss. Laina	1	26.0	0	0	STON/O2. 3101282	7.9250	NaN	S
3	4	1.0	1	Futrelle, Mrs. Jacques Heath (Lily May Peel)	1	35.0	1	0	113803	53.1000	C123	S
4	5	0.0	3	Allen, Mr. William Henry	0	35.0	0	0	373450	8.0500	NaN	S
"""
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

print(data_all.head(900))
"""
PassengerId	Survived	Pclass	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked
0	1	0.0	3	Braund, Mr. Owen Harris	0	22.0	1	0	A/5 21171	7.2500	NaN	0
1	2	1.0	1	Cumings, Mrs. John Bradley (Florence Briggs Th...	1	38.0	1	0	PC 17599	71.2833	C85	1
2	3	1.0	3	Heikkinen, Miss. Laina	1	26.0	0	0	STON/O2. 3101282	7.9250	NaN	0
3	4	1.0	1	Futrelle, Mrs. Jacques Heath (Lily May Peel)	1	35.0	1	0	113803	53.1000	C123	0
4	5	0.0	3	Allen, Mr. William Henry	0	35.0	0	0	373450	8.0500	NaN	0
...	...	...	...	...	...	...	...	...	...	...	...	...
4	896	NaN	3	Hirvonen, Mrs. Alexander (Helga E Lindqvist)	1	22.0	1	1	3101298	12.2875	NaN	0
5	897	NaN	3	Svensson, Mr. Johan Cervin	0	14.0	0	0	7538	9.2250	NaN	0
6	898	NaN	3	Connolly, Miss. Kate	1	30.0	0	0	330972	7.6292	NaN	2
7	899	NaN	2	Caldwell, Mr. Albert Francis	0	26.0	1	1	248738	29.0000	NaN	0
8	900	NaN	3	Abrahim, Mrs. Joseph (Sophie Halaut Easu)	1	18.0	0	0	2657	7.2292	NaN	1
900 rows × 12 columns
"""

# #Fareの欠損値を平均値で補完する
# data_all['Fare'].fillna(np.mean(data_all['Fare']), inplace=True)
# # data_all['Fare'] = data_all['Fare'].fillna(np.mean(data_all['Fare']))
# # 元々数値の値なので数値変換は不要
# # 欠損値は平均で補完する
# data_all

# GPTによる回答
# このエラーは、Pandasの将来のバージョンでの挙動の変化を警告しています。具体的には、チェインされた代入（chained assignment）を行う際に、inplace パラメータが期待通りに動作しない可能性があることを示しています。
# この警告を回避するために、以下のようにコードを修正することができます
# 1. inplaceパラメータを使用しない方法:
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
"""
Survived    418
Pclass        0
Sex           0
Age           0
Fare          0
Embarked      0
dtype: int64
"""





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







"""
One-Hotエンコーディングについてのメモ

今回は文字から数値に置き換える際に一つの列で置き換えた
ただ、1つの列の中で数値に置き換えると、数値の大きさに意味があるのではないかと解釈される恐れがある
例: data_all['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2])
S, C, Qはただの文字列であり、大きさや順番といった概念はないが、0~2にすると、0<2だからこの大きさや順番に意味があるのではないかとアルゴリズムが解釈する恐れがある
今回は2までなので差は小さく影響はあまりないと思われるが、10や100になってくると、その影響は看過できなくなると考えられる
そこで、One-Hotエンコーディングを使って0と1だけで値を置き換える

置き換え元のデータの種類分だけ列を作成する
S:[1,0,0], C:[0,1,0], Q:[0,0,1]
懸念点もある
データの種類分だけ新たな列が作られるため、単縦に列数がかなり増えてしまう
便利な方法だが、データサイズが増えてしまうというデメリットもある

ただ、アルゴリズムによって変わる ※LightBGMなどのアルゴリズムでは、カテゴリ(変換前の特徴量)=文字型かどうかを指定できる そうすれば、数値特有の大きさや順番を考慮しないでモデルを作ってくれるものもある

なので、One-Hotエンコーディングが必要かどうかはアルゴリズムによる
まずは使用するアルゴリズムのルール確認した上でどのように変換するかが大切
"""



############
# モデル作成 #
############

#アルゴリズムに投入するため、特徴量と目的変数を分離
y_data_train = data_train['Survived']
X_data_train = data_train.drop('Survived', axis=1)
X_data_test = data_test.drop('Survived', axis=1)

# 説明変数と目的変数をそれぞれ　x, yとして分ける
# x:説明変数はdrop()でSurvived=答えのみ削除する
# axis=1は列を削除

# 教師データだけでなく、テストデータも同様にSurvived列のみ削除する
# 特徴量エンジニアリングで結合した際に、テストデータになかったSurvived列が欠損値(NaN)として作成されてしまったので、合わせて削除する

# y_train:答え(ラベル)
# x_train:学習に使うデータ
# x_test: 予測精度評価に使うデータ
#ランダムフォレストアルゴリズムをインポート
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

# n_estimators: 決定木の本数(今回は100本の決定木でアンサンブルしている)

# max_depth: 条件をどこまで掘り下げて結果を出すか(今回は2階層分条件を分岐して結果を出す)
#   - 極端に浅いよりは深い方がいいが、必ずしも深ければ深いほどいいというわけではない
#   - なぜか？: 予測に関係ない条件も含まれることがあるため. また、処理に時間がかかる

# random_state: 同じ数値を指定すれば毎回実行結果が同じになる(数字はなんでもいい)

"""
"ランダムフォレスト"アルゴリズムについてのメモ

アルゴリズムとは:　数値に対して演算を行う処理をまとめたもの

言葉の定義や中身の理論は研究者以外は知らなくてもいいが、代表的なアルゴリズムにどういうものがあるか、どういう動きをするかは知っておいた方がいい
・ ランダムフォレストとは:　決定木というアルゴリズムをベースにした拡張アルゴリズム ・ 決定木： それぞれの特徴量に対して、if文のように条件分岐を設定し、真or偽の組み合わせ に応じて予測値を決定する

ランダムフォレストは決定木のアンサンブルによって精度向上を目指したアルゴリズム

アンサンブルとは: 複数の学習機を使って最終的に一つのモデルを作成する手法
日常生活において、複数人に意見を聞いた上で物事を判断するイメージ
アンサンブルもこの考え方で、複数人に聞いてから判断した方がきっと精度上がるだろうということ(例えばそれぞれの決定木が出した答えに対して多数決を取るとか)
決定木がどのような条件を辿り、予測結果を導くかはデータの構造に応じて、アルゴリズムが自動で決定していく

学習機とは　: 教師データを投入して学習を行うアルゴリズムそのものの呼び方 機械学習含めて、データサイエンスの世界では呼び方が複数あるケースが多いので、少しずつ覚えるようにする
"""


#教師データの学習
clf.fit(X_data_train, y_data_train)

# ポイントは説明変数と目的変数を別の引数として指定すること

"""
RandomForestClassifier
RandomForestClassifier(max_depth=2, random_state=0)
"""
#作成したモデルで予測
#predictの閾値は0.5がデフォルトなので、0.5以上を1、未満を0として返す
y_data_pred = clf.predict(X_data_test)
print(y_data_pred)
"""
array([0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 1., 0.,
       0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 1.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 1., 0.,
       0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 1., 0.,
       0., 1., 1., 0., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0.,
       0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0.,
       0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 1., 0.,
       1., 1., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0.,
       0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
       1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0.,
       0., 0., 0., 0., 0., 1., 1., 0., 1., 1., 0., 0., 1., 0., 1., 0., 1.,
       0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1.,
       0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1.,
       0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
       1., 1., 1., 1., 0., 0., 0., 0., 1., 0., 1., 1., 1., 0., 0., 0., 0.,
       0., 0., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
       1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
       0., 0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
       0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 1.,
       0., 0., 1., 0., 1., 1., 0., 1., 1., 0., 1., 1., 0., 0., 1., 0., 0.,
       1., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 1., 0., 0., 0., 0., 0.,
       1., 0., 0., 0., 1., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0.,
       0., 1., 0., 1., 0., 0., 1., 0., 0., 0.])
"""
#予測結果をSubmit用のcsvとして作成
submit = data_gender_submission
submit['Survived'] = list(map(int, y_data_pred))
submit.to_csv('randomForest_submit.csv', index=False)

# map(): 指定した要素に対して、1つ1つ関数を適用できる関数
# 1つ目の引数が適用したい関数、2つ目の引数が要素を持つ変数
# map(int, y_data_pred): 全ての予測結果に対してint()をしている