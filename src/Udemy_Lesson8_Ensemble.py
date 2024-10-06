# アンサンブル

#Submitした各ファイルを読み込む
import pandas as pd
# submit1 = pd.read_csv('../input/submit/randomForest_submit.csv')
# submit2 = pd.read_csv('../input/submit/kfold_submit.csv')
# submit3 = pd.read_csv('../input/submit/stratified_kfold_submit.csv')
submit1 = pd.read_csv('../input/ensemble-test/randomForest_submit.csv')
submit2 = pd.read_csv('../input/ensemble-test/kfold_submit.csv')
submit3 = pd.read_csv('../input/ensemble-test/stratified_kfold_submit.csv')

#各値を確認
submit_all = pd.DataFrame({'submit1': submit1['Survived'].values,
                           'submit2': submit2['Survived'].values,
                           'submit3': submit3['Survived'].values})

submit_all.head()
# submit1	submit2	submit3
# 0	0	0	0
# 1	0	0	0
# 2	0	0	0
# 3	0	0	0
# 4	1	0	0


#相関を確認
submit_all.corr() # corr()で相関係数を求められる
# submit1	submit2	submit3
# submit1	1.000000	0.722614	0.676350
# submit2	0.722614	1.000000	0.947378
# submit3	0.676350	0.947378	1.000000

"""
<重要>どのモデルをアンサンブルに選ぶか
一つの指針

なるべく毛色の異なるものを選ぶのが一般的 ・似た傾向のアルゴリズムによる偏りの影響をなるべくなくすためにも、決定木とランダムフォレストを選ぶのではなく、ロジスティック回帰とランダムフォレストといった様にアルゴリズムの考え方が全く異なるものを選ぶ

・そのための指針として相関係数を使う

 - 相関係数の高いものは予測結果も似ている＝似た動きをするアルゴリズム
     ・極端に高い相関係数が出た場合は別のモデルでのアンサンブルを検討するが筋
アンサンブルは機械学習のモデル作成において極めて重要な考え方
モデルを作る際、最も精度が高いもの1つを選べば良いと考えがちだが、この考え方は少々危険
モデル作成はあくまで作った時点での教師データを使った上でのモデルなので、この先ずっと使い続けてデータが更新されてもその精度を保ち続けられるとは限らない
また、乱数などで精度にランダム性が発生することも考えられる(ageなどの欠損値も乱数で補完している)
一言で言えば、

「今回のデータの状況的にこのモデルの精度がベストだった」というだけ
長い目で見て精度を安定させるための1つの方法がアンサンブル
1つのモデルにこだわるのはなく、一定以上の精度があるモデルを組み合わせて、アルゴリズムの幅を持たせることにより、データの更新などにもある程度耐えられる様に設計をしていくという考え方
"""

# 相関係数が高いものもあるが、今回はお試しなのでそのまま使う
#3つの結果を加算
submit_ansemble = pd.read_csv('../input/titanic/gender_submission.csv')
submit_ansemble['Survived'] = submit1['Survived'] + submit2['Survived'] + submit3['Survived']

submit_ansemble.head()


# 今回のアンサンブルでは、これまでの各モデルによる予測結果(csv)を3つ用意し、各入力に対して0, 1の判別結果を多数決し、その結果を採用している