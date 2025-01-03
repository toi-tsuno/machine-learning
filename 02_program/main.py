import pandas as pd 
import statsmodels.api as sm 
from sklearn.linear_model import LogisticRegression
import category_encoders as ce
import missingno as msno
from IPython.display import display

# データのインポート
train_data = pd.read_csv('01_data/titanic/train.csv')
test_data = pd.read_csv('01_data/titanic/test.csv')

print("")


# 欠損値の確認
# print('train_dataのNULL値は以下です\n',train_data.isnull().sum())
# print('test_dataのNULL値は以下です\n',train_data.isnull().sum())

# 欠損値を平均値で補完

train_data_fillna = train_data.fillna(train_data.mean(numeric_only=True))
test_data_fillna = test_data.fillna(test_data.mean(numeric_only=True))

train_data_fillna.fillna(train_data.mode().iloc[0],inplace=True)
test_data_fillna.fillna(test_data.mode().iloc[0],inplace=True)

# print('train_dataの変換後のNULL値は以下です\n',train_data_fillna.isnull().sum())
# print('test_dataの変換後のNULL値は以下です\n',test_data_fillna.isnull().sum())


# データ前処理
# Encodeする列を指定
encode_cols = ['Sex','Embarked']

# 説明変数の設定
exp_val = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
# exp_val = ['Pclass','Sex','Age','SibSp','Embarked']
x = train_data_fillna[exp_val]

# 序数をカテゴリに付与して変換する
ce_oe = ce.OrdinalEncoder(cols=encode_cols,handle_unknown='impute')
x_ordinal = ce_oe.fit_transform(x)

# mappingしたものは以下で確認できる
print('mappingしたものは以下です\n',pd.DataFrame(ce_oe.category_mapping))

# 説明変数のデータの中1からなる列を追加（もともと切片項が存在している場合も追加する）
x_ordinal_number = sm.add_constant(x_ordinal,has_constant='add')

# 目的変数の設定
obj_val = ['Survived']
y = train_data_fillna[obj_val]

# ロジスティック回帰モデルの作成
model = sm.Logit(y,x_ordinal_number)
result = model.fit()

# サマリの表示
print(result.summary())

