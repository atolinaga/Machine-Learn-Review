# 載入基本套件
import pandas as pd
import numpy as np

# 讀取訓練與測試資料
data_path = 'C:\python data\Part02'
df_train = pd.read_csv('C:/python data/Part02/titanic_train.csv')
df_test = pd.read_csv('C:/python data/Part02/titanic_test.csv')
##df_train.shape

# 重組資料成為訓練 / 預測用格式
train_Y = df_train['Survived']
ids = df_test['PassengerId']
df_train = df_train.drop(['PassengerId', 'Survived'] , axis=1)
df_test = df_test.drop(['PassengerId'] , axis=1)
df = pd.concat([df_train,df_test])
##df.head()

dtype_df = df.dtypes.reset_index()  #重置columns
dtype_df.columns = ["Count", "Column Type"] #輸入columns名稱
dtype_df = dtype_df.groupby("Column Type").aggregate('count').reset_index() #aggregate('count')類似count，把各項columns總數算出來
##dtype_df

int_features = []
float_features = []
object_features = []
for dtype, feature in zip(df.dtypes, df.columns): #zip將df.dtypes, df.columns對接(0對0,1對1)
    if dtype == 'float64':
        float_features.append(feature)
    elif dtype == 'int64':
        int_features.append(feature)
    else:
        object_features.append(feature)
print(f'{len(int_features)} Integer Features : {int_features}\n')
print(f'{len(float_features)} Float Features : {float_features}\n')
print(f'{len(object_features)} Object Features : {object_features}')
