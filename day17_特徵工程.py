# 程式區塊 A
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

data_path = 'C:\python data\Part02'
df_train = pd.read_csv('C:/python data/Part02/titanic_train.csv')
df_test = pd.read_csv('C:/python data/Part02/titanic_test.csv')
##df_train.shape

# 程式區塊 B
train_Y = df_train['Survived']
ids = df_test['PassengerId']
df_train = df_train.drop(['PassengerId', 'Survived'] , axis=1)
df_test = df_test.drop(['PassengerId'] , axis=1)
df = pd.concat([df_train,df_test])

# 程式區塊 C ==> 特徵工程
LEncoder = LabelEncoder()
MMEncoder = MinMaxScaler()
for c in df.columns:
    df[c] = df[c].fillna(-1) #填空值
    if df[c].dtype == 'object':
        df[c] = LEncoder.fit_transform(list(df[c].values))
    df[c] = MMEncoder.fit_transform(df[c].values.reshape(-1, 1))
##df.head()

# 程式區塊 D
train_num = train_Y.shape[0]
train_X = df[:train_num] #把df分割(~891與891~)
test_X = df[train_num:]

from sklearn.linear_model import LogisticRegression
estimator = LogisticRegression()
estimator.fit(train_X, train_Y)
pred = estimator.predict(test_X)


# 程式區塊 E
sub = pd.DataFrame({'PassengerId': ids, 'Survived': pred})
##sub.to_csv('titanic_baseline.csv', index=False) 

a = sub.loc[sub['Survived'] == 1] #截取出預測存活人

