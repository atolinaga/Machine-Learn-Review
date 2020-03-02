# 做完特徵工程前的所有準備 (與前範例相同)
import pandas as pd
import numpy as np
import copy, time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.ensemble import GradientBoostingClassifier #導入梯度提升模擬法
from sklearn.preprocessing import LabelEncoder

data_path = 'C:/python data/Part02/'
df_train = pd.read_csv(data_path + 'titanic_train.csv')
df_test = pd.read_csv(data_path + 'titanic_test.csv')

train_Y = df_train['Survived']
ids = df_test['PassengerId']
df_train = df_train.drop(['PassengerId', 'Survived'] , axis=1)
df_test = df_test.drop(['PassengerId'] , axis=1)
df = pd.concat([df_train,df_test])

##數據清洗##
#只取類別值 (object) 型欄位, 存於 object_features 中
object_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'object':
        object_features.append(feature)
print(f'{len(object_features)} Numeric Features : {object_features}\n')

# 只留類別型欄位
df = df[object_features]
df = df.fillna('None')
train_num = train_Y.shape[0]

#設定分析方法
estimatorLogi = LogisticRegression()
estimatorLine = LinearRegression()
estimatorGBC = GradientBoostingClassifier()

# 標籤編碼
df_temp = pd.DataFrame() #做出空的dataframe
for c in df.columns:
    df_temp[c] = LabelEncoder().fit_transform(df[c]) #把df該欄編碼後塞進df_temp #文字轉編號代表
train_X = df_temp[:train_num]

#迴歸並交叉分析數據正確性
start = time.time()
scoreLogi = cross_val_score(estimatorLogi, train_X, train_Y, cv=5).mean()
logiTime = time.time()-start
start = time.time()
scoreLine = cross_val_score(estimatorLine, train_X, train_Y, cv=5).mean()
lineTime = time.time()-start
start = time.time()
scoreGBC = cross_val_score(estimatorGBC, train_X, train_Y, cv=5).mean()
GBCTime = time.time()-start
print(f'shape : {train_X.shape}')
print('score for LogisticRegression: '+ str(scoreLogi))
print('Spend time: '+str(logiTime))
print('score for LinearRegression: '+ str(scoreLine))
print('Spend time: '+str(lineTime))
print('score for GBC: '+ str(scoreGBC))
print('Spend time: '+str(GBCTime))


print('===========================================================')
# 獨熱編碼
df_temp = pd.get_dummies(df) #一行即設定好獨熱編碼
train_X = df_temp[:train_num]

#迴歸並交叉分析數據正確性
start = time.time()
scoreLogi = cross_val_score(estimatorLogi, train_X, train_Y, cv=5).mean()
logiTime = time.time()-start
start = time.time()
scoreLine = cross_val_score(estimatorLine, train_X, train_Y, cv=5).mean()
lineTime = time.time()-start
start = time.time()
scoreGBC = cross_val_score(estimatorGBC, train_X, train_Y, cv=5).mean()
GBCTime = time.time()-start
print(f'shape : {train_X.shape}')
print('score for LogisticRegression: '+ str(scoreLogi))
print('Spend time: '+str(logiTime))
print('score for LinearRegression: '+ str(scoreLine))
print('Spend time: '+str(lineTime))
print('score for GBC: '+ str(scoreGBC))
print('Spend time: '+str(GBCTime))
print('===========================================================')


#均值編碼 ==> 把[Survived]同group的數值取平均，並以均值編碼
data = pd.concat([df[:train_num], train_Y], axis=1) #數據處理，把train集的Survived加回去
 
for c in df.columns:
    mean_df = data.groupby([c])['Survived'].mean().reset_index()
    mean_df.columns = [c, f'{c}_mean']
    data = pd.merge(data, mean_df, on=c, how='left')
    data = data.drop([c] , axis=1)
data = data.drop(['Survived'] , axis=1)
train_X = data[:train_num]
#迴歸並交叉分析數據正確性
start = time.time()
scoreLogi = cross_val_score(estimatorLogi, train_X, train_Y, cv=5).mean()
logiTime = time.time()-start
start = time.time()
scoreLine = cross_val_score(estimatorLine, train_X, train_Y, cv=5).mean()
lineTime = time.time()-start
start = time.time()
scoreGBC = cross_val_score(estimatorGBC, train_X, train_Y, cv=5).mean()
GBCTime = time.time()-start
print(f'shape : {train_X.shape}')
print('score for LogisticRegression: '+ str(scoreLogi))
print('Spend time: '+str(logiTime))
print('score for LinearRegression: '+ str(scoreLine))
print('Spend time: '+str(lineTime))
print('score for GBC: '+ str(scoreGBC))
print('Spend time: '+str(GBCTime))





