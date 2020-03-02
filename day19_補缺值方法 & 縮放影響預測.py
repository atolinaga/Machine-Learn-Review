# 做完特徵工程前的所有準備 (與前範例相同)
import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

df_train = pd.read_csv('C:/python data/Part02/titanic_train.csv')
df_test = pd.read_csv('C:/python data/Part02/titanic_test.csv')

train_Y = df_train['Survived']
ids = df_test['PassengerId']
df_train = df_train.drop(['PassengerId', 'Survived'] , axis=1)
df_test = df_test.drop(['PassengerId'] , axis=1)
df = pd.concat([df_train,df_test]) #把train跟test連起來，成為總數據

#只取 int64, float64 兩種數值型欄位, 存於 num_features 中
num_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'float64' or dtype == 'int64':
        num_features.append(feature)
print(f'{len(num_features)} Numeric Features : {num_features}\n')

df = df[num_features]#只留下float & int類型資料
train_num = train_Y.shape[0] #導入訓練集Y(生存人數)
#導入羅吉斯迴歸分析
estimator = LogisticRegression()
#缺值補-1
df_m1 = df.fillna(-1)
train_X = df_m1[:train_num] #由前致後擷取數據(0~890 共891數據)
#交叉驗證
score1 = cross_val_score(estimator, train_X, train_Y, cv=5).mean() #cv5 分成五組

#缺值補0
df_0 = df.fillna(0)
train_X = df_0[:train_num]
estimator = LogisticRegression()
score2 = cross_val_score(estimator, train_X, train_Y, cv=5).mean()

#缺值補上平均數
df_mn = df.fillna(df.mean())
train_X = df_mn[:train_num]
estimator = LogisticRegression()
score3 = cross_val_score(estimator, train_X, train_Y, cv=5).mean()

print('缺值補-1: '+str(score1))
print('缺值補0: '+str(score2))
print('缺值補平均數: '+str(score3))








#2，分析不同數據處理方式影響數值

#對df_0做最小最大化 ==> 最大為1，最小為0，其他以百分比分配
df_temp_min = MinMaxScaler().fit_transform(df_0) 
train_X_min = df_temp_min[:train_num]
score4 = cross_val_score(estimator, train_X_min, train_Y, cv=5).mean()


#對df_0做標準化
df_temp_s = StandardScaler().fit_transform(df_0) 
train_X_s = df_temp_s[:train_num]
score5 = cross_val_score(estimator, train_X_s, train_Y, cv=5).mean()
print('')
print('原數值: '+str(score2))
print('最大最小化: '+str(score4))
print('Z-score標準化: '+str(score5))





