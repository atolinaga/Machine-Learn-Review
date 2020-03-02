# 做完特徵工程前的所有準備 (與前範例相同)
import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

df_train = pd.read_csv('C:\python data\Part02\house_train.csv.gz')

train_Y = np.log1p(df_train['SalePrice'])
df = df_train.drop(['Id','SalePrice'],axis = 1)
##數據清洗##

num_feature = []
for dtype,feature in zip(df.dtypes,df.columns) :
    if dtype == 'int64' or dtype == 'float64':
        num_feature.append(feature)
print('Have '+str(len(num_feature))+' columns.\n'+str(num_feature))

#把總數據集改為只剩數字
df = df[num_feature]
#空值填補-1
df = df.fillna(-1)
#進行最小最大化
MMEncoder = MinMaxScaler()
#紀錄目標值數量
train_num = train_Y.shape[0]


##繪圖##
import seaborn as sns
import matplotlib.pyplot as plt
x = df['1stFlrSF'][:train_num]
y = train_Y
plt.plot(x,y,'b.')
plt.show()


##數據分析-線性回歸，觀察分數
train_X = MMEncoder.fit_transform(df)
estimator = LinearRegression()
score = cross_val_score(estimator,train_X,train_Y,cv=5).mean()

print(score)


# 將 1stFlrSF 限制在你覺得適合的範圍內, 調整離群值
df['1stFlrSF'] = df['1stFlrSF'].clip(100,2500)
sns.regplot(x = df['1stFlrSF'][:train_num], y=train_Y)
plt.show()
# 做線性迴歸, 觀察分數
train_X = MMEncoder.fit_transform(df)
estimator = LinearRegression()
cross_val_score(estimator, train_X, train_Y, cv=5).mean()


# 將 1stFlrSF 限制在你覺得適合的範圍內, 捨棄離群值
keep_indexs = (df['1stFlrSF']> 100) & (df['1stFlrSF']< 2500)
df = df[keep_indexs]
train_Y = train_Y[keep_indexs]
sns.regplot(x = df['1stFlrSF'], y=train_Y)
plt.show()

# 做線性迴歸, 觀察分數
train_X = MMEncoder.fit_transform(df)
estimator = LinearRegression()
cross_val_score(estimator, train_X, train_Y, cv=5).mean()
