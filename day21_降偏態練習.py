# 做完特徵工程前的所有準備 (與前範例相同)
import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
#建立路徑
data_path = 'C:/python data/Part02/'
df_train = pd.read_csv(data_path + 'titanic_train.csv')
df_test = pd.read_csv(data_path + 'titanic_test.csv')

####數據清洗####

train_Y = df_train['Survived']
ids = df_test['PassengerId']
df_train = df_train.drop(['PassengerId', 'Survived'] , axis=1)
df_test = df_test.drop(['PassengerId'] , axis=1)
df = pd.concat([df_train,df_test])

#只取 int64, float64 兩種數值型欄位, 存於 num_features 中
num_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'float64' or dtype == 'int64':
        num_features.append(feature)
print(f'{len(num_features)} Numeric Features : {num_features}\n')

# 削減文字型欄位, 只剩數值型欄位
df = df[num_features]
df = df.fillna(0)
MMEncoder = MinMaxScaler()
train_num = train_Y.shape[0]
df.head()

####數據清洗結束####

# 顯示 Fare 與目標值的散佈圖
import seaborn as sns
import matplotlib.pyplot as plt
sns.distplot(df['Fare'][:train_num])
plt.show()

# 計算基礎分數
df_mm = MMEncoder.fit_transform(df)
train_X = df_mm[:train_num]
estimator = LogisticRegression()
score1 = cross_val_score(estimator, train_X, train_Y, cv=5).mean()
print('base score: '+str(score1))


###以上基礎分析，接下來開始去偏態比較差異###

#對Fare使用log1p去偏態
df_log = copy.deepcopy(df)  #複製dataframe
df_log['Fare'] = np.log1p(df_log['Fare'])
sns.distplot(df_log['Fare'][:train_num])
plt.show()
#重新計算分數
df_mm = MMEncoder.fit_transform(df_log)
train_X = df_mm[:train_num]
estimator = LogisticRegression()
score2 = cross_val_score(estimator, train_X, train_Y, cv=5).mean()
print('log1p score: '+str(score2))
##交叉驗證分數上升1


#對Fare使用boxcox去偏態
from scipy import stats
df_fixed = copy.deepcopy(df)  #複製dataframe
##因為Fare裡面有0(缺值補0)，boxcox需要全部正數，為了不破壞數據結構，各項+1平移
df_fixed['Fare'] = stats.boxcox(df_fixed['Fare']+1)[0] #後面需加[0]，才不會跳長度異常

sns.distplot(df_fixed['Fare'][:train_num])
plt.show()
#重新計算分數
df_mm = MMEncoder.fit_transform(df_fixed)
train_X = df_mm[:train_num]
estimator = LogisticRegression()
score3 = cross_val_score(estimator, train_X, train_Y, cv=5).mean()
print('boxcox score: '+str(score3))
##交叉驗證分數上升0.9
