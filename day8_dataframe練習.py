import os #建立文件/目錄路徑
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#data path
dir_data = 'C:\python data\Part01'  #必須使用os.path.join方式寫路徑部分，不然會亂碼
f_app = os.path.join(dir_data ,'application_train.csv')
print("Path of read in data:"+f_app)
app_train = pd.read_csv(f_app)

#對資料集進行切分
cut_rule = [-1,0,2.1,5.1,np.inf]
app_train['CNT_CHILDREN_GROUP'] = pd.cut(app_train['CNT_CHILDREN'].values, cut_rule, include_lowest=True)
#依切分狀況分組
app_train.groupby(['CNT_CHILDREN_GROUP'])


grp = app_train['CNT_CHILDREN_GROUP']
grouped_df = app_train.groupby(grp)['AMT_INCOME_TOTAL'] #依CNT_CHILDREN_GROUP進行分組

plt_column = ['AMT_INCOME_TOTAL']
plt_by = ['CNT_CHILDREN_GROUP']

#根據有小孩子的數量區間比較收入多寡，並畫圖
app_train.boxplot(column=plt_column, by = plt_by, showfliers = False,figsize = (5,5))
plt.suptitle('AMT_INCOME_TOTAL by children')
plt.show()


#將AMT_INCOME_TOTAL數值轉成Z轉換
mean = app_train['AMT_INCOME_TOTAL'].mean()
std = app_train['AMT_INCOME_TOTAL'].std()
app_train['AMT_INCOME_TOTAL_Z'] = app_train['AMT_INCOME_TOTAL'].apply(lambda x:((x - mean)/std))

print(app_train['AMT_INCOME_TOTAL_Z'])




