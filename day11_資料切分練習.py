import os #建立文件/目錄路徑
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#data path
dir_data = 'C:\python data\Part01'  #必須使用os.path.join方式寫路徑部分，不然會亂碼
f_app = os.path.join(dir_data ,'application_train.csv')
print("Path of read in data:"+f_app)
app_train = pd.read_csv(f_app)

# 資料整理 ( 'DAYS_BIRTH'全部取絕對值 )
app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])

# 根據年齡分成不同組別 (年齡區間 - 還款與否)
age_data = app_train[['TARGET', 'DAYS_BIRTH']] # subset
age_data['YEARS_BIRTH'] = (age_data['DAYS_BIRTH']/365) # day-age to year-age

#自 20 到 70 歲，切 11 個點 (得到 10 組)
bin_cut =  [20,25,30,35,40,45,50,55,60,65,70]
age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = bin_cut) 

# 顯示不同組的數量
print(age_data['YEARS_BINNED'].value_counts())

year_group_sorted = np.sort(age_data['YEARS_BINNED'].unique()) #重新排列

plt.figure(figsize=(8,6))
for i in range(len(year_group_sorted)):  #用.loc抓出YEARS_BINNED是20~25區間及TARGET = 0的YEARS_BIRTH值
    sns.distplot(age_data.loc[(age_data['YEARS_BINNED'] == year_group_sorted[i]) & (age_data['TARGET'] == 0), 'YEARS_BIRTH'], label = str(year_group_sorted[i]))
                                                                                #&\換行，等於&##
    sns.distplot(age_data.loc[(age_data['YEARS_BINNED'] == year_group_sorted[i]) &\
                              (age_data['TARGET'] == 1), 'YEARS_BIRTH'], label = str(year_group_sorted[i]))
plt.title('KDE with Age groups')
plt.show()

age_groups  = age_data.groupby('YEARS_BINNED').mean() #計算各區間的平均值
print(age_groups)

plt.figure(figsize = (8, 8))

# 以年齡區間為 x, target 為 y 繪製 barplot
px = age_groups.index.astype(str)
py = age_groups['TARGET']
sns.barplot(px, py)

# Plot labeling
plt.xticks(rotation = 75) #旋轉坐標軸標籤
plt.xlabel('Age Group (years)')
plt.ylabel('Failure to Repay (%)')
plt.title('Failure to Repay by Age Group');
