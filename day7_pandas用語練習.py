import os #建立文件/目錄路徑
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#data path
dir_data = 'C:\python data\Part01'  #必須使用os.path.join方式寫路徑部分，不然會亂碼
f_app = os.path.join(dir_data ,'application_train.csv')
print("Path of read in data:"+f_app)
app_train = pd.read_csv(f_app)

# 1: 計算 AMT_ANNUITY 的 q0 - q100 (由小到大切成100分)
hun_num = list(range(0,101))

#np.percentile:把資料集改成q0~q100所對應的數值
##q_all = [np.percentile(app_train[~app_train['AMT_ANNUITY'].isnull()]['AMT_ANNUITY'], q = i) for i in hun_num]
                    #一樣用index方式(ture)回傳AMT_ANNUITY內不是負值的欄位   #帶入1~100(矩陣形式)計算q0~q100
q_all = np.percentile(app_train[~app_train['AMT_ANNUITY'].isnull()]['AMT_ANNUITY'],hun_num)
#data = pd.DataFrame({'q': list(range(101)),'value': q_all})

#顯示該行缺數值的總數                                                     #加總返回是true的數量
print("Before replace NAs, numbers of row that AMT_ANNUITY is NAs: %i" % sum(app_train['AMT_ANNUITY'].isnull()))
q_50 = q_all[50] #以中位數填補

#以loc方式取出app_train:row是null值的欄位，columns是'AMT_ANNUITY'這欄   #用q_50取代
app_train.loc[app_train['AMT_ANNUITY'].isnull(),'AMT_ANNUITY'] = q_50 #此行會直接修改app_train

print("After replace NAs, numbers of row that AMT_ANNUITY is NAs: %i" % sum(app_train['AMT_ANNUITY'].isnull()))
print("將AMT_ANNUITY此欄的空值用中位數取代")

#常態化AMT_ANNUITY(-1~1之間)
print("=====================================")
print("== Original data range ==")
print(app_train['AMT_ANNUITY'].describe())
value = app_train['AMT_ANNUITY'].values  #把AMT_ANNUITY值取出變成矩陣

#建立常態化function
def normalize_value(a):
    ans = ((a - np.min(a)) / (np.max(a)-np.min(a))-0.5) * 2
    return ans

app_train['AMT_ANNUITY_NORMALIZED'] = normalize_value(value) #插入一欄常態化columns

##app_train['AMT_ANNUITY_NORMALIZED'].hist() 繪圖確認
##plt.show()

print("=====================================")
print("== Normalized data range ==")
print(app_train['AMT_ANNUITY_NORMALIZED'].describe())







