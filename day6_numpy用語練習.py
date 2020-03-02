import os #建立文件/目錄路徑
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#data path
dir_data = 'C:\python data\Part01'  #必須使用os.path.join方式寫路徑部分，不然會亂碼
f_app = os.path.join(dir_data ,'application_train.csv')
print("Path of read in data:"+f_app)
app_train = pd.read_csv(f_app)

#挑出app_train裡面是連續數值得columns # dtype = int64 & float64
numeric_columns = list(app_train.columns[list(app_train.dtypes.isin([np.dtype('int64'),np.dtype('float64')]))]) #選擇連續型數值
#把是連續數值的columns裡面只有0，1的欄位拿掉   #回報為true的columns留下                  #apply依矩陣方式傳入  #將重複數去掉，該column有兩種以上數字則回報true
numeric_columns = list(app_train[numeric_columns].columns[list(app_train[numeric_columns].apply(lambda x:len(x.unique())!=2 ))])
## numeric_columns 是以index方式存在，指向我們找出的columns，不是直接儲存數值
print("Numbers of remain columns %i" % len(numeric_columns)) #顯示數量


###-----------     #以index指向app_train的columns，並繪圖   共73張-----------
##for col in numeric_columns:
##    temp = pd.DataFrame(app_train[col])
##    temp.boxplot()
##    plt.show()


print(app_train['REGION_POPULATION_RELATIVE'].describe())  #顯示'REGION_POPULATION_RELATIVE'內數值分布

# 繪製 Empirical Cumulative Density Plot (ECDF)
      #dataframe選擇REGION_POPULATION_RELATIVE #重複數值出現次數 #升序排序 #cumsum 累進加法
cdf = app_train.REGION_POPULATION_RELATIVE.value_counts().sort_index().cumsum()
plt.plot(list(cdf.index), cdf/cdf.max())
plt.xlabel('Value')
plt.ylabel('ECDF')
plt.ylim([-0.05,1.05]) # 限制顯示圖片的範圍
plt.show()

app_train['REGION_POPULATION_RELATIVE'].hist()
plt.show()

#a = app_train['REGION_POPULATION_RELATIVE'].value_counts().sort_index(ascending = False) #降序排列
