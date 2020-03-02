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

housetype = app_train['HOUSETYPE_MODE'].unique() #列出有幾種房型
nrows = len(housetype) #決定subplot圖數量
ncols = nrows // 2

plt.figure(figsize = (10,30))
for i in range(len(housetype)):
    plt.subplot(nrows,ncols,i+1)
    app_train.loc[app_train['HOUSETYPE_MODE'] == housetype[i],'AMT_CREDIT'].hist()
    plt.title(str(housetype[i]))
plt.show()
