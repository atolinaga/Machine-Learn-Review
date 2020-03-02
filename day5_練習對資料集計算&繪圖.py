import os #建立文件/目錄路徑
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#data path
dir_data = 'C:\python data\Part01'  #必須使用os.path.join方式寫路徑部分，不然會亂碼
f_app = os.path.join(dir_data ,'application_train.csv')
print("Path of read in data:"+f_app)
app_train = pd.read_csv(f_app)

#觀看資料集，針對有興趣的欄位做切分
income = app_train['AMT_CREDIT']
total = sum(income)
average = income.mean()
std = income.std()

plt.hist(income) #直方圖
plt.show()
