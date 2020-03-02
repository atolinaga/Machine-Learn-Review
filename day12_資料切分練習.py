##pandas cut函數練習

import os #建立文件/目錄路徑
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

###data path
##dir_data = 'C:\python data\Part01'  #必須使用os.path.join方式寫路徑部分，不然會亂碼
##f_app = os.path.join(dir_data ,'application_train.csv')
##print("Path of read in data:"+f_app)
##app_train = pd.read_csv(f_app)

ages = pd.DataFrame({"age": [18,22,25,27,7,21,23,37,30,61,45,41,9,18,80,100]})
ages['equal_width_age'] = pd.cut(ages['age'],4) #以數據等分切割

ages['equal_freq_age'] = pd.qcut(ages['age'],4) #以數量均分切割


cut = [0,10,20,30,50,100]

ages['custom_cut'] = pd.cut(ages['age'],bins = cut)
