import os #建立文件/目錄路徑
import numpy as np
import pandas as pd

#data path
dir_data = 'C:\python data\Part01'  #必須使用os.path.join方式寫路徑部分，不然會亂碼
f_app = os.path.join(dir_data ,'application_train.csv')
print("Path of read in data:"+f_app)
app_train = pd.read_csv(f_app)

#----------------------截取部分數據編碼
sub_train = pd.DataFrame(app_train['WEEKDAY_APPR_PROCESS_START']) #把app_train裡面的WEEKDAY_APPR_PROCESS_START這欄抓出來
print("切分資料集")
print(sub_train.shape)
print(sub_train.head(10))

#以pandas進行獨熱編碼 (one-hot)

sub_test = pd.get_dummies(sub_train)
print("獨熱編碼")
print(sub_test.shape)
print(sub_test.head(10))
