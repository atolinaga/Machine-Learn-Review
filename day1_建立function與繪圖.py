# 填寫一個函式計算MSE (Mean Square Error)

import numpy as np
import matplotlib.pyplot as plt


#--------生成function計算MSE & MAE------------
def MSE(a,b):   
    mse = sum(a-b)**2 / len(a)
    return mse

def MAE(a,d): #Mean absolute error
    mae = sum(abs(a-b)) / len(a)
    return mae

#---------------------------------------------

#生成隨機資料
w = 3   #生成直線方程式，數據合理化
b = 0.5 

x_lin = np.linspace(0,100,101)
#x_lin = range(0,101)
randnum = np.random.randn(101)
y = (x_lin + randnum * 5 ) * w + b

#繪圖------------------------------------------
##plt.plot(x_lin,y,'b.',label = 'data point')
##plt.title('test data')
##plt.legend(loc = 2) # 貼上標籤(label)
##plt.show()

y_hat = x_lin * w + b
plt.plot(x_lin,y,'b.',label = 'data')
plt.plot(x_lin,y_hat,'r-',label = 'prediction')
plt.title('predict')
plt.legend(loc = 2)
plt.show()

#-----------------------------------------------

MSE =MSE(y,y_hat)
MAE =MAE(y,y_hat)
print(MSE)
print(MAE)




