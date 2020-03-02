import numpy as np
#np.random.seed(1)

import matplotlib
import matplotlib.pyplot as plt

#--弱相關--
x = np.random.randint(0,50,1000) #生成0~50 1000個數字
y = np.random.randint(0,50,1000)
corrcoef1 = np.corrcoef(x,y) #找相關係數
plt.plot(x,y,'b.')
plt.show()


#--正相關--
x1 = np.random.randint(0,50,1000)
y1 = x1 + np.random.randint(0,10,1000)

plt.plot(x1,y1,'b.')
plt.show()
corrcoef2 = np.corrcoef(x1,y1) #找相關係數


#--負相關
x2 = np.random.randint(0,50,1000)
y2 = -x2 + np.random.randint(0,10,1000)
plt.plot(x2,y2,'b.')
plt.show()
corrcoef3 = np.corrcoef(x2,y2) #找相關係數


print('無相關')
print(corrcoef1)
print('正相關')
print(corrcoef2)
print('負相關')
print(corrcoef3)

