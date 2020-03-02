import os #建立文件/目錄路徑
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data1 = (np.random.rand(10,10))*2-1

sns.heatmap(data1, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.title('random data')
plt.show()

data2 = (np.random.rand(1000,3))*2-1
indice = np.random.choice([0,1,2], size=1000)
plot_data = pd.DataFrame(data2, indice)

grid = sns.PairGrid(data = plot_data, height = 3, diag_sharey=False)
grid.map_upper(plt.scatter , alpha = 0.2)
grid.map_diag(sns.kdeplot)
grid.map_lower(sns.kdeplot , cmap = plt.cm.OrRd_r)

plt.show()


data3 = (np.random.randn(1000,3))*2-1 #heatmap
indice = np.random.choice([0,1,2], size=1000) #隨機生成0~2index
plot_data = pd.DataFrame(data3, indice)

grid = sns.PairGrid(data = plot_data, height = 3, diag_sharey=False)
grid.map_upper(plt.scatter , alpha = 0.2)
grid.map_diag(sns.kdeplot)
grid.map_lower(sns.kdeplot , cmap = plt.cm.OrRd_r)

plt.show()
