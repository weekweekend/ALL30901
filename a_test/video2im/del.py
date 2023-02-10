import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from dtaidistance import dtw
# 这两行代码解决 plt 中文显示的问题
import os
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.rcParams.update({'xtick.bottom' : False, 'axes.titlepad':5})


data=[]
path = 'csv/'
for root,dirs,files in os.walk(path):
  print(files)
  for file in files:
    if file.endswith(".csv"):
      df_orig = pd.read_csv(path+file, parse_dates=['time'], index_col='time')
      # df_loess_2 = pd.DataFrame(lowess(df_orig.val, np.arange(len(df_orig.val)), frac=0.5)[:, 1], index=df_orig.index, columns=['val'])
      data.append([file[:-4], df_orig.val.values])
# name = ['time', 'val']
# test = pd.DataFrame(columns=name, data = csv)
# test.to_csv('csv/'+file_name[f]+'_'+tp+'.csv')

 
# 聚类数
k=3
num = 1
 

#在数据最大最小值中随机生成k个初始聚类中心，保存为t

init=[0,10,11]

for i in range(k):
  y=data[init[i]][1]
  x = range(0,len(y),1)
  z1 = np.polyfit(x, y, 5) #用4次多项式拟合，输出系数从高到0
  p1 = np.poly1d(z1) #使用次数合成多项式
  y_pre = p1(x)
  print(i,dtw.distance_fast(y, y_pre))
  fig = plt.figure(figsize= (22,8),dpi = 80)
  x = range(0,len(y),1)
  plt.plot(x, y_pre)
  plt.plot(x, y)
  plt.savefig('fit'+str(i)+'.png')


 