from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time
import statsmodels.api as sm 
import math

plt.rcParams.update({'xtick.bottom' : False, 'axes.titlepad':5})

#!/usr/bin/python3
# -*- coding: utf-8 -*-
# author:SingWeek
 
def samelen(data,length=15000):
    """
    数据长度归一化
    :param data:以数组形式输入需要归一化的数据
    :param length:归一化后的长度
    :return:result：归一化后的数据返回值
    """
    result=[]
    truelen=len(data)
    if truelen==length:
        return data
    elif truelen>length:
        integer=int(truelen/length)
        decimal=truelen-length*integer
        mid=decimal/truelen
        tmp=0
        for i in range(0,truelen,integer):
            tmp+=mid
            if tmp-1>=0:
                tmp-=1
            else:
                result.append(data[i])
        resulttmp=len(result)
        if resulttmp==length:
            return result
        elif resulttmp<length:
            for i in range(length-resulttmp):
                result.append(0.0)
            return result
        else:
            return result[:length]
    else:
        integer=int(length/truelen)
        decimal=length-truelen*integer
        mid=decimal/truelen
        tmp=0
        for i in range(truelen):
            for j in range(integer):
                result.append(data[i])
            tmp+=mid
            if tmp-1>=0:
                result.append(data[i])
                tmp-=1
        resulttmp=len(result)
        if resulttmp==length:
            return result
        elif resulttmp<length:
            for i in range(length-resulttmp):
                result.append(0.0)
            return result
        else:
            return result[:length]


plt.figure(figsize=(15, 5), dpi=80)
# 取所有原数据
path = 'csvo/30/'
for root,dirs,files in os.walk(path):
  files = sorted(files)
  for i in range(0,len(files),1):
    if files[i].endswith(".csv"):
      file_name = files[i][:-4]
      # Import
      # 读取原数据
      df_o = pd.read_csv(path+files[i], parse_dates=['time'], index_col='time')
      df = df_o.val.values 
      val = samelen(df) 
      lowess=sm.nonparametric.lowess
      val = lowess(val, np.arange(len(val)), frac=0.1)[:, 1]  
      val = (val - val.min())/(val.max()-val.min())
      x = range(0,len(val),1)
      plt.plot(x, val, lw = 10, color='gray', alpha=0.2)
      plt.tick_params(labelsize=14) 
plt.savefig('tmp/plotall.png')

# fig, axes = plt.subplots(3,1, figsize=(10, 6), sharex=True, dpi=120)
# df_orig['s'].plot(ax=axes[0],  title='before')
# df_orig['s2'].plot(ax=axes[1], title='affter Smoothed 2%')
# df_orig['s5'].plot(ax=axes[2], title='affter Smoothed 5%')
# fig.suptitle('Slope before and after smoothing', y=0.95, fontsize=14)
# plt.show()
# plt.savefig('res/'+ file_name+"_slope.png")
