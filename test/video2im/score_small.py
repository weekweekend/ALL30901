from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time
import statsmodels.api as sm 
import math

plt.rcParams.update({'xtick.bottom' : False, 'axes.titlepad':5})

types = ['shell','out','disc','line','pipe','sensor']
types = ['out']
file_name = ['0-6','10-14','14-18','18-14','14-10','14-6-14','14-0']
file_name = ['14-18']

# file_name = ['0-6','10-14','14-18','18-14','14-10','14-6-14','14-0','zl1','zl2','zr1','fl1','fl2','fr1']

note=open('score_small.txt',mode='a')

# 取所有原数据
path = 'csvo/30/'
for root,dirs,files in os.walk(path):
  files = sorted(files)
  for i in range(0,len(files),1):
    if files[i].endswith(".csv"):
      file_name = files[i][:-4]
      # Import
      note.write(file_name+'\n')
      # 读取原数据
      df_o = pd.read_csv(path+files[i], parse_dates=['time'], index_col='time')
      df = df_o.val.values 
      ll = len(df)//3001
      df = df[0:ll*3001]
      res = []
      rate = []
      start =time.perf_counter()
      # 以步长为3000  截取 6000长度段
      for cut in range(0,len(df)-3001,3001):
        s = df[cut:cut+6002]
        lowess=sm.nonparametric.lowess
        s0 = lowess(s, np.arange(len(s)), frac=0.09)[:, 1]  
        s1=[]
        s2=[]
        for ci in range(1,len(s0),1):
          s1.append(s0[ci]-s0[ci-1])
        for ci in range(1,len(s1),1):
          s2.append(s1[ci]-s1[ci-1])

        # fig = plt.figure(figsize= (22,8),dpi = 80)
        # x = range(0,len(s2),1)
        # y = s2
        # plt.plot(x, y)
        # plt.savefig('tmp/test2.png')
        # aa
        
        s1=np.array(s1)[0:-1]
        s2=np.array(s2)

        print('len:',len(s0),len(s1),len(s2))
        ans = 0
        for j in range(0,6000,1200):
          t1 = s1[j:j+1200]
          t2 = s2[j:j+1200]
          tt = t1
          tt[tt>0]=0
          t2 = np.multiply(t2,tt)
          t2 = t2[t2<0]*-1
          # 一阶导数小于0的帧数占比（下降趋势）
          # 一阶导数小于0的值的绝对值的和（）下降趋势
          # 对应一阶导数小于0的位置中二阶导数小于0部分的绝对值的和（下降趋势中）
          ans += sum(val<0 for val in t1)/12/5*(-np.sum(t1[t1<0]))*(np.sum(t2))*100*1
    
        for j in range(0,6000,2000):
          t1 = s1[j:j+2000]
          t2 = s2[j:j+2000]
          tt = t1
          tt[tt>0]=0
          t2 = np.multiply(t2,tt)
          t2 = t2[t2<0]*-1
          ans += sum(val<0 for val in t1)/20/3*(-np.sum(t1[t1<0]))*(np.sum(t2))*100*3

        t1 = s1[0:6000]
        t2 = s2[0:6000]
        tt = t1
        tt[tt>0]=0
        t2 = np.multiply(t2,tt)
        t2 = t2[t2<0]*-1
        ans += sum(val<0 for val in t1)/60/1*(-np.sum(t1[t1<0]))*(np.sum(t2))*100*6
        print(ans)
        res.append(ans)
        if ans<=1:
         rate.append((31**ans-1)/60) 
        elif ans<=700 and ans>1:
         rate.append((math.exp((ans-1)/30)- math.exp(-((ans-1)/30)))/(math.exp((ans-1)/30)+math.exp(-((ans-1)/30)))/2+0.5) 
        else:
          rate.append(1)
        
      end = time.perf_counter()
      note.write(str(res))
      note.write('\n')
      note.write(str(rate))
      note.write('\n')
      note.write('runtime: '+str(end-start))
      note.write('\n\n')
note.close()
  



# fig, axes = plt.subplots(3,1, figsize=(10, 6), sharex=True, dpi=120)
# df_orig['s'].plot(ax=axes[0],  title='before')
# df_orig['s2'].plot(ax=axes[1], title='affter Smoothed 2%')
# df_orig['s5'].plot(ax=axes[2], title='affter Smoothed 5%')
# fig.suptitle('Slope before and after smoothing', y=0.95, fontsize=14)
# plt.show()
# plt.savefig('res/'+ file_name+"_slope.png")
