from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time


plt.rcParams.update({'xtick.bottom' : False, 'axes.titlepad':5})

types = ['shell','out','disc','line','pipe','sensor']

file_name = ['0-6','10-14','14-18','18-14','14-10','14-6-14','14-0']
file_name = ['14-18']

# file_name = ['0-6','10-14','14-18','18-14','14-10','14-6-14','14-0','zl1','zl2','zr1','fl1','fl2','fr1']

note=open('score.txt',mode='a')

path = 'csv/'
for root,dirs,files in os.walk(path):
  files = sorted(files)
  for i in range(0,len(files),2):
    if files[i].endswith(".csv"):
      file_name = files[i][:-11]
      # Import
      note.write(file_name+'\n')
      df_first = pd.read_csv(path+files[i], parse_dates=['time'], index_col='time')
      s1 = df_first.val.values 
      s1 = s1[:-1]

      df_second = pd.read_csv(path+files[i+1], parse_dates=['time'], index_col='time')
      s2 = df_second.val.values
      res = []
      print(len(s1),len(s2))
      start =time.perf_counter()
      for i in range(0,len(s1),3000):
        if(i+6000 <= len(s1)):
          ans = 0
          for j in range(i,i+6000,1200):
            t1 = s1[j:j+1200]
            t2 = s2[j:j+1200]
            tt = t1
            tt[tt>0]=0
            t2 = np.multiply(t2,tt)
            t2 = t2[t2<0]*-1
            ans += sum(val<0 for val in t1)/12/4*1*(-np.sum(t1[t1<0]))*(np.sum(t2))*1000000
      
          for j in range(i,i+6000,2400):
            t1 = s1[j:j+2400]
            t2 = s2[j:j+2400]
            tt = t1
            tt[tt>0]=0
            
            t2 = np.multiply(t2,tt)
            t2 = t2[t2<0]*-1
            ans += sum(val<0 for val in t1)/24/2*3*(-np.sum(t1[t1<0]))*(np.sum(t2))*1000000

          t1 = s1[i:i+6000]
          t2 = s2[i:i+6000]
          tt = t1
          tt[tt>0]=0
          t2 = np.multiply(t2,tt)
          t2 = t2[t2<0]*-1
          ans += sum(val<0 for val in t1)/60*6*(-np.sum(t1[t1<0]))*(np.sum(t2))*1000000

          res.append(ans)
      end = time.perf_counter()
      note.write(str(res))
      note.write('\n')
      note.write('runtime: '+str(start-end))
      note.write('\n')
note.close()
  



# fig, axes = plt.subplots(3,1, figsize=(10, 6), sharex=True, dpi=120)
# df_orig['s'].plot(ax=axes[0],  title='before')
# df_orig['s2'].plot(ax=axes[1], title='affter Smoothed 2%')
# df_orig['s5'].plot(ax=axes[2], title='affter Smoothed 5%')
# fig.suptitle('Slope before and after smoothing', y=0.95, fontsize=14)
# plt.show()
# plt.savefig('res/'+ file_name+"_slope.png")
