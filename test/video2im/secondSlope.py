from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.rcParams.update({'xtick.bottom' : False, 'axes.titlepad':5})

fw=[-0.0051,0.0052]
tp = 'out'
file_name = ['0-6','10-14','14-18','18-14','14-10','14-6-14','14-0']
# Import
for f in range(0,len(file_name),1):
  df_orig = pd.read_csv('csv/'+file_name[f] +'_'+ tp +'_slope.csv', parse_dates=['time'], index_col='time')

  ## 计算二阶
  num = df_orig.s2.values
  res = []

  for i in range(0,len(num),1):
    res.append(num[i])
    
  slope = []

  for i in range(1,len(res),1):
    slope.append([i-1,(res[i]-res[i-1])/1])



  name = ['time', 's']
  s = pd.DataFrame(columns=name, data = slope)
  s.to_csv('res/'+ file_name[f] +'_'+ tp +'_secondSlope.csv')

  df = pd.read_csv('csv/'+file_name[f] +'_'+ tp +'.csv', parse_dates=['time'], index_col='time')
  df2 = pd.DataFrame(lowess(df.val, np.arange(len(df.val)), frac=0.02)[:, 1], index=df.index, columns=['val'])

  fig, axes = plt.subplots(3,1, figsize=(7, 7), sharex=True, dpi=120)
  df2['val'].plot(ax=axes[0],  title='before')
  df_orig['s2'].plot(ax=axes[1], title='first derivative')
  # plt.ylim(fw)
  s['s'].plot(ax=axes[2], title='second derivative')

  fig.suptitle('The original data and the first and second derivatives', y=0.95, fontsize=14)
  plt.show()
  plt.savefig('res/'+ file_name[f] +'_'+ tp +'_secondSlope.png')