from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import statsmodels.api as sm 

# plt.rcParams.update({'xtick.bottom' : False, 'axes.titlepad':5})
# fw=[-0.0051,0.0052]
# file_name = '10-14'
# # Import
# df_orig = pd.read_csv('csv/'+file_name+'_out.csv', parse_dates=['time'], index_col='time')
# print(df_orig.val)
# aa



# # 1. Moving Average
# df_ma = df_orig.val.rolling(3, center=True, closed='both').mean()

 
# # 2. Loess Smoothing (2% and 5%)
# df_loess_2 = pd.DataFrame(lowess(df_orig.val, np.arange(len(df_orig.val)), frac=0.02)[:, 1], index=df_orig.index, columns=['val'])
# df_loess_5 = pd.DataFrame(lowess(df_orig.val, np.arange(len(df_orig.val)), frac=0.05)[:, 1], index=df_orig.index, columns=['val'])
 
 
# # Plot
# fig, axes = plt.subplots(3,1, figsize=(7, 7), sharex=True, dpi=120)
# df_orig['val'].plot(ax=axes[0], color='k', title='Original Series')
# df_loess_2['val'].plot(ax=axes[1], title='Loess Smoothed 2%')
# df_loess_5['val'].plot(ax=axes[2], title='Loess Smoothed 5%')
# # df_ma.plot(ax=axes[3], title='Moving Average (3)')
# fig.suptitle('Before and after smoothing', y=0.95, fontsize=14)
# plt.show()
# plt.savefig('res/'+file_name+"_smoothing.png")

# 归一化后，获取一阶二阶导数

# path = 'csvo/one/'
# for root,dirs,files in os.walk(path):
#   # print(files)
#   for file in files:
#     if file.endswith(".csv"):
#       file_name = file[:-4]
#       df_orig = pd.read_csv(path+file, parse_dates=['time'], index_col='time')
#       df_loess = pd.DataFrame(lowess(df_orig.val, np.arange(len(df_orig.val)), frac=0.03)[:, 1], index=df_orig.index, columns=['val'])
#       ## 计算斜率
#       num = df_loess.val.values
#       # 归一化
#       num = (num-num.min())/(num.max()-num.min())
#       tmp=[]
#       res = []
#       slope1 = []
#       for i in range(1,len(num),1):
#         slope1.append([i-1,num[i]-num[i-1]])
#         res.append(num[i]-num[i-1])
#         tmp.append([i-1,num[i]])

#       slope2 = []
#       for i in range(1,len(res),1):
#         slope2.append([i-1,res[i]-res[i-1]])

#       name = ['time', 'val']
#       s1 = pd.DataFrame(columns=name, data = slope1)
#       s1.to_csv('csv/'+ file_name+'_slope1.csv')
#       s2 = pd.DataFrame(columns=name, data = slope2)
#       s2.to_csv('csv/'+ file_name+'_slope2.csv')

#       stmp = pd.DataFrame(columns=name, data = tmp)


#       fig, axes = plt.subplots(3,1, figsize=(10, 10), sharex=True, dpi=120)
#       stmp['val'].plot(ax=axes[0],  title='The data after smoothed 3%')
#       s1['val'].plot(ax=axes[1],  title='The first derivative of data after smoothed 3%')
#       s2['val'].plot(ax=axes[2], title='The second derivative of data after smoothed 3%')
#       # plt.ylim(fw)
#       # fig.suptitle('Slope before and after smoothing', y=0.95, fontsize=14)
#       plt.show()
#       plt.savefig('pic/'+ file_name+"_slope.png")
      




# 短窗口平滑程度
      
path = 'csvo/one/'
for root,dirs,files in os.walk(path):
  # print(files)
  for file in files:
    if file.endswith(".csv"):
      file_name = file[:-4]
      df_orig = pd.read_csv(path+file, parse_dates=['time'], index_col='time')
      # df3 = pd.DataFrame(lowess(df_orig.val, np.arange(len(df_orig.val)), frac=0.03)[:, 1], index=df_orig.index, columns=['val'])
      # df5 = pd.DataFrame(lowess(df_orig.val, np.arange(len(df_orig.val)), frac=0.05)[:, 1], index=df_orig.index, columns=['val'])
      
      lowess=sm.nonparametric.lowess
      df0 = (df_orig.val.values)[12000:18000]
      minn = df0.min()
      maxx = df0.max()
      df3 = lowess(df0, np.arange(len(df0)), frac=0.03)[:, 1]  
      df5 = lowess(df0, np.arange(len(df0)), frac=0.08)[:, 1]  

      # df0 = (df0-df0.min())/(df0.max()-df0.min())
      # df3 = (df3-df3.min())/(df3.max()-df3.min())
      # df5 = (df5-df5.min())/(df5.max()-df5.min())

      o0=[]
      o1=[]
      o2=[]

      for i in range(len(df0)):
        o0.append([i,df0[i]])
        o1.append([i,df3[i]])
        o2.append([i,df5[i]])

      name = ['time', 'val']
      s0 = pd.DataFrame(columns=name, data = o0)
      s1 = pd.DataFrame(columns=name, data = o1)
      s2 = pd.DataFrame(columns=name, data = o2)

      fw=[244.6,246]
      fig, axes = plt.subplots(3,1, figsize=(10, 10), sharex=True, dpi=120)
      

      s0['val'].plot(ax=axes[0],  title='The data after smoothed 0%')
      s1['val'].plot(ax=axes[1],  title='The first derivative of data after smoothed 3%')
      s2['val'].plot(ax=axes[2], title='The second derivative of data after smoothed 8%')
      # fig.suptitle('Slope before and after smoothing', y=0.95, fontsize=14)
      # plt.ylim(fw)
      
      axes[0].set_ylim(ymin = fw[0], ymax = fw[1]) 
      axes[1].set_ylim(ymin = fw[0], ymax = fw[1]) 
      axes[2].set_ylim(ymin = fw[0], ymax = fw[1]) 
      plt.show()
      plt.savefig('tmp/smoothed.png')
      