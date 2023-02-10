from dtaidistance import dtw
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.signal import savgol_filter
import os

# 投票方法，设定好字典库，2400，4800，9600
# 每个曲线以2400帧4800，9600，的尺度进行投票

# ## 构造对比库
file_name = ['10-14','14-6-14']
normal=[]
dic_normal=[]
anomalous=[]
dic_anomalous=[]
alllen = 9601
num=0
for i in range(len(file_name)):
  df_orig = pd.read_csv('csv/'+file_name[i]+'_out.csv', parse_dates=['time'], index_col='time')
  df_2 = pd.DataFrame(lowess(df_orig.val, np.arange(len(df_orig.val)), frac=0.2)[:, 1], index=df_orig.index, columns=['val'])
  # y=df_orig.val.values
  # x = range(0,len(y),1)
  # z1 = np.polyfit(x, y, 4) #用4次多项式拟合，输出系数从高到0
  # p1 = np.poly1d(z1) #使用次数合成多项式
  data = df_2.val.values
  # data = (data-data.min())/(data.max()-data.min())
  addd=0
  if i==2:
    addd=20000
  for j in range(0+addd, alllen+addd, 2400):
    if( j+2400 <= alllen+addd):
      ttmp = data[j:j+2400]
      ttmp = (ttmp -ttmp.min())/(ttmp.max()-ttmp.min())
      if i==0 or i==2:
        dic_normal.append(ttmp)
        normal.append([num,ttmp])
      else:
        dic_anomalous.append(ttmp)
        anomalous.append([num,ttmp])
      num+=1

  for j in range(0+addd, alllen+addd, 4800):
    if( j+ 4800 <= alllen+addd):
      ttmp = data[j:j+4800]
      ttmp = (ttmp -ttmp.min())/(ttmp.max()-ttmp.min())
      if i==0 or i==2:
        normal.append([num,ttmp])
        dic_normal.append(ttmp)
      else:
        dic_anomalous.append(ttmp)
        anomalous.append([num,ttmp])
      num+=1

  for j in range(0+addd, alllen+addd, 9600):
    if( j+ 9600 <= alllen+addd):
      ttmp = data[j:j+9600]
      ttmp = (ttmp -ttmp.min())/(ttmp.max()-ttmp.min())
      if i==0 or i==2:
        normal.append([num,ttmp])
        dic_normal.append(ttmp)
      else:
        dic_anomalous.append(ttmp)
        anomalous.append([num,ttmp])
      num+=1

for i in range(len(dic_normal)):
  fig = plt.figure(figsize= (16,8),dpi = 80)
  x = range(0,len(dic_normal[i]),1)
  y = dic_normal[i]
  plt.plot(x, y)
  plt.savefig('tmp/normal_'+str(i)+'.png')

for i in range(len(dic_anomalous)):
  fig = plt.figure(figsize= (16,8),dpi = 80)
  x = range(0,len(dic_anomalous[i]),1)
  y = dic_anomalous[i]
  plt.plot(x, y)
  plt.savefig('tmp/anomalous_'+str(i)+'.png')

# name = ['time', 'val']
# s = pd.DataFrame(columns=name, data = normal)
# s.to_csv('res/dic_out_normal.csv')

 
# name = ['time', 'val']
# s = pd.DataFrame(columns=name, data = anomalous)
# s.to_csv('res/dic_out_anomalous.csv')

# # # 分类
res=[]
path = 'csv/'

for root,dirs,files in os.walk(path):
  # print(files)
  for file in files:
    if file.endswith(".csv"):
      df_orig = pd.read_csv(path+file, parse_dates=['time'], index_col='time')
      df_loess_2 = pd.DataFrame(lowess(df_orig.val, np.arange(len(df_orig.val)), frac=0.2)[:, 1], index=df_orig.index, columns=['val'])
      data = df_loess_2.val.values
      # data = (data-data.min())/(data.max()-data.min())

      num_n=0
      num_a=0
    # 前k小的距离大部分是哪一个
      for j in range(0, len(data), 2400):
        if( j+2400 <= len(data)):
          dis=[]
          tmp = data[j:j+2400]
          tmp = (tmp -tmp.min())/(tmp.max()-tmp.min())
          for k in range(len(dic_normal)):
            dis.append([dtw.distance_fast(tmp, dic_normal[k]),'normal'])
          for k in range(len(dic_anomalous)):
            dis.append([dtw.distance_fast(tmp, dic_anomalous[k]),'anomalous'])
          dis.sort()
          nn=0
          na=0
          for k in range(5):
            if dis[k][1] == 'normal': nn+=1
            else: na+=1
          if nn>na: num_n+=1
          else: num_a+=1

      for j in range(0, len(data), 4800):
        if( j+ 4800 <= len(data)):
          dis=[]
          tmp = data[j:j+4800]
          tmp = (tmp -tmp.min())/(tmp.max()-tmp.min())
          for k in range(len(dic_normal)):
            dis.append([dtw.distance_fast(tmp, dic_normal[k]),'normal'])
          for k in range(len(dic_anomalous)):
            dis.append([dtw.distance_fast(tmp, dic_anomalous[k]),'anomalous'])
          dis.sort()
          nn=0
          na=0
          for k in range(5):
            if dis[k][1] == 'normal': nn+=1
            else: na+=1
          if nn>na: num_n+=1
          else: num_a+=1

      for j in range(0, len(data), 9600):
        if( j+ 9600 <= len(data)):
          dis=[]
          tmp = data[j:j+9600]
          tmp = (tmp -tmp.min())/(tmp.max()-tmp.min())
          for k in range(len(dic_normal)):
            dis.append([dtw.distance_fast(tmp, dic_normal[k]),'normal'])
          for k in range(len(dic_anomalous)):
            dis.append([dtw.distance_fast(tmp, dic_anomalous[k]),'anomalous'])
          dis.sort()
          nn=0
          na=0
          for k in range(5):
            if dis[k][1] == 'normal': nn+=1
            else: na+=1
          if nn>na: num_n+=1
          else: num_a+=1
      dis=[]
      tmp = data
      tmp = (tmp -tmp.min())/(tmp.max()-tmp.min())
      for k in range(len(dic_normal)):
        dis.append([dtw.distance_fast(tmp, dic_normal[k]),'normal'])
      for k in range(len(dic_anomalous)):
        dis.append([dtw.distance_fast(tmp, dic_anomalous[k]),'anomalous'])
      dis.sort()
      nn=0
      na=0
      for k in range(5):
        if dis[k][1] == 'normal': nn+=1
        else: na+=1
      if nn>na: num_n+=1
      else: num_a+=1

      res.append([file[:-4],'normal:',num_n,'anomalous',num_a])

    # # 最小距离是哪一个
    #   for j in range(0, len(data), 2400):
    #     if( j+2400 <= len(data)):
    #       minn = 9999999
    #       mina = 9999999
    #       tmp = data[j:j+2400]
    #       for k in range(len(dic_normal)):
    #         minn = min(minn,dtw.distance_fast(tmp, dic_normal[k]))
    #       for k in range(len(dic_anomalous)):
    #         mina = min(minn,dtw.distance_fast(tmp, dic_anomalous[k]))
    #       if minn>mina:
    #         num_a+=1
    #       else:
    #         num_n+=1

    #   for j in range(0, len(data), 4800):
    #     if( j+ 4800 <= len(data)):
    #       minn = 9999999
    #       mina = 9999999
    #       tmp = data[j:j+4800]
    #       for k in range(len(dic_normal)):
    #         minn = min(minn,dtw.distance_fast(tmp, dic_normal[k]))
    #       for k in range(len(dic_anomalous)):
    #         mina = min(minn,dtw.distance_fast(tmp, dic_anomalous[k]))
    #       if minn>mina:
    #         num_a+=1
    #       else:
    #         num_n+=1

    #   for j in range(0, len(data), 9600):
    #     if( j+ 9600 <= len(data)):
    #       minn = 9999999
    #       mina = 9999999
    #       tmp = data[j:j+9600]
    #       for k in range(len(dic_normal)):
    #         minn = min(minn,dtw.distance_fast(tmp, dic_normal[k]))
    #       for k in range(len(dic_anomalous)):
    #         mina = min(minn,dtw.distance_fast(tmp, dic_anomalous[k]))
    #       if minn>mina:
    #         num_a+=1
    #       else:
    #         num_n+=1
    #   minn = 9999999
    #   mina = 9999999
    #   tmp = data
    #   for k in range(len(dic_normal)):
    #     minn = min(minn,dtw.distance_fast(tmp, dic_normal[k]))
    #   for k in range(len(dic_anomalous)):
    #     mina = min(minn,dtw.distance_fast(tmp, dic_anomalous[k]))
    #   if minn>mina:
    #     num_a+=1
    #   else:
    #     num_n+=1
    #   res.append([file[:-4],'normal:',num_n,'anomalous',num_a])

# name = ['time', 'val']

print(res)

# # dic_normal = pd.read_csv('csv/dic_out_normal.csv', parse_dates=['time'], index_col='time').val.values
# # dic_anomalous = pd.read_csv('csv/dic_out_anomalous.csv', parse_dates=['time'], index_col='time').val.values

# f_name = '18-14'
# data = pd.read_csv('csv/'+f_name+'_shell.csv', parse_dates=['time'], index_col='time').val.values

# min_normal = 999999999999
# min_anomalous = 999999999999
# sum_normal = 0
# sum_anomalous=0
# for i in range(0, len(data), 1200):
#   if( i+1200 <= len(data)):
#     tmp = data[i:i+1200]
#     for j in range(len(dic_normal)):
#       dist = dtw.distance_fast(tmp, dic_normal[j])
#       sum_normal += dist*0.1
#       if dist < min_normal:
#         min_normal = dist
#     for j in range(len(dic_anomalous)):
#       dist = dtw.distance_fast(tmp, dic_anomalous[j])
#       sum_anomalous += dist*0.1
#       if dist < min_anomalous:
#         min_anomalous = dist
  

# for i in range(0, len(data), 3600):
#   if( i+ 3600 <= len(data)):
#     tmp = data[i:i+3600]
#     for j in range(len(dic_normal)):
#       dist = dtw.distance_fast(tmp, dic_normal[j])
#       sum_normal += dist*0.3
#       if dist < min_normal:
#         min_normal = dist
#     for j in range(len(dic_anomalous)):
#       dist = dtw.distance_fast(tmp, dic_anomalous[j])
#       sum_anomalous += dist*0.3
#       if dist < min_anomalous:
#         min_anomalous = dist

# for i in range(0, len(data), 6000):
#   if( i+ 6000 <= len(data)):
#     tmp = data[i:i+6000]
#     for j in range(len(dic_normal)):
#       dist = dtw.distance_fast(tmp, dic_normal[j])
#       sum_normal += dist*0.6
#       if dist < min_normal:
#         min_normal = dist
#     for j in range(len(dic_anomalous)):
#       dist = dtw.distance_fast(tmp, dic_anomalous[j])
#       sum_anomalous += dist*0.6
#       if dist < min_anomalous:
#         min_anomalous = dist

# print(min_normal,min_anomalous)
# print(sum_normal,sum_anomalous)




# np.random.seed(0)
# a = np.random.normal(3, 2.5, size=(2, 1000))

# file_name1 = '10-14'
# file_name2 = '14-0'
# # Import
# df1 = pd.read_csv('csv/'+file_name1+'_out.csv', parse_dates=['time'], index_col='time')
# print(df1)
# aa
# df2 = pd.read_csv('csv/'+file_name2+'_out.csv', parse_dates=['time'], index_col='time')

# df1_2 = pd.DataFrame(lowess(df1.val, np.arange(len(df1.val)), frac=0.02)[:, 1], index=df1.index, columns=['val'])
# df2_2 = pd.DataFrame(lowess(df2.val, np.arange(len(df2.val)), frac=0.02)[:, 1], index=df2.index, columns=['val'])

# df1_5 = pd.DataFrame(lowess(df1.val, np.arange(len(df1.val)), frac=0.05)[:, 1], index=df1.index, columns=['val'])
# df2_5 = pd.DataFrame(lowess(df2.val, np.arange(len(df2.val)), frac=0.05)[:, 1], index=df2.index, columns=['val'])



# # 使用dtaidistance计算dtw距离
# dist = dtw.distance_fast(df1.val.values, df2.val.values)
# dist2 = dtw.distance_fast(df1_2.val.values, df2_2.val.values)
# dist5 = dtw.distance_fast(df1_5.val.values, df2_5.val.values)
# print(dist, dist2, dist5)