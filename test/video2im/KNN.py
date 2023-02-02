from dtaidistance import dtw
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.signal import savgol_filter
import os
import statsmodels.api as sm 
lowess = sm.nonparametric.lowess

# 投票方法，设定好字典库，2400，4800，9600 截取，并规整为一致的长度 4800
# 比较时也做长度和数值的归一化
# 每个曲线以2400帧4800，9600，的尺度进行投票

Note=open('KNN.txt',mode='a')


def samelen(data,length=4800):
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

# 构造对比库  （前2为正常，后2为异常）
file_name = ['10-14_out','14-6-14_out']
Note.write('字典截取自: ' + str(file_name) +  ' \n\n') #\n 换行符

normal=[]
dic_normal=[]
anomalous=[]
dic_anomalous=[]
alllen = 9601
num=0
for i in range(len(file_name)):
  df_orig = pd.read_csv('csvo/30/'+file_name[i]+'.csv', parse_dates=['time'], index_col='time')
  # 全局平滑
  df_2 = pd.DataFrame(lowess(df_orig.val, np.arange(len(df_orig.val)), frac=0.2)[:, 1], index=df_orig.index, columns=['val'])
  data = df_2.val.values
  addd=0
  if i==2:
    addd=20000
  for j in range(0+addd, alllen+addd, 2400):
    if( j+2400 <= alllen+addd):
      ttmp = data[j:j+2400]
      ttmp = np.array(samelen(ttmp))
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
      ttmp = np.array(samelen(ttmp))
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
      ttmp = np.array(samelen(ttmp))
      ttmp = (ttmp -ttmp.min())/(ttmp.max()-ttmp.min())
      if i==0 or i==2:
        normal.append([num,ttmp])
        dic_normal.append(ttmp)
      else:
        dic_anomalous.append(ttmp)
        anomalous.append([num,ttmp])
      num+=1

# 绘制字典
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

 

# # # 分类
res=[]
path = 'csvo/30/'

for root,dirs,files in os.walk(path):
  # print(files)
  for file in files:
    if file.endswith(".csv"):
      df_orig = pd.read_csv(path+file, parse_dates=['time'], index_col='time')
      df_loess_2 = pd.DataFrame(lowess(df_orig.val, np.arange(len(df_orig.val)), frac=0.2)[:, 1], index=df_orig.index, columns=['val'])
      data = df_loess_2.val.values
      #　全局归一化
      # data = (data-data.min())/(data.max()-data.min())

      num_n=0
      num_a=0

    # 前k小的距离大部分是哪一个

    # 截取 2400做对比
      for j in range(0, len(data), 2400):
        if( j+2400 <= len(data)):
          dis=[]
          tmp = data[j:j+2400]
          # 局部归一化
          tmp = (tmp -tmp.min())/(tmp.max()-tmp.min())
          for k in range(len(dic_normal)):
            dis.append([dtw.distance_fast(tmp, dic_normal[k]),'normal'])
          for k in range(len(dic_anomalous)):
            dis.append([dtw.distance_fast(tmp, dic_anomalous[k]),'anomalous'])
          dis.sort()
          nn=0
          na=0
          # 距离哪个字典近就属于哪个
          for k in range(5):
            if dis[k][1] == 'normal': nn+=1
            else: na+=1
          if nn>na: num_n+=1
          else: num_a+=1
      # 截取4800
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
      # 截取9600
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
      # 整段
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
      # 曲线样本中属于正常或异常字典的的段数
      res.append([file[:-4],'normal:',num_n,'anomalous',num_a])
      Note.write(file[:-4] + ' 分类情况为: \n') #\n 换行符
      Note.write('normal: \n') #\n 换行符
      Note.write(str(num_n) + '\n') #\n 换行符
      Note.write('anomalous: \n') #\n 换行符
      Note.write(str(num_a) + '\n\n\n') #\n 换行符

 
print(res)