import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import statsmodels.api as sm 

import time

# 读取视频
file_name = '10-14'
v_path="video/" + file_name + ".mp4"

# 读入一个mask并转化为二值（0为背景，1为前景）
mask = cv2.imread('label/'+ file_name +'.png',0)
mask[mask !=128 ] = 0
mask[mask ==128 ] = 1

cap = cv2.VideoCapture(v_path)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) # 总帧数

# 存储区域均值前3000和后3000
ave1 = []
ave2 = []
# 读取每帧图片
start =time.perf_counter() # 用于计算运行时间
count=0
#存放所有异常分数
score = []
# for i in range(0,int(frame_count),1):
for i in range(0,int(6100),1):
  if count==0:
    start = time.perf_counter()
  _,img=cap.read()
  old =img
  # 读取到的每帧图像
  img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

  sz = img.shape
  img = img*mask

  # 对前景区域求均值并放入数组
  num = img[img>0].mean()
  if count<3001:
    ave2.append(num)
  count+=1
  if count==3001:
    if len(ave1)==0:
      ave1=ave2
      ave2=[]
      count=0
      continue
    else:
      # ave 为待求的曲线
      ave = np.array(ave1+ave2)
      

      ave1=ave2
      ave2=[]
      count=0
      # 归一化
      # ave = (ave-ave.min())/(all_max-ave.min())
      #　数据平滑
      lowess=sm.nonparametric.lowess
      df_loess = lowess(ave, np.arange(len(ave)), frac=0.03)[:, 1]
      # 计算一阶导数
      slope1 = []
      for i in range(1,len(df_loess),1):
        slope1.append((df_loess[i]-df_loess[i-1])/1)
      slope1 = np.array(slope1)

      # 计算二阶导数
      slope2 = []
      for i in range(1,len(slope1),1):
        slope2.append(slope1[i]-slope1[i-1])

      slope1 = slope1[:-1]
      slope2 = np.array(slope2)
      print(len(slope1),len(slope2))


      for i in range(0,len(slope1),3000):
        if(i+6000 <= len(slope1)):
          ans = 0
          for j in range(i,i+6000,1200):
            t1 = slope1[j:j+1200]
            t2 = slope2[j:j+1200]
            tt = t1
            tt[tt>0]=0
            t2 = np.multiply(t2,tt)
            t2 = t2[t2<0]*-1
            ans += sum(val<0 for val in t1)/12/4*1*(-np.sum(t1[t1<0]))*(np.sum(t2))*100

          for j in range(i,i+6000,2400):
            t1 = slope1[j:j+2400]
            t2 = slope2[j:j+2400]
            tt = t1
            tt[tt>0]=0
            
            t2 = np.multiply(t2,tt)
            t2 = t2[t2<0]*-1
            ans += sum(val<0 for val in t1)/24/2*3*(-np.sum(t1[t1<0]))*(np.sum(t2))*100

          t1 = slope1[i:i+6000]
          t2 = slope2[i:i+6000]
          tt = t1
          tt[tt>0]=0
          t2 = np.multiply(t2,tt)
          t2 = t2[t2<0]*-1
          ans += sum(val<0 for val in t1)/60*6*(-np.sum(t1[t1<0]))*(np.sum(t2))*100
          end = time.perf_counter()
          score.append(ans)
          # 用于实时打印分数
          print(ans)
          print('runtime:', end-start)

print('all_score: ',score)
          