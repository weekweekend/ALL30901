import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import statsmodels.api as sm 
import time

# 读取视频
file_name = '14-10'
v_path="video/" + file_name + ".mp4"

# 读入一个mask并转化为二值（0为背景，1为前景）
mask = cv2.imread('label/'+ file_name +'.png',0)
mask[mask > 0 ] = 1
mask[mask !=1 ] = 0

cap = cv2.VideoCapture(v_path)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) # 总帧数

# 存储区域均值
ave = []
# 读取每帧图片
start =time.perf_counter() # 用于计算运行时间
for i in range(0,int(frame_count),1):
    _,img=cap.read()
    old =img
    # 读取到的每帧图像
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    sz = img.shape
    img = img*mask

    # 对前景区域求均值并放入数组
    num = img[img>0].mean()
    ave.append(num)
# 归一化
ave = np.array(ave)
ave = (ave-ave.min())/(ave.max()-ave.min())

end = time.perf_counter() 
print('逐帧取值：',end - start)


#　数据平滑
start =time.perf_counter()
lowess=sm.nonparametric.lowess
df_loess = lowess(ave, np.arange(len(ave)), frac=0.02)[:, 1]
end = time.perf_counter()
print('平滑',end-start)



# 计算一阶导数
start =time.perf_counter()
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

#存放所有异常分数
score = []
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
      ans += sum(val<0 for val in t1)/12/4*1*(-np.sum(t1[t1<0]))*(np.sum(t2))*1000000

    for j in range(i,i+6000,2400):
      t1 = slope1[j:j+2400]
      t2 = slope2[j:j+2400]
      tt = t1
      tt[tt>0]=0
      
      t2 = np.multiply(t2,tt)
      t2 = t2[t2<0]*-1
      ans += sum(val<0 for val in t1)/24/2*3*(-np.sum(t1[t1<0]))*(np.sum(t2))*1000000

    t1 = slope1[i:i+6000]
    t2 = slope2[i:i+6000]
    tt = t1
    tt[tt>0]=0
    t2 = np.multiply(t2,tt)
    t2 = t2[t2<0]*-1
    ans += sum(val<0 for val in t1)/60*6*(-np.sum(t1[t1<0]))*(np.sum(t2))*1000000

    score.append(ans)
    # 用于实时打印分数
    print(ans)
    
end = time.perf_counter()
print('score: ',score)
print('计算异常: ',end-start)