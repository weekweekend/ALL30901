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
import statsmodels.api as sm 
lowess=sm.nonparametric.lowess
plt.rcParams.update({'xtick.bottom' : False, 'axes.titlepad':5})

# K-means++ 聚类代码

# 聚类中心数
c_num = 4
 
Note=open('K-means++.txt',mode='a')

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

data_tmp=[]
minlen = 99999999
# 聚类数据
path = 'csvo/30/'
for root,dirs,files in os.walk(path):
  # print(files)
  for file in files:
    if file.endswith(".csv"):
      df_orig = pd.read_csv(path+file, parse_dates=['time'], index_col='time')
      # df_loess_2 = pd.DataFrame(lowess(df_orig.val, np.arange(len(df_orig.val)), frac=0.02)[:, 1], index=df_orig.index, columns=['val'])
      tmp = df_orig.val.values
      data_tmp.append([file[:-4],tmp ])  # [name, data] 格式存储
      minlen = min(minlen,len(tmp))

# 多项式拟合 等长 数值归一化
data=[]
for i in range(len(data_tmp)):
  y = samelen(data_tmp[i][1])
  x = range(0,len(y),1)
  z1 = np.polyfit(x, y,4) #用4次多项式拟合，输出系数从高到0
  p1 = np.poly1d(z1) #使用次数合成的多项式
  y_pre = p1(range(0,len(y),1))
  y_pre = (y_pre-y_pre.min())/(y_pre.max()-y_pre.min())
  data.append([data_tmp[i][0],y_pre]) # [name, data] 格式存储
  print(data_tmp[i][0])

# 平滑 等长 数值归一化
data2=[]
for i in range(len(data_tmp)):
  y = samelen(data_tmp[i][1])
  y_pre = lowess(y, np.arange(len(y)), frac=0.8)[:, 1]  
  y_pre = (y_pre-y_pre.min())/(y_pre.max()-y_pre.min())
  data2.append([data_tmp[i][0],y_pre]) # [name, data] 格式存储
  print(data_tmp[i][0])

# 迭代次数
count = 1
 
centre = []

# 先随机取 1 个多项式拟合的曲线作为初始聚类中心
init = random.randint(0, len(data_tmp)-1)
print('初始中心：\n',init)
Note.write('初始中心: '+'\n' + str(init)) #\n 换行符
centre.append(data[init][1])
 
for i in range(1,c_num,1):
  # i为已初始化的中心个数
  # 计算与每个拟合曲线与所有中心最短的距离记录
  D = []
  for j in range(len(data)):
    dmin = np.inf 
    for k in range(i):
      dmin = min(dmin,dtw.distance_fast(data[j][1], centre[k]) )
    if dmin > 0 :
      D.append([j,dmin]) # 存储格式 [原始下标，值]
  D = np.array(D)
  D_sum = sum(D[:,1:2])[0]
  P = []
  for d in D:
    P.append([ d[0], d[1]/D_sum ])
  P_sum=[]
  pre = 0
  for p in P:
    P_sum.append([ int(p[0]), p[1]+pre])
    pre += p[1]
  # 使用轮盘法进行选择 ( 0-1的随机数在哪个区间就取哪个
  rn = random.random()
  pre = 0
  for j in range(len(P_sum)):
    if rn >= pre and rn < P_sum[j][1]:
      centre.append(data[P_sum[j][0]][1])
      print(P_sum[j][0])
      Note.write(str(P_sum[j][0])) #\n 换行符
      break
    else:
      pre = P_sum[j][1]

  # centre.append(y)
print('centre:',len(centre))
Note.write('centre:'+ str(len(centre)) +'\n') #\n 换行符

# while num < 5:
while True:
  print(count)
  Note.write(str(count) +'\n') #\n 换行符
  # 每个点到每个中心点的距离矩阵
  dis = []
  for i in range(len(data2)):
    dis.append([])
    # 每条曲线与各中心的距离
    for j in range(c_num):
      # 取到曲线
      tmp = data2[i][1]
      # dtw
      dis[i].append(dtw.distance_fast(tmp, centre[j])) 
      print('data: _ ',i,' _ ',data2[i][1])
      print('centre: _ ',j,' _ ', centre[j])
      print('dis:  _ ',dis[i])
      Note.write('data: '+ str(i) + ' _ ' + str(data2[i][1]) +'\n') #\n 换行符
      Note.write('centre: '+ str(j) + ' _ ' +   str(centre[j]) +'\n') #\n 换行符
      Note.write('dis: '+ str(dis[i]) + ' _ ' + '\n\n') #\n 换行符

    
  #初始化分类矩阵
  classify = []
  for i in range(c_num):
      classify.append([])

  #比较距离并分类
  for i in range(len(data2)):
      List = dis[i]
      dis[i] = np.array(dis[i])
      index = List.index(dis[i].min())
      classify[index].append([i,data2[i][0]])
    
  #构造新的中心点（全部等长
  new_centre = []
  for i in range(c_num):
    res = np.zeros((15000,))
    for j in range(len(classify[i])):
      res += data2[classify[i][j][0]][1]
    y = res / len(classify[i])
    new_centre.append(y)
     
  # 比较新的中心点和旧的中心点距离
  d0 = dtw.distance_fast(new_centre[0], centre[0])
  d1 = dtw.distance_fast(new_centre[1], centre[1])
  d2 = dtw.distance_fast(new_centre[2], centre[2])
  d3 = dtw.distance_fast(new_centre[3], centre[3])
  if (d0<1 and d1<1 and d2<1 and d3<1):
    break
  else:
    centre = new_centre
    count += 1
 
 
print('迭代次数为：',count)
print('聚类中心为：',centre)
print('分类情况为：',classify)
Note.write('聚类中心为:'+ str(centre) +'\n') #\n 换行符
Note.write('分类情况为:'+ str(classify) +'\n\n\n\n\n\n') #\n 换行符

fig = plt.figure(figsize= (22,8),dpi = 80)
x = range(0,len(centre[0]),1)
y = centre[0]
plt.plot(x, y)
plt.savefig('jlzx1.png')

fig = plt.figure(figsize= (22,8),dpi = 80)
x = range(0,len(centre[1]),1)
y = centre[1]
plt.plot(x, y)
plt.savefig('jlzx2.png')

fig = plt.figure(figsize= (22,8),dpi = 80)
x = range(0,len(centre[2]),1)
y = centre[2]
plt.plot(x, y)
plt.savefig('jlzx3.png')
Note.close()

fig = plt.figure(figsize= (22,8),dpi = 80)
x = range(0,len(centre[3]),1)
y = centre[3]
plt.plot(x, y)
plt.savefig('jlzx4.png')
Note.close()