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
plt.rcParams.update({'xtick.bottom' : False, 'axes.titlepad':5})

# 聚类代码

Note=open('cluster.txt',mode='a')

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
path = 'csvo_14/'
for root,dirs,files in os.walk(path):
  # print(files)
  for file in files:
    if file.endswith(".csv"):
      df_orig = pd.read_csv(path+file, parse_dates=['time'], index_col='time')
      # df_loess_2 = pd.DataFrame(lowess(df_orig.val, np.arange(len(df_orig.val)), frac=0.02)[:, 1], index=df_orig.index, columns=['val'])
      tmp = df_orig.val.values
      # tmp = (tmp-tmp.min())/(tmp.max()-tmp.min())
      data_tmp.append([file[:-4],tmp ])  # [name, data] 格式存储
      minlen = min(minlen,len(tmp))
# name = ['time', 'val']
# test = pd.DataFrame(columns=name, data = csv)
# test.to_csv('csv/'+file_name[f]+'_'+tp+'.csv')

# 不拟合不等长
data = data_tmp

# 拟合 不等长 数值归一化
# data=[]
# for i in range(len(data_tmp)):
#   y=data_tmp[i][1]
#   x = range(0,len(y),1)
#   z1 = np.polyfit(x, y,4) #用4次多项式拟合，输出系数从高到0
#   p1 = np.poly1d(z1) # 使用次数合成的多项式
#   y_pre = p1(range(0,len(y),len(y)//minlen)) # 长度不同间隔取样，但不截取到等长
#   # print(len(y),minlen,len(y_pre))
#   y_pre = (y_pre-y_pre.min())/(y_pre.max()-y_pre.min())
#   data.append([data_tmp[i][0],y_pre]) # [name, data] 格式存储
#   print(len(y_pre))

# 拟合 等长 数值归一化
data=[]
for i in range(len(data_tmp)):
  y = samelen(data_tmp[i][1])
  x = range(0,len(y),1)
  z1 = np.polyfit(x, y,4) #用4次多项式拟合，输出系数从高到0
  p1 = np.poly1d(z1) #使用次数合成的多项式
  y_pre = p1(range(0,len(y),1))
  y_pre = (y_pre-y_pre.min())/(y_pre.max()-y_pre.min())
  data.append([data_tmp[i][0],y_pre]) # [name, data] 格式存储

# 聚类数
k=3
num = 1
 
centre = []

# 随机取k个曲线作为初始聚类中心
init=[random.randint(0, len(data_tmp)-1),random.randint(0, len(data_tmp)-1),random.randint(0, len(data_tmp)-1)]

# 指定k个曲线, 拟合后作为聚类中心
# init=[0,10,11]
for i in range(k):
  y = data[init[i]][1]
  # x = range(0,len(y),1)
  # z1 = np.polyfit(x, y, 4) #用4次多项式拟合，输出系数从高到0
  # p1 = np.poly1d(z1) #使用次数合成多项式
  # y_pre = p1(x)
  # y_pre = (y_pre-y_pre.min())/(y_pre.max()-y_pre.min())
  centre.append(y)
print('centre:',len(centre))
Note.write('centre:'+ str(len(centre)) +'\n') #\n 换行符

# while num < 5:
while True:
  print(num)
  Note.write(str(num) +'\n') #\n 换行符
  # 每个点到每个中心点的距离矩阵
  dis = []
  for i in range(len(data)):
    dis.append([])
    # 每条曲线与各中心的距离
    for j in range(k):
      tmp=data[i][1]
      # tmp=(tmp-tmp.min())/(tmp.max()-tmp.min())
      # dtw
      dis[i].append(dtw.distance_fast(tmp, centre[j])) 
      # # os / mhd
      # dis[i].append(np.linalg.norm(data[i][1]-centre[j],ord=1))
      print('data: _ ',i,' _ ',data[i][1])
      print('centre: _ ',j,' _ ', centre[j])
      print('dis:  _ ',dis[i])
      Note.write('data: '+ str(i) + ' _ ' + str(data[i][1]) +'\n') #\n 换行符
      Note.write('centre: '+ str(j) + ' _ ' +   str(centre[j]) +'\n') #\n 换行符
      Note.write('dis: '+ str(dis[i]) + ' _ ' + '\n\n') #\n 换行符

    
  #初始化分类矩阵
  classify = []
  for i in range(k):
      classify.append([])

  #比较距离并分类
  for i in range(len(data)):
      List = dis[i]
      dis[i] = np.array(dis[i])
      index = List.index(dis[i].min())
      classify[index].append([i,data[i][0]])
    
  #构造新的中心点（全部等长
  new_centre = []
  for i in range(k):
    res = np.zeros((15000,))
    for j in range(len(classify[i])):
      res += data[classify[i][j][0]][1]
    y = res / len(classify[i])
    new_centre.append(y)
    
  # #构造新的中心点(拟合但不等长)
  # new_centre = []
  # for i in range(k):
  #   minlen2 = 9999999
  #   # 第 j 类 中最短的长度为minlen2
  #   for j in range(len(classify[i])):
  #     tmp = data[classify[i][j][0]][1]
  #     minlen2 = min(minlen2,len(tmp))
  #     # 变等长后累加求均值，作为新中心
  #   res = np.zeros((minlen2,))
  #   for j in range(len(classify[i])):
  #     y = data[classify[i][j][0]][1]
  #     y_pre = y[0:len(y):len(y)//minlen2]
  #     y_pre = y_pre[0:minlen2]
  #     res += y_pre
  #   y = res/len(classify[i])
  #   new_centre.append(y)

  # #构造新的中心点(不拟合中心等长)
  # new_centre = []
  # for i in range(k):
  #   minlen2 = 9999999
  #   for j in range(len(classify[i])):
  #     tmp=data[classify[i][j][0]][1]
  #     minlen2=min(minlen2,len(tmp))
  #   res=np.zeros((minlen2,))
  #   for j in range(len(classify[i])):
  #     y=data[classify[i][j][0]][1]
  #     x = range(0,len(y),1)
  #     z1 = np.polyfit(x, y,4) #用4次多项式拟合，输出系数从高到0
  #     p1 = np.poly1d(z1) #使用次数合成多项式
  #     y_pre = p1(range(0,len(y),len(y)//minlen2))
  #     y_pre = y_pre[0:minlen2]
  #     y_pre = (y_pre-y_pre.min())/(y_pre.max()-y_pre.min())
  #     res += y_pre
  #   y = res/len(classify[i])
  #   new_centre.append(y)
     
  # 比较新的中心点和旧的中心点距离
  d0 = dtw.distance_fast(new_centre[0], centre[0])
  d1 = dtw.distance_fast(new_centre[1], centre[1])
  d2 = dtw.distance_fast(new_centre[2], centre[2])
  if (d0<1 and d1<1 and d2<1):
    break
  else:
    centre = new_centre
    num = num + 1
 
 
print('迭代次数为：',num)
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