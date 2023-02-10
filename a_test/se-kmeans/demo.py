from sys import flags
import cv2
from matplotlib.image import imread
import numpy as np
from matplotlib import pyplot as plt
from numpy.core.defchararray import center 

import seaborn as sns
# sns.set()
# 对图像用kmeans聚类
 

img = cv2.imread('img/qx2.png')
# mask = cv2.imread('mask4/线圈.png')
# mask = mask/255
# img = img*mask
o = img.copy()
print('img.shape>>>>',img.shape)
h,w,c = img.shape
# 将一个像素点的rgb值作为一个单元处理，这一点很重要
data = img.reshape((-1,3))
print('data.shape>>>',data.shape)
# 转换数据类型
data = np.float32(data)
# 设置Kmeans参数
critera = (cv2.TermCriteria_EPS+cv2.TermCriteria_MAX_ITER,10,0.1)
# 初始中心的选择使眼色相差最大
flags = cv2.KMEANS_PP_CENTERS
# 对图片进行四分类
r,best,center = cv2.kmeans(data,4,None,criteria=critera,attempts=10,flags=flags)
print('r>>>',r)
print('best.shape>>>',best.shape)
print("center>>>",center)
center = np.uint8(center)
# 将不同分类的数据重新赋予另外一种颜色，实现分割图片

mean0 = data[best.ravel()==0].mean()
mean1 = data[best.ravel()==1].mean()
mean2 = data[best.ravel()==2].mean()
mean3 = data[best.ravel()==3].mean()

data[best.ravel()==0] = (mean0,mean0,mean0) 
data[best.ravel()==1] = (mean1,mean1,mean1)
data[best.ravel()==2] = (mean2,mean2,mean2)
data[best.ravel()==3] = (mean3,mean3,mean3) 



# 将结果转换为图片需要的格式
data = np.uint8(data)
oi = data.reshape((img.shape))

sns.heatmap(oi[:,:,1:2].reshape((h,w)), xticklabels =False,yticklabels =False,cmap='gray')
plt.savefig('res/heatmap.png')
# 显示图片
cv2.imwrite('res/2-2.png', oi)
 


