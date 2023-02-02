import numpy as np
import cv2
import os
from skimage import io

# 计算dotd指标
label = 'zl2'
types = ['shell','out','disc','line','pipe','sensor']
types = ['shell']
for tp in types:
  if tp == 'shell':
      lb = 113
  elif tp =='out':
      lb=128
  elif tp =='disc':
      lb=89
  elif tp =='line':
      lb=52
  elif tp =='pipe':
      lb=75
  elif tp =='sensor':
      lb=14
  # 拿到GT并求框和中心
  gt = cv2.imread('GT/small/'+ label +'.png',0) 
  gt[gt<=250]=0
  gt[gt>250]=1
  left=10000
  right=0
  up=10000
  down=0
  for x in range(gt.shape[0]):
    for y in range(gt.shape[1]):
      if gt[x][y]==1:
        left = min(left,x)
        right = max(right,x)
        up = min(up,y)
        down = max(down,y)
  c1 = np.array([(left + right)/2,(up + down)/2])



  path = label+'/'+tp+'/res/'
  for root,dirs,files in os.walk(path):
    files = sorted(files)
    # print(files)
    # iou = 交集 / 并集
    dotd=[]
    flag=0
    num=0
    for f in files:
      if f.endswith(".jpg"):
        num+=1
        im_path = os.path.join(path,f)
        im = cv2.imread(im_path,cv2.IMREAD_GRAYSCALE)
        im[im<=250]=0
        im[im>250]=1
        left=10000
        right=0
        up=10000
        down=0
        for x in range(im.shape[0]):
          for y in range(im.shape[1]):
            if im[x][y]==1:
              left = min(left,x)
              right = max(right,x)
              up = min(up,y)
              down = max(down,y)
        c2 = np.array([(left + right)/2,(up + down)/2])
         
        if label == 'fr1':
          if num<20:
            gt = cv2.imread('GT/small/'+ label +'.png',0) 
          elif num>=20 and num<28:
            gt = cv2.imread('GT/small/'+ label +'-2.png',0) 
          elif num>=28 and num<36:
            gt = cv2.imread('GT/small/'+ label +'-3.png',0) 
          elif num>=36 and num<44:
            gt = cv2.imread('GT/small/'+ label +'-4.png',0) 
          elif num>=44 and num<52:
            gt = cv2.imread('GT/small/'+ label +'-5.png',0) 
          else:
            gt = cv2.imread('GT/small/'+ label +'-6.png',0) 
          gt[gt<=250]=0
          gt[gt>250]=1
          left=10000
          right=0
          up=10000
          down=0
          for x in range(gt.shape[0]):
            for y in range(gt.shape[1]):
              if gt[x][y]==1:
                left = min(left,x)
                right = max(right,x)
                up = min(up,y)
                down = max(down,y)
          c1 = np.array([(left + right)/2,(up + down)/2])

        if c1[0] == 5000 or c2[0]==5000:
          dotd.append(1)
        else:
          dis = ((c1[0]-c2[0])*(c1[0]-c2[0]) + (c1[1]-c2[1])*(c1[1]-c2[1]))**0.5
          dotd.append(0.37 ** (dis/25))  

    print(dotd)
    dotd1 = np.array(dotd)
    dotd1 = dotd1[dotd1<1]
    print(sum(dotd1)/len(dotd1))
    print(len(dotd),len(dotd1))