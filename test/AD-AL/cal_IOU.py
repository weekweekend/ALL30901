import numpy as np
import cv2
import os
from skimage import io

# 第二步，取多帧叠加，定位点
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
  gt = cv2.imread('GT/small/'+ label +'.png',0) 
  gt[gt<=250]=0
  gt[gt>250]=1
  gt1 = cv2.imread('GT/black.png',0) 
  gt1[gt1<=250]=0
  gt1[gt1>250]=1

  path = label+'/'+tp+'/res/'
  for root,dirs,files in os.walk(path):
    files = sorted(files)
    # print(files)
    # iou = 交集 / 并集
    iou=[]
    flag=0
    num=0
    for f in files:
      if f.endswith(".jpg"):
        num+=1
        im_path = os.path.join(path,f)
        im = cv2.imread(im_path,cv2.IMREAD_GRAYSCALE)
        im[im<=250]=0
        im[im>250]=1
        # if f=='074.jpg':
        #   flag=1
        # if flag==0:
        #   gt=gt1
        # else:
        #    gt=gt
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

        jj = gt*im
        bj = gt+im
        bj[bj>0]=1
        jj_num = np.sum(jj)
        bj_num = np.sum(bj)
        if jj_num==0 :
          iou.append(1)
        else:
          iou.append(jj_num/bj_num)
    print(iou)
    iou2 = np.array(iou)
    iou2 = iou2[iou2<1]
    print(sum(iou2)/len(iou2))
    print(len(iou),len(iou2))