import numpy as np
import cv2
import os
from skimage import io

# 第二步，取多帧叠加，定位点
label = 'fl2'
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
  mask = cv2.imread('label/'+ label +'.png',0) 
  mask[mask!= lb ]=0
  mask[mask== lb ]=1
  if mask.max()==0:
    continue
  mask = cv2.resize(mask,(768,576))
  
  path = label+'/'+tp+'/need/'
  for root,dirs,files in os.walk(path):
    files = sorted(files)
    # print(files)
    num=-1
    oldname=''
    for f in files:
      if f.endswith(".jpg"):
        num=num+1
        im_path = os.path.join(path,f)
        im = cv2.imread(im_path,cv2.IMREAD_GRAYSCALE)
        # im[im>200]=255
        # im[im!=255]=0
        # im=im/255
        if num%5==0:
          if num!=0:
            # img[img>1]=255
            # img[img!=255]=0
            # img = img*edge
            # 255的前90%，5张中3张存在 255/5*3*0.85 = 137.7
            img[img>138]=255
            img[img!=255]=0
            k1 = np.ones((3,3), np.uint8)
            # k2 = np.ones((2,2), np.uint8)
            # k2 = np.array((2,2), np.uint8)
            #图像开运算
            # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k1)
            # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, k2)
            file_path = '{}/{}/point/{}'.format(label,tp,oldname)
            io.imsave(file_path, img)
          img=im
        else:
          img+=im
        oldname = f
