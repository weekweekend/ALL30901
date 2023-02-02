import numpy as np
import cv2
import os
from skimage import io
import matplotlib.pyplot as plt
from  PIL import Image
import math

#区域生长
def regionGrow(gray, seeds, thresh, p):  #thresh表示与领域的相似距离，小于该距离就合并
  seedMark = np.zeros(gray.shape)
  #八邻域
  if p == 8:
    connection = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
  #四邻域
  elif p == 4:
    connection = [(-1, 0), (0, 1), (1, 0), (0, -1)]

  #seeds内无元素时候生长停止
  while len(seeds) != 0:
    #栈顶元素出栈
    pt = seeds.pop(0)
    for i in range(p):
      tmpX = int(pt[0] + connection[i][0])
      tmpY = int(pt[1] + connection[i][1])

      #检测边界点
      if tmpX < 0 or tmpY < 0 or tmpX >= gray.shape[0] or tmpY >= gray.shape[1]:
        continue
      if math.fabs(int(gray[tmpX, tmpY]) - int(gray[pt[0],pt[1]])) < thresh and seedMark[tmpX, tmpY] == 0:
        seedMark[tmpX, tmpY] = 255
        seeds.append((tmpX, tmpY))
  return seedMark

tp='shell'
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
mask = cv2.imread('label/zl2.png',0) 
mask[mask!= lb ]=0
mask[mask== lb ]=1

path1 = '01_0001/'
path2 = 'tmp/shell-point1/'

i=0
for root2,dirs2,files2 in os.walk(path2):
  files2 = sorted(files2)
for root1,dirs1,files1 in os.walk(path1):
  files1 = sorted(files1)
  # print(files)
  num1=-1
  for f1 in files1:
    if f1.endswith(".jpg"):
      num1=num1+1
      im_path1 = os.path.join(path1,f1)
      im = cv2.imread(im_path1,cv2.IMREAD_GRAYSCALE)
      im = im*mask
      
      seeds=[]
      if num1%5==0:
        if i<len(files2):
          pst = cv2.imread(os.path.join(path2,files2[i]),cv2.IMREAD_GRAYSCALE)
          pst = cv2.resize(pst,(1024,768))
          pst[pst>200]=255
          pst[pst!=255]=0
          i+=1
        num2 = np.sum(pst==255)
        if num2>0:
          # res = im
          # im = im*pst
          # im2 = im[im>0]
          # ave = np.sum(im)/num2
          # print(ave)
          # res[res>ave+30]=0
          # # res[res<ave-30]=0
          # res[res!=0] = 255
          idx = np.where(mask == 1)
          seeds.append([idx[0][-1],idx[1][-1]])
          print(seeds)
          res= regionGrow(im, seeds, thresh=4, p=8)
          res[res>170]=255
          res[res!=255]=0
          res[res==255]=2
          res[res==0]=255
          res[res==2]=0
          res=res*mask
          tmp = res*pst
          if tmp.max()!=0:
            res=res
          else:
            res = np.zeros((768,1024))
        else:
          res = np.zeros((768,1024))
        file_path = 'tmp/shell-res/{}'.format(f1)
        io.imsave(file_path, res)
    
      # else:
      #   num2 = np.sum(pst==255)
      #   if num2>0:
      #     # res = im
      #     # im = im*pst
      #     # ave = np.sum(im)/num2
      #     # print(ave)
      #     # res[res>ave+30]=0
      #     # # res[res<ave-30]=0
      #     # res[res!=0] = 255
      #     idx = np.where(mask == 1)
      #     seeds.append([idx[0][-1],idx[1][-1]])
      #     print(seeds)
      #     res= regionGrow(im, seeds, thresh=3, p=8)
      #     res[res>170]=255
      #     res[res!=255]=0
      #     tmp = res*pst
      #     if tmp.max()==0:
      #       res[res==255]=2
      #       res[res==0]=255
      #       res[res==2]=0
      #       res=res*mask
      #   else:
      #     res = np.zeros((768,1024))
      #   file_path = 'tmp/shell-res/{}'.format(f1)
      #   io.imsave(file_path, res)

         