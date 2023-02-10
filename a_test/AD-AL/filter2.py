# 筛选真实位置点
import numpy as np
import cv2
import os
from skimage import io
import matplotlib.pyplot as plt
from  PIL import Image
import math
import warnings
warnings.filterwarnings("ignore")

# 所有小块结合后二值化

# 全局直方图均衡化
def equalHist(img, z_max = 255): # z_max = L-1 = 255
    # 灰度图像矩阵的高、宽
    H, W = img.shape
    # S is the total of pixels
    S = H * W

    out = np.zeros(img.shape)
    sum_h = 0
    for i in range(256):
        ind = np.where(img == i)
        sum_h += len(img[ind])
        z_prime = z_max / S * sum_h
        out[ind] = z_prime

    out = out.astype(np.uint8)
    return out


def numIslands(grid):
    def dfs(grid,x,y):
      if not 0<=x<len(grid) or not 0<=y<len(grid[0]):return 
      if grid[x][y]!=255:return  #注意题目这里输入的是字符"0",“1”
      grid[x][y]=2
      dfs(grid,x-1,y)
      dfs(grid,x+1,y)
      dfs(grid,x,y-1)
      dfs(grid,x,y+1)
      return 1,x,y  #是岛屿,则返回1

    res=0
    points=[]
    for  x in range(len(grid)):
        for y in range(len(grid[0])):
            if grid[x][y]==255: 
                n,x,y = dfs(grid,x,y)
                res+=n  #累加岛屿的数量
                points.append([x,y])
    return res,points
 
note=open('res.txt',mode='a')
label='zl1'
types = ['shell','out','disc','line','pipe','sensor']
types = ['shell']


path1 = 'data/01_0001/'
note.write(path1)
note.write('\n')

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
  mask = cv2.imread('label/'+label+'.png',0) 
  mask[mask!= lb ]=0
  mask[mask== lb ]=1
  if mask.max()==0:
    continue
  path2 = label+'/'+tp+'/point/'


  i=0
  # 得到点坐标
  points=[]
  for root2,dirs2,files2 in os.walk(path2):
    files2 = sorted(files2)
    print(files2)
    for f2 in files2:
      if f2.endswith(".jpg"):
        im_path2 = os.path.join(path2,f2)
        im = cv2.imread(im_path2,cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im,(1024,768))
        im[im>200]=255
        im[im!=255]=0
        
        n,p = numIslands(im)  
        # 取对应原始图片
        im = cv2.imread(os.path.join(path1,f2),cv2.IMREAD_GRAYSCALE)
        im = equalHist(im)
        imm = im*mask
        mean_all = (imm[imm>0]).mean()
        # 每张图
        num_yc=0
        type_yc=0
        res = np.zeros([768,1024])
        if len(p)>0:
          # print(len(p))
          num_yc=0
          # 每一个待选点
          for [x,y] in p:
            # 每个尺度
            type_yc=0
            varmax=0
            # 每个尺度的方差
            var=[]
            for j in range(2,30,2):
              # 5*5  9*9  13*13  17*17 ...
              imm1 = imm[x-j:x+j,y-j:y+j]
              file_path = 'range/{}_{}{}_{}.jpg'.format(f2[:-4],x,y,str(j))
              io.imsave(file_path, imm1)
              imm2 = imm1[imm1>0]
              var.append(int(np.var(imm2)))
              print(var)
      
              varmax = max([varmax,int(np.var(imm2))])
            # note.write(f2[:-4]+str(x)+','+str(y)+var)
            # 方差归一化
              var1 = (var-np.min(var))/(np.max(var)-np.min(var))
              # 方差斜率
              var2=[]
              # 方差斜率绝对值
              var3=[]
              before=var1[0]
              # 方差斜率和
              varsum=0
              for ele in var1:
                var2.append(ele-before)
                var3.append(abs(ele-before))
                varsum+=ele-before
                before=ele
              var2 = var2[1:]
              var3 = var3[1:]
            # 附近有异常点
            if np.max(var)-np.min(var)>300 and varsum>0:
            # if varsum>0.4 and np.max(var)-np.min(var)>500  and var[0]>0 and var[0]<1000:
              # print(var)
              # print(var1)
              # print(var2)
              # print(varsum)
              # if f2=='104.jpg':
              #   print(var)
              # if f2=='109.jpg':
              #   aaa
              # r = var3.index(np.max(var3))*2+4
              # print(2*(idx+1))
              # aaa
              num_yc=1
              r = 12
              # 无mask叠加小块图
              im1  =  im[x-r:x+r,y-r:y+r]
              # 有mask叠加小块图
              imm1 = imm[x-r:x+r,y-r:y+r]
              imm2 = imm1[imm1>0]
              mask2 = imm1
              mask2[mask2>0]=1
              _,bina = cv2.threshold(im1, 10, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
              
              bina[bina>250]=255
              bina[bina!=255]=0
              type_yc=2
              # 如果是冷点
              if np.mean(imm2) < mean_all:
                bina[bina==255]=2
                bina[bina==0]=255
                bina[bina==2]=0
                type_yc=1
              bina = bina*mask2
              res[x-r:x+r,y-r:y+r] = bina
              k1 = np.ones((3,3), np.uint8)
              # k2 = np.ones((2,2), np.uint8)
              # #图像开闭运算
              res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, k1)
              # res = cv2.morphologyEx(res, cv2.MORPH_OPEN, k2)
        text = '帧: '+f2[:-4]+'  部位: '+tp + '  是否异常: '+str(num_yc)
        note.write(text)
        note.write('\n')
        file_path = '{}/{}/res/{}'.format(label,tp,f2)
        io.imsave(file_path, res)    
note.close()



# # 
# for root1,dirs1,files1 in os.walk(path1):
#   files1 = sorted(files1)
#   # print(files)
#   i=-1
#   num1=-1
#   for f1 in files1:
#     if f1.endswith(".jpg"):
#       num1+=1
#       im_path1 = os.path.join(path1,f1)
#       im = cv2.imread(im_path1,cv2.IMREAD_GRAYSCALE)
#       im = im*mask
#       if num1%5==0 and i<len(points)-1:
#         i+=1
#       if len(points[i])==0: continue
#       ave = [[i,f1]]
#       var = []
#       for [x,y] in points[i]:
#         for j in range(10,11):
#           # 5*5
#           # 9*9
#           # 13*13
#           # 17*17
#           im1 = im[x-j:x+j,y-j:y+j]
#           im2 = im1[im1>0]
#           _,bina = cv2.threshold(im1, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#           # file_path = 'tmp/range/{}_{}.jpg'.format(f1[:-4],str(j))
#           # io.imsave(file_path, im1)
#           bina[bina>250]=255
#           bina[bina!=255]=0
#           ## 热点还是冷点
#           if np.mean(im2) <200:
#             bina[bina==255]=2
#             bina[bina==0]=255
#             bina[bina==2]=0
#           res = np.zeros([768,1024])
#           res[x-j:x+j,y-j:y+j] = bina
#           file_path = 'tmp/shell-res/{}'.format(f1)
#           io.imsave(file_path, res)
#           # ave.append(int(np.mean(im2)))
#           # var.append(int(np.var(im2)))
#       # print(ave,var)
      
      
     