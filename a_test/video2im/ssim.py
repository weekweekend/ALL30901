from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import pandas as pd
from fitter import Fitter # 拟合分布
import time
import scipy.stats
import statsmodels.api as sm 

tp = 'shell'
file_name = ['0-6','10-14','14-18','18-14','14-10','14-6-14','14-0']
file_name = ['zl1','zl2','zr1','fl1','fl2','fr1']
# 
# file_name = ['0-6','10-14','14-18','18-14','14-10','14-6-14','14-0','zl1','zl2','zr1','fl1','fl2','fr1']
file_name = ['10-14']


for f in range(0,len(file_name),1):
    v_path="video/"+ file_name[f]  +".mp4"
    mask = cv2.imread('label/'+file_name[f] +'.png',0)
    solid = cv2.imread('video/'+file_name[f] +'.png',0)
 
    if tp == 'shell':
        lb = 113
    else :
        lb=128
    mask[mask!=lb ]=0
    mask[mask== lb ]=1

    top = 9999
    left=9999
    bottom = 0
    right=0

    for row in range(mask.shape[0]):
      for col in range(mask.shape[1]):
        if mask[row,col]!=0:
          left = min(col,left)
          right = max(col,right)
          bottom = max(row,bottom)
          top =  min(row,top)
    
    cap=cv2.VideoCapture(v_path)
    frame_count=cap.get(cv2.CAP_PROP_FRAME_COUNT)

    score=[]
    s2=[]

    for i in range(0,5000,5):
        print(i)
        _,img=cap.read()
        old =img
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # if i < int(frame_count)//4:
        #     img = solid

        if i%3000==0:
          init = img
          init = (img*mask)[top:bottom, left:right]
          
          continue

        img = (img*mask)[top:bottom, left:right]
        cv2.imwrite('c1.png',init)
        cv2.imwrite('c2.png',img)
        if i>0:
          score.append(ssim(img,init,multichannel=True))
          s2.append([i,ssim(img,init,multichannel=True)])
        

    name = ['time', 'val']
    s = pd.DataFrame(columns=name, data = s2)
    s.to_csv('res/'+ file_name[f]+'_ssim.csv')
    # lowess=sm.nonparametric.lowess
    # df_loess = lowess(score, np.arange(len(score)), frac=0.02)[:, 1]


    fig = plt.figure(figsize= (22,8),dpi = 80)
    x = range(0,len(score),1)
    y = score
    plt.plot(x, y)
    plt.savefig('res/'+file_name[f] + '_' + tp +'_ssim.png')

    
