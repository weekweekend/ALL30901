import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import pandas as pd
import os
import matplotlib.pyplot as plt


types = ['shell','out','disc','line','pipe','sensor']
file_name = ['0-6','10-14','14-18','18-14','14-10','14-6-14','14-0']
# file_name = ['zl1','zl2','zr1','fl1','fl2','fr1']
# 
file_name = ['0-6','10-14','14-18','18-14','14-10','14-6-14','14-0','zl1','zl2','zr1','fl1','fl2','fr1']
# file_name = ['10-14']


# 遍历所有视频文件
for fname in file_name:
    v_path="video/"+ fname  +".mp4"
    solid = cv2.imread('video/'+fname +'.png',0)
    # 遍历所有部位
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
        mask = cv2.imread('label/'+fname +'.png',0)
        mask[mask!= lb ]=0
        mask[mask== lb ]=1

        ave = []
        csv=[]

        cap=cv2.VideoCapture(v_path)
        frame_count=cap.get(cv2.CAP_PROP_FRAME_COUNT)

        s = 0
        temp=0
        for i in range(0,int(frame_count),1):
        # for i in range(int(frame_count)):
            print(i)
            _,img=cap.read()
            old =img
            img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            sz = img.shape

            # 与mask 点乘， 背景区域全为0
            img = img*mask
        

            # 对前景区域求均值并放入数组
            num = img[img>0].mean()
            csv.append([i,num])
            ave.append(num)      

        # 绘制温度变化图并保存

        name = ['time', 'val']
        test = pd.DataFrame(columns=name, data = csv)
        test.to_csv('csv/'+fname+'_'+tp+'.csv')

        fig = plt.figure(figsize= (22,8),dpi = 80)
        x = range(0,len(ave),1)
        y = ave
        plt.plot(x, y)
        plt.savefig('pic/'+fname+'_'+tp+".png")
        ave = []




