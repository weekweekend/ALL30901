import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import pandas as pd
from fitter import Fitter # 拟合分布
import time
import scipy.stats

pre=[]
inint=[]
tp = 'shell'
file_name = ['0-6','10-14','14-18','18-14','14-10','14-6-14','14-0']
# file_name = ['zl1','zl2','zr1','fl1','fl2','fr1']
# 
# file_name = ['0-6','10-14','14-18','18-14','14-10','14-6-14','14-0','zl1','zl2','zr1','fl1','fl2','fr1']


note=open('mask.txt',mode='a')
note.write(tp+'\n')

for f in range(0,len(file_name),1):
    v_path="video/"+ file_name[f]  +".mp4"
    mask = cv2.imread('label/'+file_name[f] +'.png',0)
    solid = cv2.imread('video/'+file_name[f] +'.png',0)
    note.write(file_name[f]+'\n')
    # mask1 = cv2.imread('label/'+file_name[f] +'_no.png',0)
    # np.savetxt(r'mask.txt', mask, fmt='%d', delimiter=',')
    if tp == 'shell':
        lb = 113
    else :
        lb=128
    mask[mask!= lb ]=0
    mask[mask== lb ]=1

    # mask1[mask1!= lb ]=0
    # mask1[mask1== lb ]=1

    err = []

    jz = []
    zs=[]
    zws=[]

    qj=[]
    nj=[]
    fc=[]
    bzc=[]

    kl=[]
    kl_g=[]
    kl_12=[]
    kl_g_12=[]

    klall=[]

    cap=cv2.VideoCapture(v_path)
    frame_count=cap.get(cv2.CAP_PROP_FRAME_COUNT)


    for i in range(0,int(frame_count),int(frame_count)//3):
        _,img=cap.read()
        old =img
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # print(i)
        sz = img.shape
        
        # img = cv2.blur(img, (10,10))

        # if i < int(frame_count)//3:
        #     img = solid


        img = img*mask

        im = img[img>0]
        

        # mean = np.mean(im)
        # std = np.std(im)
        # im2 = (im-mean)/std
        # im3 = im/255

    #     if i==0:
    #         init = im
    #         pre = im
    #     if len(pre)>0:
    #         l = len(pre) if len(pre)<len(im) else len(im)
    #         pre = pre[0:l]
    #         af = im[0:l]
    #         KL = scipy.stats.entropy(pre, af)*1000
    #         kl.append(KL)
    #         kl_12.append(KL)

    #         l = len(init) if len(init)<len(im) else len(im)
    #         init = init[0:l]
    #         af = im[0:l]

    #         KL = scipy.stats.entropy(init, af)*1000
    #         kl_g.append(KL)
    #         kl_g_12.append(KL)
        
    #     # if i%1200 == 0 and i>0  :
    #     #     init = im
    #     if i%200 ==0:
    #         # print(i,len(kl_12))
    #         klall.append(np.array(kl_12).mean() + np.array(kl_g_12).mean())
    #         kl_12=[]
    #         kl_g_12=[]
            
    #     pre=im

    # kl = kl[1:]
    # kl_g = kl_g[1:]
    # fig = plt.figure()
    # fig.suptitle(file_name[f]+'kl', y=0.95, fontsize=14)
    # x = range(0, len(kl), 1)
    # plt.plot(x,kl, label='kl')
    # plt.plot(x,kl_g, label='kl_g ')
    # plt.legend()
    # plt.show()
    # plt.savefig('res/'+file_name[f] + '_' + tp +'_kl.png')
    # print(len(klall))
    # fig2 = plt.figure()
    # fig2.suptitle(file_name[f]+'kl_score', y=0.95, fontsize=14)
    # xx = range(0, len(klall), 1)
    # plt.plot(xx,klall )
    # plt.show()
    # plt.savefig('res/'+file_name[f] + '_' + tp +'_kl_score.png')

    
    # note.write(file_name[f]+'kl \n')
    # note.write(str(kl))
    # note.write(file_name[f]+'kl_g \n')
    # note.write(str(kl_g))
    # note.write(file_name[f]+'klall \n')
    # note.write(str(klall))
    # note.write('\n')
# note.close()
        
        # print(i,np.min(im))
        cv2.imwrite('123.png',img)
        aaa
        fitter = Fitter(im,distributions=['gamma'],density=True)  # 创建Fitter类
        fitter.fit()  # 调用fit函数拟合分布
        # err.append(fitter.df_errors.sumsquare_error.values[0])
        # print(fitter.summary()) # 输出拟合结果
        # print(fitter.fitted_pdf['gamma'])
        note.write(str(fitter.fitted_pdf['gamma'])) #返回拟合分布的参数
        note.write('\n')
        fig, axes = plt.subplots(1,1,   sharex=True, dpi=120)
        fitter.summary()
        fig.suptitle('The original data and the and gamma fitting curve', y=0.95, fontsize=14)
        plt.savefig('res/'+file_name[f] + '_' + tp +'_fit_'+ str(int(i/(int(frame_count)//3))) +'.png')
    note.write('\n')   
note.close()   
        # zss = np.argmax(np.bincount(im))
        # im4 = np.sort(im.ravel())
        # zwss = im4[len(im4)//2]
        # jz.append(mean)
        # zs.append(zss)
        # zws.append(zwss)
        # fig = plt.figure()
        # fig.suptitle(file_name[f]+' mean mode and median', y=0.95, fontsize=14)
        # x = range(0, len(jz), 1)
        # plt.plot(x,jz, label='mean')
        # plt.plot(x,zs, label='mode')
        # plt.plot(x,zws, label='median')
        # plt.legend()
        # plt.show()
        # plt.savefig('res/'+file_name[f] + '_' + tp +'_jzzszws.png')


        # qj.append(im.max()-im.min())
        # im4 = np.sort(im.ravel())
        # im5 = im4[0:len(im4)//2]
        # im6 = im4[len(im4)//2:len(im4)]
        # nj.append(im6[len(im6)//2] - im5[len(im5)//2])
        # fc.append(np.var(im))
        # bzc.append(std)
        # fig = plt.figure()
        # fig.suptitle(file_name[f]+' range var std and quartile deviation', y=0.95, fontsize=14)
        # x = range(0, len(qj), 1)
        # plt.plot(x,qj, label='range')
        # plt.plot(x,nj, label='quartile deviation')
        # plt.plot(x,fc, label='var')
        # plt.plot(x,bzc, label='std')
        # plt.legend()
        # plt.show()
        # plt.savefig('res/'+file_name[f] + '_' + tp +'_qjnjfcbzc2.png')
        
 
        # fig = plt.figure()
        # fig.suptitle(file_name[f]+' range var std and quartile deviation', y=0.95, fontsize=14)
        # x = range(0, len(im.ravel()), 1)
        # plt.plot(x,im.ravel(), label='range')
        # plt.legend()
        # plt.show()
        # plt.savefig('res/'+file_name[f] + '_' + tp +'.png')


    
    # aaa

    
        


        # 对前景区域求灰度分布
        # fig = plt.figure(figsize= (22,8),dpi = 80)
        # plt.hist(im3.ravel())
        # plt.savefig('res/fb.png')
        # print(im.max(),im.min())
        # num = img[img>0].mean()

        # csv.append([i,num])
        # ave.append(num)
    # plt.savefig("tmp/"+file_name+str(int(time.time()))+".png")
    # print(np.array(err).max())
    # 绘制温度变化图并保存

    # name = ['time', 'val']
    # test = pd.DataFrame(columns=name, data = csv)
    # test.to_csv(file_name+'.csv')



    # fig = plt.figure(figsize= (22,8),dpi = 80)
    # x = range(0,len(ave),1)
    # y = ave
    # plt.plot(x, y)
    # plt.savefig(file_name+".png")
    # # ave = []

