import numpy as np
import cv2
import os
from skimage import io

# 第一步 取原始图与mask叠加
label = 'fl2'
types = ['shell','out','disc','line','pipe','sensor']
# types = ['shell']


path = 'results/'+label+'-score/'

for root,dirs,files in os.walk(path):
  files = sorted(files)
  # print(files)
  for f in files:
    if f.endswith(".jpg"):
      im_path = os.path.join(path,f)
      im = cv2.imread(im_path,cv2.IMREAD_GRAYSCALE)
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
        mask = cv2.resize(mask,(768,576))
        if mask.max() ==0:
          continue
        im = im*mask
        file_path = label+'/{}/need/{}'.format(tp,f)
        io.imsave(file_path, im)

