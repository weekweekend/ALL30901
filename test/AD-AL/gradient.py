import numpy as np
import cv2
import os
from skimage import io

import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFilter


im_path = '13.png'
im = cv2.imread(im_path,cv2.IMREAD_GRAYSCALE)
mask = cv2.imread('mask.png',0) 
mask = cv2.resize(mask,(1024,768))
mask[mask<250]=0
mask[mask>= 250 ]=1
# im = im*mask
im = 255-im
w = 30
h = 30
# x = 500
# y = 500
im=cv2.resize(im,(h,h))
# gx,gy = np.gradient(im,10,10)
# gx = np.zeros([x-2,y-2])
# print(gx.shape)

# for i in range(1,x-1):
#   for j in range(1,y-1):
#     gx[i-1,j-1] = (im[i+1][j] - im[i-1][j])/2



# gy = np.zeros([x-2,y-2])
# for j in range(1,y-1):
#   for i in range(1,x-1):
#     gy[i-1,j-1] = (im[i][j+1] - im[i][j-1])/2

# print(gx,gy)
x, y = np.mgrid[0:h:30j, 0:w:30j]

dy, dx = np.gradient(im)
print(dy.max(),dx.max())

skip = (slice(None, None, 2), slice(None, None, 2))

fig, ax = plt.subplots()

ax.quiver(x[skip], y[skip], dx[skip].T, dy[skip].T)

ax.set(aspect=1, title='Quiver Plot')
plt.show()
plt.savefig('g13.png')