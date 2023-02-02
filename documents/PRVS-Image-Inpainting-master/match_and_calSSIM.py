##################################################################
# 配准代码参考
# https://docs.opencv.org/4.0.1/dc/dc3/tutorial_py_matcher.html
# SSIM代码参考
# https://blog.csdn.net/weixin_41036461/article/details/108406265
##################################################################

from skimage import io
import cv2 as cv
import torch.nn as nn
import numpy as np
from skimage.feature import canny

def ssim(img1,img2):

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1,img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def image_registration(img1,img2):

    # 使用 ORB 找到关键点和描述符
    orb = cv.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # Draw first 20 matches.
    # img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:20],None, flags=2)
    goodMatch = matches[:20]
    if len(goodMatch) > 4:
        ptsA = np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
    ransacReprojThreshold = 4
    H, status =cv.findHomography(ptsA,ptsB,cv.RANSAC,ransacReprojThreshold)
    #其中H为求得的单应性矩阵矩阵
    #status则返回一个列表来表征匹配成功的特征点。
    #ptsA,ptsB为关键点
    #cv2.RANSAC, ransacReprojThreshold这两个参数与RANSAC有关

    # imOut是配准以后的完整图
    imgOut = cv.warpPerspective(img2, H, (img1.shape[1],img1.shape[0]),flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP,borderValue=[255,255,255])
    return imgOut

def cal_mask_edge(img1,img2):
    # 1为破损图，2为模板图

    img_match = image_registration(img1,img2)
    # 二值化
    # img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    # img2_gray = cv.cvtColor(img_match, cv.COLOR_BGR2GRAY)
    _, binary1 = cv.threshold(img1, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    _, binary2 = cv.threshold(img_match, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # print(binary1.shape)
    outi = binary1-binary2
    # kernel1 = np.ones((3,3),np.uint8)
    kernel1 = [[1,1,1],[0,1,0],[1,1,1]]
    kernel1 = np.uint8(kernel1)
    kernel2 = np.ones((2,2),np.uint8)
    # print(type(kernel1))
    # print(type(kernel2))
    # print(kernel1)
    # print(kernel2)
    ## c.图像的腐蚀，默认迭代次数
    outi = cv.erode(outi,kernel1)
    outi = cv.dilate(outi,kernel2)
    oute = canny(outi, sigma=2).astype(np.float)
    return outi, oute




# img_path1 = "/home/zcq/documents/RFR/datasets/break_5.png"
# img_path2 = "/home/zcq/documents/RFR/datasets/md.png"

img1 = cv.imread("datasets/img_224.png",0)
img2 = cv.imread("datasets/md.png",0)
img1 = np.uint8(img1)
img2 = np.uint8(img2)

matchi = image_registration(img1,img2)
_, binary1 = cv.threshold(img1, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
_, binary2 = cv.threshold(matchi, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
io.imsave("datasets/mask.png", abs(binary1-binary2))
# io.imsave("datasets/matchmd.png", matchi)
# for i in range(0, 225, 32):
#     for j in range(0,225,32):
#         score = calculate_ssim(img1[i:i+31, j:j+31], matchi[i:i+31, j:j+31])
#         file_test_path1 = 'src/block_{:.4f}_1.png'.format(score)
#         file_test_path2 = 'src/block_{:.4f}_2.png'.format(score)
#         io.imsave(file_test_path1, img1[i:i+31, j:j+31])
#         io.imsave(file_test_path2, matchi[i:i+31, j:j+31])
#         print(score)

# bini, edgei = cal_mask_edge(img1,img2)
# io.imsave("datasets/outedge.png", edgei)
# io.imsave("datasets/outbini.png", bini)
# before = calculate_ssim(img1, img2)
# after = calculate_ssim(img1, matchi)
# print("before:%.4f" %(before))
# print("after:%.4f" %(after))
