######################################################################
# 参考                                                               #
# https://blog.csdn.net/weixin_41036461/article/details/108406265    #
######################################################################



import cv2 as cv
import numpy as np
 
 
def ssim(img1, img2):
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
 
 
def calculate_ssim(img1, img2):
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
 
img1 = cv.imread("/home/zcq/documents/PRVS-Image-Inpainting-master/datasets/md.png", 0)
img2 = cv.imread("/home/zcq/documents/PRVS-Image-Inpainting-master/datasets/break_2.png", 0)
img3 = cv.imread("/home/zcq/documents/PRVS-Image-Inpainting-master/datasets/out.png", 0)
# _, img1 = cv.threshold(img1, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
# _, img2 = cv.threshold(img2, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
img1 = np.uint8(img1)
img2 = np.uint8(img2)
img3 = np.uint8(img3)
print(type(img1))
before = calculate_ssim(img1, img2)
after = calculate_ssim(img2, img3)
print("before:%.4f" %(before))
print("after:%.4f" %(after))