import os
import glob
import scipy
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.utils import save_image
from imageio import imread
from skimage.feature import canny
from skimage.color import rgb2gray
import cv2
from skimage import io




class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_path, mask_path, mask_mode, target_size, augment=True, training=True, mask_reverse = False):
        super(Dataset, self).__init__()
        self.augment = augment
        self.training = training
        self.data = self.load_list(image_path)
        self.mask_data = self.load_list(mask_path)

        self.target_size = target_size
        self.mask_type = mask_mode
        self.mask_reverse = mask_reverse

        self.sigma = 2
        self.nms = 1
        # self.mask_reverse = mask_reverse

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):
        
        # load image
        img = imread(self.data[index])
        if self.training:
            img = self.resize(img)
        else:
            img = self.resize(img, True, True, True)
        img_gray = rgb2gray(img)
        # loasd mask
        mask = self.load_mask(img, index)
        # load mask_edge
        mask_edge = self.load_mask_edge(img,mask)
        # load edge
        edge = self.load_edge(img_gray, mask)

        #变大变小
        if self.training:
            if self.augment and np.random.binomial(1, 0.5) > 0:
                img = img[:, ::-1, ...]
                img_gray = img_gray[:, ::-1, ...]
                edge = edge[:, ::-1, ...]
                mask = mask[:, ::-1, ...]
        return self.to_tensor(img), self.to_tensor(img_gray), self.to_tensor(mask_edge), self.to_tensor(edge), self.to_tensor(mask)

    def load_edge(self, img, mask):
        sigma = self.sigma
        edge_canny = canny(img, sigma=sigma).astype(np.float)

        return edge_canny

    def load_mask_edge(self, img, mask):
        m = np.stack((mask,)*3, axis=-1)
        masked_img_gray = rgb2gray(img*m)
        img2 = cv2.imread("datasets/md.png",0) # 模板即img2
        img2 = np.array(Image.fromarray(img2).resize(size=(self.target_size, self.target_size)))
        img1 = np.array(masked_img_gray)
        # print(img1.max())
        # save_image(img1, "datasets/img1.png")
        img1 = np.uint8(255-img1*255)
        img2 = np.uint8(img2)
        # io.imsave("datasets/img1.png", img1)
        # io.imsave("datasets/img2.png", img2)

        matchi = self.image_registration(img1,img2)
        # io.imsave("datasets/matchmd.png", matchi)
        bini, edgei = self.cal_mask_edge(img1,img2)
        # io.imsave("datasets/outedge.png", edgei)
        # io.imsave("datasets/outbini.png", bini)
        before = self.calculate_ssim(img1, img2)
        after = self.calculate_ssim(img1, matchi)

        return edgei

    def load_mask(self, img, index):
        imgh, imgw = img.shape[0:2]
        
        #external mask, random order
        if self.mask_type == 0:
            mask_index = random.randint(0, len(self.mask_data) - 1)
            mask = imread(self.mask_data[mask_index])
            mask = self.resize(mask, False)
            mask = (mask > 0).astype(np.uint8)       # threshold due to interpolation
            if self.mask_reverse:
                return (1 - mask) * 255
            else:
                return mask * 255
        #generate random mask
        if self.mask_type == 1:
            mask = 1 - generate_stroke_mask([self.target_size, self.target_size])
            # return (mask * 255).astype(np.uint8)
            mask = (mask>0).astype(np.uint8)* 255
            mask = self.resize(mask,False)
            return mask
        
        #external mask, fixed order
        if self.mask_type == 2:
            mask_index = index
            mask = imread(self.mask_data[mask_index])
            mask = self.resize(mask, False)
            mask = (mask > 0).astype(np.uint8)       # threshold due to interpolation
            if self.mask_reverse:
                return (1 - mask) * 255
            else:
                return mask * 255

    def resize(self, img, aspect_ratio_kept = True, fixed_size = False, centerCrop=False):
        
        if aspect_ratio_kept:
            imgh, imgw = img.shape[0:2]
            side = np.minimum(imgh, imgw)
            if fixed_size:
                if centerCrop:
                # center crop
                    j = (imgh - side) // 2
                    i = (imgw - side) // 2
                    img = img[j:j + side, i:i + side, ...]
                else:
                    j = (imgh - side)
                    i = (imgw - side)
                    h_start = 0
                    w_start = 0
                    if j != 0:
                        h_start = random.randrange(0, j)
                    if i != 0:
                        w_start = random.randrange(0, i)
                    img = img[h_start:h_start + side, w_start:w_start + side, ...]
            else:
                if side <= self.target_size:
                    j = (imgh - side)
                    i = (imgw - side)
                    h_start = 0
                    w_start = 0
                    if j != 0:
                        h_start = random.randrange(0, j)
                    if i != 0:
                        w_start = random.randrange(0, i)
                    img = img[h_start:h_start + side, w_start:w_start + side, ...]
                else:
                    side = random.randrange(self.target_size, side)
                    j = (imgh - side)
                    i = (imgw - side)
                    h_start = random.randrange(0, j)
                    w_start = random.randrange(0, i)
                    img = img[h_start:h_start + side, w_start:w_start + side, ...]
        img = np.array(Image.fromarray(img).resize(size=(self.target_size, self.target_size)))

        return img
    
    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t
    
    def load_list(self, flist):

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if flist[-3:] == "txt":
                line = open(flist,"r")
                lines = line.readlines()
                file_names = []
                for line in lines:
                    file_names.append("../../Dataset/Places2/train/data_256"+line.split(" ")[0])
                return file_names
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]

        return []

    def ssim(self, img1, img2):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())
        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()
    
    def calculate_ssim(self, img1, img2):
        '''calculate SSIM
        the same outputs as MATLAB's
        img1, img2: [0, 255]
        '''
        # img1 = self.img1
        # img2 = self.img2
        if not img1.shape == img2.shape:
            raise ValueError('Input images must have the same dimensions.')
        if img1.ndim == 2:
            return self.ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(self.ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return self.ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError('Wrong input image dimensions.')

    def image_registration(self, img1, img2):
        # img1 = self.img1
        # img2 = self.img2
        # 使用 ORB 找到关键点和描述符
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img1,None)
        kp2, des2 = orb.detectAndCompute(img2,None)
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
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
        H, status =cv2.findHomography(ptsA,ptsB,cv2.RANSAC,ransacReprojThreshold)
        #其中H为求得的单应性矩阵矩阵
        #status则返回一个列表来表征匹配成功的特征点。
        #ptsA,ptsB为关键点
        #cv2.RANSAC, ransacReprojThreshold这两个参数与RANSAC有关

        # imOut是配准以后的完整图
        imgOut = cv2.warpPerspective(img2, H, (img1.shape[1],img1.shape[0]),flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,borderValue=[255,255,255])
        return imgOut

    def cal_mask_edge(self, img1, img2):
        # 1为破损图，2为模板图
        # img1 = self.img1
        # img2 = self.img2
        img_match = self.image_registration(img1,img2)
        # 二值化
        # img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        # img2_gray = cv.cvtColor(img_match, cv.COLOR_BGR2GRAY)
        _, binary1 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        _, binary2 = cv2.threshold(img_match, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # io.imsave("datasets/bi1.png",binary1)
        # io.imsave("datasets/bi2.png", binary2)
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
        outi = cv2.erode(outi,kernel1)
        outi = cv2.dilate(outi,kernel2)
        oute = canny(outi, sigma=2).astype(np.float)
        return outi, oute
   

def generate_stroke_mask(im_size, parts=15, maxVertex=25, maxLength=100, maxBrushWidth=24, maxAngle=360):
    mask = np.zeros((im_size[0], im_size[1], 1), dtype=np.float32)
    for i in range(parts):
        mask = mask + np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, im_size[0], im_size[1])
    mask = np.minimum(mask, 1.0)
    mask = np.concatenate([mask, mask, mask], axis = 2)
    return mask

def np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
    mask = np.zeros((h, w, 1), np.float32)
    numVertex = np.random.randint(maxVertex + 1)
    startY = np.random.randint(h)
    startX = np.random.randint(w)
    brushWidth = 0
    for i in range(numVertex):
        angle = np.random.randint(maxAngle + 1)
        angle = angle / 360.0 * 2 * np.pi
        if i % 2 == 0:
            angle = 2 * np.pi - angle
        length = np.random.randint(maxLength + 1)
        brushWidth = np.random.randint(10, maxBrushWidth + 1) // 2 * 2
        nextY = startY + length * np.cos(angle)
        nextX = startX + length * np.sin(angle)
        nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(np.int)
        nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(np.int)
        cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
        startY, startX = nextY, nextX
    cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
    return mask