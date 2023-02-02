import math
from modules.partialconv2d import PartialConv2d
from modules.PConvLayer import PConvLayer
from modules.VSRLayer import VSRLayer
import modules.Attention as Attention
from torchvision.utils import save_image
# save_image(edge, "src/edge.png")
from skimage import io
# io.imsave("datasets/img1.png", img1)
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision import models
from PIL import Image
from imageio import imread
import cv2 as cv
import random
import copy
from skimage.feature import canny
from skimage.color import rgb2gray


PartialConv = PartialConv2d

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu(out)

        return out

class PRVSNet(BaseNetwork):
    def __init__(self, layer_size=8, input_channels=3, att = True):
        super().__init__()
        self.target_size = 256
        self.layer_size = layer_size
        self.enc_1 = VSRLayer(3, 64, kernel_size = 7)
        self.enc_2 = VSRLayer(64, 128, kernel_size = 5)
        self.enc_3 = PConvLayer(128, 256, sample='down-5')
        self.enc_4 = PConvLayer(256, 512, sample='down-3')
        
        # 后四层
        for i in range(4, self.layer_size):
            name = 'enc_{:d}'.format(i + 1)
            setattr(self, name, PConvLayer(512, 512, sample='down-3'))
        self.deconv = nn.ConvTranspose2d(512, 512, 4, 2, 1)
        for i in range(4, self.layer_size):
            name = 'dec_{:d}'.format(i + 1)
            setattr(self, name, PConvLayer(512 + 512, 512, activ='leaky', deconv = True))
        self.dec_4 = PConvLayer(512 + 256, 256, activ='leaky', deconv = True)
        if att:
            self.att = Attention.AttentionModule()
        else:
            self.att = lambda x:x
        self.dec_3 = PConvLayer(256 + 128, 128, activ='leaky', deconv = True)
        self.dec_2 = VSRLayer(128 + 64, 64, stride = 1, activation='leaky', deconv = True)
        self.dec_1 = VSRLayer(64 + input_channels, 64, stride = 1, activation = None, batch_norm = False)
        self.resolver = Bottleneck(64,16)
        self.output = nn.Conv2d(128, 3, 1)
        
    def forward(self, input, input_mask, mask_edge, input_edge):
        
        h_dict = {}  # for the output of enc_N
        h_mask_dict = {}  # for the output of enc_N
        h_edge_list = []
        h_dict['h_0'], h_mask_dict['h_0'] = input, input_mask
        edge = input_edge
        # save_image(edge, "src/edge.png")
        h_key_prev = 'h_0'
        for i in range(1, self.layer_size + 1):
            l_key = 'enc_{:d}'.format(i)
            h_key = 'h_{:d}'.format(i)
            if i not in [1, 2]:
                h_dict[h_key], h_mask_dict[h_key] = getattr(self, l_key)(h_dict[h_key_prev], h_mask_dict[h_key_prev])
            else:
                h_dict[h_key], h_mask_dict[h_key], edge = getattr(self, l_key)(h_dict[h_key_prev], h_mask_dict[h_key_prev], edge, mask_edge)
                h_edge_list.append(edge)
            h_key_prev = h_key

        h_key = 'h_{:d}'.format(self.layer_size)
        h, h_mask = h_dict[h_key], h_mask_dict[h_key]
        h = self.deconv(h)
        h_mask = F.interpolate(h_mask, scale_factor = 2)
        
        for i in range(self.layer_size, 0, -1):
            enc_h_key = 'h_{:d}'.format(i - 1)
            dec_l_key = 'dec_{:d}'.format(i)
            h_mask = torch.cat([h_mask, h_mask_dict[enc_h_key]], dim = 1)
            h = torch.cat([h, h_dict[enc_h_key]], dim = 1)
            if i not in [2, 1]:
                h, h_mask = getattr(self, dec_l_key)(h, h_mask)
            else:
                edge = h_edge_list[i-1]
                h, h_mask, edge = getattr(self, dec_l_key)(h, h_mask, edge, mask_edge)
                h_edge_list.append(edge)
            if i == 4:
                h = self.att(h, h_mask)
     ######################################## ########################################
     ################################# edge of small mask
        # img2 = cv.imread("datasets/md.png",0) # 模板即img2
        # img2 = np.array(Image.fromarray(img2).resize(size=(self.target_size, self.target_size)))
        # img1 = masked_grey_image.view(256,256)
        # # print(img1.max())

        # # save_image(img1, "datasets/img1.png")
        # img1 = 255 - img1.data.cpu().numpy()*255
        # img1 = np.uint8(img1)
        # img2 = np.uint8(img2)
        # io.imsave("datasets/img1.png", img1)

        # matchi = self.image_registration(img1,img2)
        # io.imsave("datasets/matchmd.png", matchi)
        # bini, edgei = self.cal_mask_edge(img1,img2)
        # io.imsave("datasets/outedge.png", edgei)
        # io.imsave("datasets/outbini.png", bini)
        # before = self.calculate_ssim(img1, img2)
        # after = self.calculate_ssim(img1, matchi)
        # # print("before:%.4f" %(before))
        # # print("after:%.4f" %(after))            
        # # if after-before>0.1:
        # edgei128 = Image.fromarray(edgei).resize(size=(128,128))
        # edgei128 = np.array(edgei128)
        # edgei128 = torch.from_numpy(edgei128)
        # edgei128 = edgei128.view(1,1,128,128)

        # edgei = torch.from_numpy(edgei)
        # edgei = edgei.view(1,1,256,256)

        # edgei128 = edgei128.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        # edgei = edgei.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # h_edge_list[0] += edgei
        # h_edge_list[1] += edgei128
        # h_edge_list[2] += edgei128
        # h_edge_list[3] += edgei

            
     ######################################## ########################################
     ######################################## edge of small mask
     #输出中间边界图
        # for i in range(len(h_edge_list)):
        #     file_path = 'src/edge{:d}.png'.format(i)
        #     save_image(h_edge_list[i], file_path)

        h_out = self.resolver(h)
        h_out = torch.cat([h_out, h], dim = 1)
        h_out = self.output(h_out)
        # h_edge_list[-1] 是预测后的edge
        return h_out, h_mask, h_edge_list[-2], h_edge_list[-1]

    def ssim(self, img1, img2):
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

    def cal_mask_edge(self, img1, img2):
        # 1为破损图，2为模板图
        # img1 = self.img1
        # img2 = self.img2
        img_match = self.image_registration(img1,img2)
        # 二值化
        # img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        # img2_gray = cv.cvtColor(img_match, cv.COLOR_BGR2GRAY)
        _, binary1 = cv.threshold(img1, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        _, binary2 = cv.threshold(img_match, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        # io.imsave("datasets/bi1.png",binary1)
        # io.imsave("datasets/bi2.png", binary2)
        # print(binary1.shape)
        outi = binary1-binary2
        # kernel1 = np.ones((3,3),np.uint8)
        kernel1 = [[1,1,1],[0,1,0],[1,1,1]]
        kernel1 = np.uint8(kernel1)
        kernel2 = np.ones((2,2),np.uint8)

        ## c.图像的腐蚀，默认迭代次数
        outi = cv.erode(outi,kernel1)
        outi = cv.dilate(outi,kernel2)
        oute = canny(outi, sigma=2).astype(np.float)
        return outi, oute
   

    def train(self, mode=True, finetune = False):
        super().train(mode)
        if finetune:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                    module.eval()
    
