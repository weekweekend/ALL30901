from modules.partialconv2d import PartialConv2d
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

PartialConv = PartialConv2d
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        # kernel_size = 1,3,1;
        # channel numbers = 64,16,64
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

class EdgeGenerator(nn.Module):
    def __init__(self, in_channels_feature, kernel_s = 3, add_last_edge = True):
        super(EdgeGenerator, self).__init__()

        self.p_conv = PartialConv2d(in_channels_feature + 1, 64, kernel_size = kernel_s, stride = 1, padding = kernel_s // 2, multi_channel = True, bias = False)
        
        self.edge_resolver = Bottleneck(64, 16)
        self.out_layer = nn.Conv2d(64, 1, 1, bias = False)
        
    def forward(self, in_x, mask):
        # 得到的 Xpc1, Mpc1
        x, mask_updated = self.p_conv(in_x, mask)
        # Xpc1经过Bottleneck Block 和 卷积层，得到Econv
        x = self.edge_resolver(x)
        edge_out = self.out_layer(x)
        return edge_out, mask_updated

class VSRLayer(nn.Module):
    def __init__(self, in_channel, out_channel, stride = 2, kernel_size = 3, batch_norm = True, activation = "ReLU", deconv = False):
        super(VSRLayer, self).__init__()
        self.edge_generator = EdgeGenerator(in_channel, kernel_s = kernel_size)
        self.feat_rec = PartialConv(in_channel+1, out_channel, stride = stride, kernel_size = kernel_size, padding = kernel_size//2, multi_channel = True)
        if deconv:
            self.deconv = nn.ConvTranspose2d(out_channel, out_channel, 4, 2, 1)
        else:
            self.deconv = lambda x:x
            
        if batch_norm: 
            self.batchnorm = nn.BatchNorm2d(out_channel)
        else:
            self.batchnorm = lambda x:x
            
        self.stride = stride
        
        if activation == "ReLU":
            self.activation = nn.ReLU(True)
        elif activation == "Leaky":
            self.activation = nn.LeakyReLU(0.2, True)
        else:
            self.activation = lambda x:x
    
    def forward(self, feat_in, mask_in, edge_in, mask_edge_in):
        edge_in = F.interpolate(edge_in, size = feat_in.size()[2:])
        mask_edge_in = F.interpolate(mask_edge_in, size = feat_in.size()[2:])
        save_image(edge_in,"src/edge.png")
        # Econv, Mpc1
        edge_updated, mask_updated = self.edge_generator(torch.cat([feat_in, edge_in], dim = 1), torch.cat([mask_in, mask_in[:,:1,:,:]], dim = 1))
        ##### 与匹配所得mask相加
        # edge_updated = edge_updated * mask_edge_in
        # mask_edge_in = mask_edge_in.float() * edge_updated.max()
        # edge_updated = edge_updated + mask_edge_in

        # Eeg = Ein * Min + Econv * (Mpc1 - Min)
        edge_reconstructed = edge_in * mask_in[:,:1,:,:] + edge_updated * (mask_updated[:,:1,:,:] - mask_in[:,:1,:,:]) 
        # 部分反褶积上采样(Partial-deconvolution up-sampling)
        # mask_edge_in = mask_edge_in.float() * edge_reconstructed.max()
        # edge_reconstructed = edge_reconstructed + mask_edge_in
        save_image(mask_edge_in,"src/edgenew.png")
        # 将Xin 和利用Xin预测得到的Eeg串联(按通道拼接），再用另一个PC卷积预测特征
        # 对应各自的mask
        feat_out, feat_mask = self.feat_rec(torch.cat([feat_in, edge_reconstructed], dim = 1), torch.cat([mask_in, mask_updated[:,:1,:,:]], dim = 1))
        # 反卷积得到图
        feat_out = self.deconv(feat_out)
        feat_out = self.batchnorm(feat_out)
        feat_out = self.activation(feat_out)

        # 上下/采样
        mask_updated = F.interpolate(mask_updated, size = feat_out.size()[2:])
        feat_mask = F.interpolate(feat_mask, size = feat_out.size()[2:])

       
        # file_path = 'datasets/featout-{:d}.png'.format((feat_out[0,0:1,:,:].mean()))
        # for i in range(63):
        #     file_path = 'src/featout-{:d}.png'.format(i)
        #     save_image(feat_out[0,i:i+1,:,:], file_path)
        return feat_out, feat_mask*mask_updated[:,0:1,:,:], edge_reconstructed