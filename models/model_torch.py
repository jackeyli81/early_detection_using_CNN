
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from common import config
import os
os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIB_DEVI



def conv3x3(in_planes, out_planes, strides = 1):
    
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes,
                   kernel_size = 3, stride = strides,
                   padding = 1, bias = False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU()
    )
    # return nn.Conv2d(in_planes, out_planes,
    #                  kernel_size = 3, stride = strides,
    #                  padding = 1, bias = False)

class double_block(nn.Module):
    def __init__(self, in_channel = 3,
                 out_channel = 16, strides = 2,
                 pool_manner = 'MAX'):
        super(double_block, self).__init__()
        self.conv1 = conv3x3(in_channel, out_channel)
        self.conv2 = conv3x3(out_channel, out_channel)
        self.strides = strides
    def forward(self, x):
        out = self.conv2(self.conv1(x))
        h, w = out.shape[2:]
        out = nn.AdaptiveAvgPool2d( ((1+h)//self.strides,
                                     w//self.strides ) )(out)
        return out


class torch_model(nn.Module):
    
    def __init__(self):
        super(torch_model, self).__init__()
        self.layers = {}
        
        self.layer1 = double_block(in_channel=3, out_channel=16)
        self.layer2 = double_block(in_channel=16, out_channel=32)
        self.layer3 = double_block(in_channel=32, out_channel=64)
        self.layer4 = double_block(in_channel=64, out_channel=128)
        self.layer5 = double_block(in_channel=128, out_channel=256,
                                   strides=4)
        self.layer6 = double_block(in_channel=256, out_channel=16,
                                   strides=4)
        self.fc_layer1 = nn.Linear(in_features = 16*2, out_features = 2,
                                  bias = False )
        # self.fc_layer2 = nn.Linear(in_features = 512, out_features = 1,
                                #   bias = False )
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        # print('line 68', out.shape)
        out = out.reshape( (out.shape[0],-1) )
        out = self.fc_layer1(out)

        # print('line 70', out.shape)
        # out = self.fc_layer2(out.squeeze())
        # print('line 72', out.shape)
        return nn.Softmax(dim=1)(out.squeeze())










