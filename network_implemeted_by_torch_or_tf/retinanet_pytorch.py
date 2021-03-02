import torch
import torch.nn as nn

import numpy  as np
def conv(in_channels,out_channels,kernel_size,stride=1,pad=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, 
        stride=stride, padding=pad),
        nn.BatchNorm2d(out_channels), #添加了BN层
        nn.ReLU(inplace=True)
    )
def upconv(in_channels,out_channels,kernel_size,stride=1,pad=0):
     return nn.ConvTranspose2d(in_channels, out_channels, kernel_size, 
stride=stride, padding=pad, dilation=1, groups=1, bias=True, padding_mode='zeros')
def pool(kernel_size,stride):
    return nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

class retinanet(nn.Module):
    def __init__(self):

        pass 
    def forward(self,x):
        x=self.conv(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        


img=torch.ones([4,3,672,640])




