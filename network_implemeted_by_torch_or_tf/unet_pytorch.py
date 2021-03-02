import torch
import torch.nn as nn
import numpy as np
from torchsummary import summary
from torchviz import make_dot 
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

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.downconv1_1=conv(3,64,3,1)
        self.downconv1_2=conv(64,64,3,1)
        self.pool_down1=pool(2,2)
        self.downconv2_1=conv(64,128,3,1)
        self.downconv2_2=conv(128,128,3,1)
        self.pool_down2=pool(2,2)
        self.downconv3_1=conv(128,256,3,1)
        self.downconv3_2=conv(256,256,3,1)
        self.pool_down3=pool(2,2)
        self.downconv4_1=conv(256,512,3,1)
        self.downconv4_2=conv(512,512,3,1)      
        self.pool_down4=pool(2,2)
        self.downconv5_1=conv(512,1024,3,1)
        self.downconv5_2=conv(1024,1024,3,1)   


        self.up_conv4_1=upconv(1024,512,2,2)   
        self.up_conv4_2=conv(1024,512,3,1)      
        self.up_conv4_3=conv(512,512,3,1)  
        
        self.up_conv3_1=upconv(512,256,2,2)  
        self.up_conv3_2=conv(512,256,3,1)      
        self.up_conv3_3=conv(256,256,3,1)  

        self.up_conv2_1=upconv(256,128,2,2)  
        self.up_conv2_2=conv(256,128,3,1)      
        self.up_conv2_3=conv(128,128,3,1)  

        self.up_conv1_1=upconv(128,64,2,2)  
        self.up_conv1_2=conv(128,64,3,1)      
        self.up_conv1_3=conv(64,64,3,1)  
        self.up_conv1_4=conv(64,2,1,1,0)  

    def forward(self,x):
        print("ste1",x.shape)
        x= self.downconv1_1(x)
        print("ste2",x.shape)
        pool_1_x_before= self.downconv1_2(x)
        pool_1_x=self.pool_down1(pool_1_x_before)

        x=self.downconv2_1(pool_1_x)
        pool_2_x_before=self.downconv2_2(x)
        pool_2_x=self.pool_down2(pool_2_x_before)

        x=self.downconv3_1(pool_2_x)
        pool_3_x_before=self.downconv3_2(x)
        pool_3_x=self.pool_down3(pool_3_x_before)

        x=self.downconv4_1(pool_3_x)
        pool_4_x_before=self.downconv4_2(x)
        pool_4_x=self.pool_down4(pool_4_x_before)

        x=self.downconv5_1(pool_4_x)
        x=self.downconv5_2(x)

        x=self.up_conv4_1(x)
        print("pool_4_x_before",pool_4_x_before.shape)
        print("x",x.shape)
        x=torch.cat([pool_4_x_before,x],1)
        x=self.up_conv4_2(x)
        x=self.up_conv4_3(x)

        x=self.up_conv3_1(x)
        print("pool_3_x_before",pool_3_x_before.shape)
        print("x",x.shape)
        x=torch.cat([pool_3_x_before,x],1)
        x=self.up_conv3_2(x)
        x=self.up_conv3_3(x)

        x=self.up_conv2_1(x)
        x=torch.cat([pool_2_x_before,x],1)
        x=self.up_conv2_2(x)
        x=self.up_conv2_3(x)


        x=self.up_conv1_1(x)
        x=torch.cat([pool_1_x_before,x],1)
        x=self.up_conv1_2(x)
        x=self.up_conv1_3(x)
        x=self.up_conv1_4(x)
        return  x




torch.cuda.set_device(-1)


x=torch.ones([4,3,400,400])
import numpy as np

# unet=Unet()
# unet=unet.cuda()
# torch.save(unet,"lll.pth")
# out=unet(x)
net=torch.load("lll.pth")
summary(net,(3,400,400))
net=net.cuda()
x=x.cuda()
out=net(x)
# print(unet)
print("out",out.shape)
vis_graph = make_dot(out, params=dict(net.named_parameters()))
# print(out)

print("-------------------")
# vis_graph.view()
vis_graph.render(filename='xxx', view=False, format='pdf')

