import torch
import torch.nn as nn
import torchvision

def ConvBNReLU(in_channels,out_channels,kernel_size):

    if kernel_size==3 or kernel_size==5:
        return nn.Sequential(
            # nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,padding=kernel_size//2),

            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=[kernel_size,1], stride=1,padding=[kernel_size//2,0]),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=[1,kernel_size], stride=1,padding=[0,kernel_size//2]),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True))
    else:
        return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,padding=kernel_size//2),

    
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True) 

    )

class InceptionV1Module(nn.Module):
    def __init__(self, in_channels,out_channels1, out_channels2reduce,out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV1Module, self).__init__()

        self.branch1_conv = ConvBNReLU(in_channels=in_channels,out_channels=out_channels1,kernel_size=1)
        
        self.branch2_conv1 = ConvBNReLU(in_channels=in_channels,out_channels=out_channels2reduce,kernel_size=1)
        self.branch2_conv2 = ConvBNReLU(in_channels=out_channels2reduce,out_channels=out_channels2,kernel_size=3)

        self.branch3_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels3reduce, kernel_size=1)
        self.branch3_conv2 = ConvBNReLU(in_channels=out_channels3reduce, out_channels=out_channels3, kernel_size=5)

        self.branch4_pool = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.branch4_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels4, kernel_size=1)
    def forward(self,x):
        out1 = self.branch1_conv(x)
        out2 = self.branch2_conv2(self.branch2_conv1(x))
        out3 = self.branch3_conv2(self.branch3_conv1(x))
        out4 = self.branch4_conv1(self.branch4_pool(x))
        print(out4.shape)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        
        return out


inputs=torch.ones([4,3,224,224])
module=InceptionV1Module(3,16,16,16,16,16,16)
out=module(inputs)
print(out.shape)