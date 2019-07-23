import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from collections import OrderedDict

from Configshufflenet import net_scale
from Configshufflenet import num_output

def conv3x3(in_channels,out_channels,stride=1,padding=1,bias=True,groups=1):
    return nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=padding,bias=bias,groups=groups)
def conv1x1(in_channels,out_channels,bias=True,groups=1):
    return nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0,bias=bias,groups=groups)
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x
def channel_split(x,splits=[24,24]):
    return torch.split(x,splits,dim=1)

isBias = True
isDropout = True
isResidual = True
Activate = nn.LeakyReLU
def cls_loss(pred_label,gt_label,useGPU):
    
    pred_label = torch.squeeze(pred_label)
    gt_label = torch.squeeze(gt_label)
    
    weight = torch.FloatTensor([2.0,2.0,2.0,1.0])
    #weight = torch.FloatTensor([1.0,1.0,1.0,1.0,1.0])
    if (useGPU or useGPU==0) and torch.cuda.is_available():
        if isinstance(useGPU,list):
            loss_cls = nn.CrossEntropyLoss(weight=weight,reduce=True,size_average=True).cuda(useGPU[-1])
        else:
            loss_cls = nn.CrossEntropyLoss(weight=weight,reduce=True,size_average=True).cuda(useGPU)
    else:
        loss_cls = nn.CrossEntropyLoss(weight=weight,reduce=True,size_average=True)
    
    return loss_cls(pred_label,gt_label)

class PrimaryModule(nn.Module):
    def __init__(self, in_channels=3, out_channels=24):
        super(PrimaryModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.PrimaryModule = nn.Sequential(
                                            OrderedDict(
                                                        [
                                                         ('ParimaryConv',conv3x3(in_channels,out_channels,2,1,isBias,1)),
                                                         ('ParimaryConvBN',nn.BatchNorm2d(out_channels)),
                                                         ('ParimaryConvReLU',Activate()),
                                                         ('ParimaryMaxPool',nn.MaxPool2d(kernel_size=3,stride=2,padding=0,ceil_mode=True))
                                                        ]
                                                        )
                                            )
        
    def forward(self,x):
        x = self.PrimaryModule(x)
        return x

class FinalModule(nn.Module):
    def __init__(self,in_channels=464,out_channels=1024,num_classes=1000):
        super(FinalModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        
        self.FinalConv = nn.Sequential(
                                       OrderedDict(
                                                   [
                                                    ('FinalConv',conv1x1(in_channels,out_channels,isBias,1)),
                                                    ('FinalConvBN',nn.BatchNorm2d(out_channels)),
                                                    ('FinalConvReLU',Activate())
                                                   ]
                                                   )
                                       )
        if isDropout:
            self.FinalDropout = nn.Sequential(
                                              OrderedDict(
                                                          [
                                                           ('FinalDropout', nn.Dropout(0.25))
                                                          ]
                                                          )
                                              )
        self.FC = nn.Sequential(
                                OrderedDict(
                                            [
                                             ('FC',conv1x1(100,num_classes,isBias,1))
                                            ]
                                            )
                                )
        self.final_DW = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=0,
                                  groups=out_channels, bias=False),
                                     )
        self.convv=nn.Sequential(nn.Conv2d(out_channels,100,kernel_size=3))
    def forward(self,x):
        x = self.FinalConv(x)
        print("final",x.shape)
        #x = F.avg_pool2d(x, x.data.size()[-2:])
        x=self.final_DW(x)
        x=self.convv(x)
        print("av pool",x.shape)
        if isDropout:
            x = self.FinalDropout(x)
        x = self.FC(x)
        print("FC",x.shape)
        x = x.view(x.size(0),x.size(1))
        print("view",x.shape)
        return x

class ShuffleNetV2Block1(nn.Module):
    def __init__(self, in_channels, split_rate=2):
        super(ShuffleNetV2Block1, self).__init__()
        self.in_channels = in_channels
        self.left_channels = in_channels//split_rate
        self.right_channels = in_channels - self.left_channels
        
        self.right = nn.Sequential(
                                   OrderedDict(
                                               [
                                                ('Equal1x1Conv0', conv1x1(self.right_channels,self.right_channels,isBias,1)),
                                                ('Equal1x1Conv0BN', nn.BatchNorm2d(self.right_channels)),
                                                ('Equal1x1Conv0ReLU', Activate()),
                                                ('Depthwise3x3Conv', conv3x3(self.right_channels,self.right_channels,1,1,isBias,self.right_channels)),
                                                ('Depthwise3x3ConvBN', nn.BatchNorm2d(self.right_channels)),
                                                ('Equal1x1Conv1', conv1x1(self.right_channels,self.right_channels,isBias,1)),
                                                ('Equal1x1Conv1BN', nn.BatchNorm2d(self.right_channels))
                                               ]
                                               )
                                   )
        self.FinalReLU = nn.Sequential(
                                       OrderedDict(
                                                   [
                                                    ('FinalReLU', Activate())
                                                   ]
                                                   )
                                       )
        
    def forward(self, x):
        x_left,x_right = channel_split(x, splits=[self.left_channels,self.right_channels])
        
        if isResidual:
            _x_right = x_right
        x_right = self.right(x_right)
        if isResidual:
            x_right = x_right + _x_right
        x_right = self.FinalReLU(x_right)
        
        x = torch.cat([x_left,x_right], dim=1)
        x = channel_shuffle(x, 2)
        return x
class ShuffleNetV2Block2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, padding=1):
        super(ShuffleNetV2Block2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        
        self.left = nn.Sequential(
                                  OrderedDict(
                                              [
                                               ('Depthwise3x3Conv', conv3x3(in_channels,in_channels,stride,self.padding,isBias,in_channels)),
                                               ('Depthwise3x3ConvBN', nn.BatchNorm2d(in_channels)),
                                               ('NoEqual1x1Conv', conv1x1(in_channels,out_channels//2,isBias,1)),
                                               ('NoEqual1x1ConvBN', nn.BatchNorm2d(out_channels//2)),
                                               ('NoEqual1x1ConvReLU', Activate())
                                              ]
                                              )
                                  )
        self.right = nn.Sequential(
                                   OrderedDict(
                                               [
                                                ('Equal1x1Conv', conv1x1(in_channels,in_channels,isBias,1)),
                                                ('Equal1x1ConvBN', nn.BatchNorm2d(in_channels)),
                                                ('Equal1x1ConvReLU', Activate()),
                                                ('Depthwise3x3Conv', conv3x3(in_channels,in_channels,stride,self.padding,isBias,in_channels)),
                                                ('Depthwise3x3ConvBN', nn.BatchNorm2d(in_channels)),
                                                ('NoEqual1x1Conv', conv1x1(in_channels,out_channels//2,isBias,1)),
                                                ('NoEqual1x1ConvBN', nn.BatchNorm2d(out_channels//2)),
                                                ('NoEqual1x1ConvReLU', Activate())
                                               ]
                                               )
                                   )
    def forward(self, x):
        x_left = self.left(x)
        x_right = self.right(x)
        x = torch.cat([x_left,x_right], dim=1)
        x = channel_shuffle(x, 2)
        return x

class ShuffleNetV2(nn.Module):
    def __init__(self, in_channels=3, num_output=num_output, net_scale=net_scale, split_rate=2):
        super(ShuffleNetV2, self).__init__()
        self.in_channels = in_channels
        self.num_output = num_output
        self.split_rate = split_rate
        
        if net_scale==0.3:
            self.out_channels = [16,32,64,128,1024]
        elif net_scale==0.5:
            self.out_channels = [24,48,96,192,1024]
        elif net_scale==0.7:
            self.out_channels = [32,64,128,256,1024]
        elif net_scale==0.8:
            self.out_channels = [48,96,192,384,1024]
        elif net_scale==1.0:
            self.out_channels = [24,116,232,464,1024]
        elif net_scale==1.5:
            self.out_channels = [24,176,352,704,1024]
        elif net_scale==2.0:
            self.out_channels = [24,244,488,976,2048]
        else:
            raise ValueError('net_scale must be 0.3,0.5,0.7,1.0,1.5 or 2.0')
        
        self.PrimaryModule = PrimaryModule(in_channels,self.out_channels[0])
        
        self.Stage1 = self.Stage(1, [1,3], 2)
        self.Stage2 = self.Stage(2, [1,7], 2)
        self.Stage3 = self.Stage(3, [1,3], 2)
        
        self.FinalModule = FinalModule(self.out_channels[3],self.out_channels[4],self.num_output)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
        
    def Stage(self, stage, BlockRepeat=[1,3], stride=2, padding=1):
        modules = OrderedDict()
        name = 'ShuffleNetV2Stage_{}'.format(stage)
        
        if BlockRepeat[0]==1:
            modules[name+'_0'] = ShuffleNetV2Block2(self.out_channels[stage-1],self.out_channels[stage],stride,padding)
        else:
            raise ValueError('stage first block must only repeat 1 time')
        
        for i in range(BlockRepeat[1]):
            modules[name+'_{}'.format(i+1)] = ShuffleNetV2Block1(self.out_channels[stage],self.split_rate)
        
        return nn.Sequential(modules)
    
    def forward(self, x):
        x = self.PrimaryModule(x)
        print("x",x.shape)
        x = self.Stage1(x)
        print("x",x.shape)
        x = self.Stage2(x)
        print("x",x.shape)
        x = self.Stage3(x)
        print("x",x.shape)
        x = self.FinalModule(x)
        print("x",x.shape)
        return x
        
        
if __name__=='__main__':
    import time
    net = ShuffleNetV2(3,num_output,net_scale)
    input = torch.randn(1,3,224,224)
    start_time = time.time()
    output = net(input)
    print('spend time : {}'.format(time.time()-start_time))
    print('input size : {}'.format(input.size()))
    print('output size : {}'.format(output.size()))
    
    params = list(net.parameters())
    num = 0
    for i in params:
        l=1
        #print('Size:{}'.format(list(i.size())))
        for j in i.size():
            l *= j
        num += l
    print('All Parameters:{}'.format(num))
    
    #torch.save(net.state_dict(),'ShuffleNetV2.pth')
