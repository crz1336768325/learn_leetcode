import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim

from DataLoader_shufflenet import DataLoader
from ShuffleNetV2 import ShuffleNetV2 as Model
from ShuffleNetV2 import cls_loss
#from antispoofShuffleNetV2 import ShuffleNetV2 as Model
#from antispoofShuffleNetV2 import cls_loss
#from squeezenet1_1 import SqueezeNet as Model
#from squeezenet1_1 import cls_loss
import Configshufflenet as Config

import os
import shutil

useGPU = Config.useGPU

useDistributed = Config.useDistributed
DistBackend = Config.DistBackend
DistUrl = Config.DistUrl
WorldSize = Config.WorldSize

BestScore = 0

StartEpoch = Config.StartEpoch
Epoch = Config.Epoch
Lr = Config.Lr
Momentum = Config.Momentum
WeightDecay = Config.WeightDecay

ModelPath = Config.ModelPath
PretrainModelPath = Config.PretrainModelPath

PrintFreq = Config.PrintFreq
Mode = Config.Mode

BatchSize = Config.BatchSize
VarianceFreq = 50
VarianceStep = 32
MaxBatchSize = 1024

in_channels = Config.in_channels
PatchSize = Config.PatchSize

def Main():
    global BatchSize
    print("in_channels",in_channels)
    #model = Model(in_channels=in_channels)
    #model = Model(3,4,0.7)
    model = Model(3,4,0.8)
    #model=Model()
    print("env setup")
    model,optimizer = EnvironmentSetup(model)
    print("load param")
    model,optimizer = LoadParameters(model,optimizer,PretrainModelPath)
    
    train_loader,train_sample = DataLoader(mode='Train',batch_size=BatchSize,PatchSize=PatchSize)
    val_loader,val_sample = DataLoader(mode='Val',batch_size=BatchSize,PatchSize=PatchSize)
    test_loader,test_sample = DataLoader(mode='Test',batch_size=BatchSize,PatchSize=PatchSize)
    if Mode=='Val':
        _ = validate_or_test(val_loader,model,cls_loss)
        return
    elif Mode=='Test':
        _ = validate_or_test(test_loader,model,cls_loss)
        return
    else:
        for epoch in range(StartEpoch,Epoch):
            if epoch%VarianceFreq==0 and epoch!=0:
                BatchSize = BatchSize+VarianceStep if BatchSize+VarianceStep<=MaxBatchSize else MaxBatchSize
                train_loader,train_sample = DataLoader(mode='Train',batch_size=BatchSize,PatchSize=PatchSize)
                
            if useDistributed:
                train_sample.set_epoch(epoch)
            adjust_learning_rate(optimizer,epoch)

            train(train_loader,model,cls_loss,optimizer,epoch)
            acc = validate_or_test(val_loader,model,cls_loss)             
            
            
            SaveParameters(model,optimizer,epoch,acc)

def train(train_loader,model,loss_fn,optimizer,epoch):
    losses = AverageMeter()
    Acc = AverageMeter()

    model.train()

    for i, (input, target) in enumerate(train_loader):
        if (useGPU or useGPU==0) and torch.cuda.is_available():
            if isinstance(useGPU,list):
                input = input.cuda(useGPU[-1], non_blocking=True)
                target = target.cuda(useGPU[-1], non_blocking=True)
            else:
                input = input.cuda(useGPU, non_blocking=True)
                target = target.cuda(useGPU, non_blocking=True)

        output = model(input)
        print("output",output.size())
        print('input',input.size())
        print('target',target.size())
        loss = loss_fn(output, target, useGPU)

        
        acc = accuracy(output, target)
        losses.update(loss.item(), input.size(0))
        Acc.update(acc.item(), input.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #if i %100 ==0:
            #SaveParameters(model,optimizer,epoch,acc)
        if i % PrintFreq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.6f} ({loss.avg:.6f})\t'
                  'Acc {Acc.val:.5f} ({Acc.avg:.5f})'.format(
                   epoch, i, len(train_loader), loss=losses, Acc=Acc))
def validate_or_test(loader, model, loss_fn):
    batch_time = AverageMeter()
    losses = AverageMeter()
    Acc = AverageMeter()
    
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            if (useGPU or useGPU==0) and torch.cuda.is_available():
                if isinstance(useGPU,list):
                    input = input.cuda(useGPU[-1], non_blocking=True)
                    target = target.cuda(useGPU[-1], non_blocking=True)
                else:
                    input = input.cuda(useGPU, non_blocking=True)
                    target = target.cuda(useGPU, non_blocking=True)

            output = model(input)
            #print("out",output.size())
            loss = loss_fn(output, target, useGPU)
            #print('input',input)
            #print('target',target)
            acc = accuracy(output, target)
            losses.update(loss.item(), input.size(0))
            Acc.update(acc.item(), input.size(0))
            
            if i % PrintFreq == 0:
                print('Test: [{0}/{1}]\t'
                      'Loss {loss.val:.6f} ({loss.avg:.6f})\t'
                      'Acc {Acc.val:.5f} ({Acc.avg:.5f})'.format(
                       i, len(loader), loss=losses, Acc=Acc))

        print(' * Acc {Acc.avg:.5f}'.format(Acc=Acc))

    return Acc.avg
def LoadParameters(model,optimizer,path):
    if not os.path.exists(path):
        print('pretrain model file is not exists...')
        return model,optimizer
    else:
        pass
    
    net_dict = model.state_dict()
    optimizer_dict = optimizer.state_dict()
    
    global StartEpoch
    global BestScore
    
    try:
        pretrain = torch.load(path)
    except:
        pretrain = torch.load(path, map_location=lambda storage, loc: storage)
    
    for k, v in pretrain.items():
        try:
            #All model parameters
            if k=='state_dict':
                for keys in v:
                    net_dict.update( { keys:v[keys] } )
                print('load state_dict')
            elif k=='Epoch':
                StartEpoch = v
                print('load Epoch')
            elif k=='BestScore':
                BestScore = v
                print('load BestScore')
            elif k=='optimizer':
                for keys in v:
                    optimizer_dict.update( { keys:v[keys] } )
                print('load optimizer')
            else:
                raise ValueError('should load state_dict')
        except:
            #Only net state_dict
            net_dict.update( { k:v } )
    
    model.load_state_dict(net_dict)
    optimizer.load_state_dict(optimizer_dict)
    return model,optimizer
def SaveParameters(model,optimizer,epoch,score,path=os.path.join(ModelPath,'checkpointshufflenet.pth'),only_save_state_dict=False):
    print("save model")
    global BestScore
    try:
        is_best = score.cpu()>BestScore.cpu()
    except:
        is_best = score>BestScore
    
    if only_save_state_dict:
        torch.save(model.state_dict(),path)
    else:
        state = {
                'state_dict':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'Epoch':epoch,
                'BestScore':max(score,BestScore)
                }
        torch.save(state,path)
    
    if is_best:
        BestScore = score
        print('save new best model')
        shutil.copyfile(path, os.path.join(ModelPath,'model_bestshufflenet.pth'))
def EnvironmentSetup(model):
    if (useGPU or useGPU==0) and torch.cuda.is_available():
        print("env cuda")
        if isinstance(useGPU,list):
            print('use DataParallel by GPU {}'.format(useGPU))
            model = torch.nn.DataParallel(model.cuda(useGPU[-1]),device_ids=useGPU)
        else:
            print('use GPU {}'.format(useGPU))
            model = model.cuda(useGPU)
    elif useDistributed:
        print('use Distributed')
        dist.init_process_group(backend=DistBackend,init_method=DistUrl,world_size=WorldSize)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda())
    else:
        print('use CPU, Low Bitch')
        model = model
    
    #optimizer = torch.optim.SGD(model.parameters(),Lr,momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(),lr=Lr,betas=(0.9,0.999),eps=1e-9,weight_decay=WeightDecay)
    
    return model,optimizer
def adjust_learning_rate(optimizer,epoch):
    lr = Lr * (0.1 ** (epoch // VarianceFreq))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def accuracy(output, target):
    with torch.no_grad():
        #print('output crz',output)
        prob_cls = torch.squeeze(output)
        #print('prob_cls',prob_cls)
        prob_cls = nn.Softmax(dim=-1)(prob_cls)
        #print('prob_cls1',prob_cls)
        prob_cls = torch.max(prob_cls, dim=-1)[1]
        #print('prob_cls2',prob_cls)
        prob_cls = prob_cls.type(torch.FloatTensor)
        #print('prob_cls3',prob_cls)
        gt_cls = torch.squeeze(target).type(torch.FloatTensor)
        #print('gt_cls',gt_cls)

        size = min(gt_cls.size()[0], prob_cls.size()[0])
        right_ones = torch.eq(prob_cls,gt_cls).float()

        return torch.div(torch.mul(torch.sum(right_ones),float(1.0)),float(size))

if __name__=='__main__':
    Main()
