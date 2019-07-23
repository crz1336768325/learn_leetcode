import torch
import torch.nn as nn

import torchvision.datasets as datasets

import Configshufflenet as Config

import cv2
import numpy as np
import pickle

import os
import glob
import platform
if platform.system()=='Windows':
    SplitSym = '\\'
else:
    SplitSym = '/'

TrainPath = Config.TrainPath
ValPath = Config.ValPath
TestPath = Config.TestPath

_PatchSize = Config.PatchSize

useDistributed = Config.useDistributed
Workers = Config.Workers

class Dataset():
    def __init__(self, data_path, batch_size=64, PatchSize=48,data_mode='Train' ,shuffle=True):
        self.data_path = data_path
        self.batch_size = batch_size
        self.PatchSize = PatchSize
        self.shuffle = shuffle
        self.data_mode =data_mode
        self.prepare()
    
    def prepare(self):
        self.dataset = []
        if self.data_mode=='Train':
            f_mobileattack = open(os.path.join(self.data_path,'attack_mobile_all_expand_flip_color.txt'),'r')
            for line in f_mobileattack.readlines():
        
                line = line.strip().split(' ')
                image_name1 = line[0][:-4]
                image_name = os.path.join(self.data_path,"attack_mobile_all_expand_flip_color"+SplitSym+image_name1+'.jpg')    
                self.dataset.append([image_name,0.0])
            f_real = open(os.path.join(self.data_path,'real_mobile_all_expand_shuffle_flip_night_zoulang_expand_color_remote.txt'),'r')
            for line in f_real.readlines():
        
                line = line.strip().split(' ')
                image_name1 = line[0][:-4]
                image_name = os.path.join(self.data_path,"real_mobile_all_expand_shuffle_flip_night_zoulang_expand_color_remote"+SplitSym+image_name1+'.jpg')    
                self.dataset.append([image_name,3.0])
            '''f_real_night = open(os.path.join(self.data_path,'night_flip.txt'),'r')
            for line in f_real_night.readlines():
        
                line = line.strip().split(' ')
                image_name1 = line[0][:-4]
                image_name = os.path.join(self.data_path,"night_flip"+SplitSym+image_name1+'.jpg')    
                self.dataset.append([image_name,4.0])'''
            f_pc_attack = open(os.path.join(self.data_path,'attack_pc_flip_color.txt'),'r')
            for line in f_pc_attack.readlines():
        
                line = line.strip().split(' ')
                image_name1 = line[0][:-4]
                image_name = os.path.join(self.data_path,"attack_pc_flip_color"+SplitSym+image_name1+'.jpg')    
                self.dataset.append([image_name,1.0])
            f_photo_attack = open(os.path.join(self.data_path,'attackcolorphoto_small1.txt'),'r')
            for line in f_photo_attack.readlines():
        
                line = line.strip().split(' ')
                image_name1 = line[0][:-4]
                image_name = os.path.join(self.data_path,"attackcolorphoto_small1"+SplitSym+image_name1+'.jpg')    
                self.dataset.append([image_name,2.0])
        else:
            f_mobileattack = open(os.path.join(self.data_path,'attack_mobile_all_expand_flip_color_test.txt'),'r')
            for line in f_mobileattack.readlines():
        
                line = line.strip().split(' ')
                image_name1 = line[0][:-4]
                image_name = os.path.join(self.data_path,"attack_mobile_all_expand_flip_color"+SplitSym+image_name1+'.jpg')    
                self.dataset.append([image_name,0.0])
            f_real = open(os.path.join(self.data_path,'real_mobile_all_expand_shuffle_flip_night_zoulang_expand_color_remote_test.txt'),'r')
            for line in f_real.readlines():
        
                line = line.strip().split(' ')
                image_name1 = line[0][:-4]
                image_name = os.path.join(self.data_path,"real_mobile_all_expand_shuffle_flip_night_zoulang_expand_color_remote"+SplitSym+image_name1+'.jpg')    
                self.dataset.append([image_name,3.0])
            '''f_real_night = open(os.path.join(self.data_path,'night_flip.txt'),'r')
            for line in f_real_night.readlines():
        
                line = line.strip().split(' ')
                image_name1 = line[0][:-4]
                image_name = os.path.join(self.data_path,"night_flip"+SplitSym+image_name1+'.jpg')    
                self.dataset.append([image_name,4.0])'''
            f_pc_attack = open(os.path.join(self.data_path,'attack_pc_flip_color_test.txt'),'r')
            for line in f_pc_attack.readlines():
        
                line = line.strip().split(' ')
                image_name1 = line[0][:-4]
                image_name = os.path.join(self.data_path,"attack_pc_flip_color"+SplitSym+image_name1+'.jpg')    
                self.dataset.append([image_name,1.0])
            f_photo_attack = open(os.path.join(self.data_path,'attackcolorphoto_small1_test.txt'),'r')
            for line in f_photo_attack.readlines():
        
                line = line.strip().split(' ')
                image_name1 = line[0][:-4]
                image_name = os.path.join(self.data_path,"attackcolorphoto_small1"+SplitSym+image_name1+'.jpg')    
                self.dataset.append([image_name,2.0])
        if self.shuffle:
            print('shuffle dataset ...')
            np.random.shuffle(self.dataset)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        dataset = self.dataset[idx]
        
        pickle_path = dataset[0]
        label = dataset[1]
        #print("f",pickle_path)
        image=cv2.imread(pickle_path)
        #scale = float(800 / w)
        #size = (int(480),int(640))
        #image = cv2.resize(image,size)
        #image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        image = (image-127.5)*0.0078125
        #Image=image.astype(np.float32)
        Image = image.transpose((2,0,1)).astype(np.float32)
        Label = np.array(label).astype(np.int64)

        Image = torch.from_numpy(Image).type(torch.FloatTensor)
        Label = torch.from_numpy(Label).type(torch.LongTensor)
        #print("Image",len(Image))
        #print("Label",Label)
        return Image, Label

def DataLoader(mode='Train', batch_size=64, PatchSize=224):
    if mode=='Train':
        data_path = TrainPath
    elif mode=='Val':
        data_path = ValPath
    elif mode=='Test':
        data_path = TestPath
    else:
        raise ValueError('mode must be Train,Val or Test')
    
    dataset = Dataset(data_path, batch_size=batch_size, PatchSize=PatchSize,data_mode=mode)
    
    if useDistributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None
    
    dataloader = torch.utils.data.DataLoader(
                                              dataset,
                                              batch_size=batch_size,
                                              shuffle=(sampler is None) if mode=='Train' else False,
                                              num_workers=Workers,
                                              pin_memory=True,
                                              sampler=sampler if mode=='Train' else None
                                             )
    
    return dataloader, sampler

if __name__=='__main__':
    
    dataloader,sampler = DataLoader(mode='Train',batch_size=64,PatchSize=200)
    for Input,Target in dataloader:
        print(Input.shape,Target.shape,Target)
    
    #data=Dataset(TrainPath)
    #face=data[1]
    #print(face)
    
    
