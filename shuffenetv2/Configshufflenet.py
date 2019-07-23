import os
import torch
root = os.getcwd()

#DataLoader
TrainPath = os.path.join(root,'dataset')
ValPath = os.path.join(root,'dataset')
TestPath = os.path.join(root,'dataset')

PatchSize = 224
in_channels = 3

BatchSize = 32
Workers = 0
num_output=4
#num_output=5
#Distributed
useDistributed = False#False           ,True
DistBackend = 'gloo'
DistUrl = 'tcp://224.66.41.62:23456'
WorldSize = 1
net_out_list = [2]
target_out_list = [2]
net_scale = 0.7
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
#GPU
useGPU = 0       #(0,1)��1��False ,False

#model
usePretrain = False
StartEpoch = 0
Epoch = 20
Lr = 0.001
Momentum = 0.9
WeightDecay = 4e-6
ModelPath = os.path.join(root,'model_store')
PretrainModelPath = os.path.join(ModelPath,'model_shufflenet.pth')
PrintFreq = 5
Mode = 'Train'#'Train','Val','Test'
