import data
import argparse
from model import EDSR
import tensorflow as tf
parser = argparse.ArgumentParser()
parser.add_argument("--dataset",default="/data1/crz/superresolution/DIV2K/DIV2K_train_HR/train_part")
#parser.add_argument("--dataset",default="/data1/crz/superresolution/General100/train")
parser.add_argument("--imgsize",default=100,type=int)
parser.add_argument("--scale",default=2,type=int)
parser.add_argument("--layers",default=8,type=int)
parser.add_argument("--featuresize",default=64,type=int)
parser.add_argument("--batchsize",default=10,type=int)
parser.add_argument("--savedir",default='saved_models')
parser.add_argument("--iterations",default=10000,type=int)
args = parser.parse_args()
data.load_dataset(args.dataset,args.imgsize)
if args.imgsize % args.scale != 0:
    print(f"Image size {args.imgsize} is not evenly divisible by scale {arg.scale}")
    exit()
tf.device('/gpu:2')
down_size = args.imgsize//args.scale
#network = edsr(2,args.layers,args.featuresize)
network = EDSR(down_size,args.layers,args.featuresize,args.scale)
network.set_data_fn(data.get_batch,(args.batchsize,args.imgsize,down_size),data.get_test_set,(args.imgsize,down_size))
network.train(args.iterations,args.savedir)
