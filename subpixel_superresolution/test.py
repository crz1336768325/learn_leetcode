from model import EDSR
import scipy.misc
import argparse
import data
import os
import cv2
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("--dataset",default="data/General-100")
parser.add_argument("--imgsize",default=32,type=int)
parser.add_argument("--scale",default=2,type=int)
parser.add_argument("--layers",default=8,type=int)
parser.add_argument("--featuresize",default=128,type=int)
parser.add_argument("--batchsize",default=5,type=int)
parser.add_argument("--savedir",default="saved_models")
parser.add_argument("--iterations",default=10000,type=int)
parser.add_argument("--numimgs",default=5,type=int)
parser.add_argument("--outdir",default="out")
parser.add_argument("--image")
args = parser.parse_args()
if not os.path.exists(args.outdir):
	os.mkdir(args.outdir)
down_size = args.imgsize//args.scale
savedir='/data1/crz/superresolution/subpixel/subpixel_tensor_crz/EDSR-Tensorflow-master/saved_models'
network = EDSR(down_size,args.layers,args.featuresize,scale=args.scale)
network.resume(savedir)

x = cv2.imread("/data1/crz/superresolution/EDSR/ooo_modified/EDSR-Tensorflow-master/testdata/0021x2.png")
x=np.asarray(x)
inputs = x
crz=np.expand_dims(x,0)
outputs = network.predict_simple(crz)
outputs=np.squeeze(outputs)
outputs=(outputs-outputs.min())/(outputs.max()-outputs.min())*255
outputs=np.clip(outputs,0.0,255.0)
print("outputs",outputs)
print("outputs",outputs.shape)
cv2.imwrite("/data1/crz/superresolution/subpixel/subpixel_tensor_crz/EDSR-Tensorflow-master/input.jpg",inputs)
cv2.imwrite("/data1/crz/superresolution/subpixel/subpixel_tensor_crz/EDSR-Tensorflow-master/output.jpg",outputs)
if args.image:
	scipy.misc.imsave(args.outdir+"/input_"+args.image,inputs)
	scipy.misc.imsave(args.outdir+"/output_"+args.image,outputs)
