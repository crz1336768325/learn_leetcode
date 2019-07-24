from model import EDSR
import scipy.misc
import argparse
import data
import os
import cv2
import time
import tensorflow as tf
import numpy as np
from os import listdir
parser = argparse.ArgumentParser()
parser.add_argument("--dataset",default="/data1/crz/superresolution/EDSR/EDSR-Tensorflow-master/testdata")
parser.add_argument("--imgsize",default=100,type=int)
parser.add_argument("--scale",default=2,type=int)
parser.add_argument("--layers",default=3,type=int)
parser.add_argument("--featuresize",default=128,type=int)
parser.add_argument("--batchsize",default=10,type=int)
parser.add_argument("--savedir",default="saved_models")
parser.add_argument("--iterations",default=1000,type=int)
parser.add_argument("--numimgs",default=5,type=int)
parser.add_argument("--outdir",default="out")
parser.add_argument("--image")
args = parser.parse_args()
if not os.path.exists(args.outdir):
	os.mkdir(args.outdir)
down_size = args.imgsize//args.scale
#network = EDSR(down_size,args.layers,args.featuresize,scale=args.scale)
#network.resume_pb(args.savedir)
savedir='/data1/crz/superresolution/EDSR/ooo_modified/EDSR-Tensorflow-master/saved_models'
'''
x = cv2.imread("/data1/crz/superresolution/EDSR/EDSR-Tensorflow-master/testdata/0001x2.png")
outputs = network.predict(x)
'''

from tensorflow.python.platform import gfile
 
sess = tf.Session()
with gfile.FastGFile(savedir+'/crz.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='') # 导入计算图
print("sddsdsdsddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",sess.graph)
sess.run(tf.global_variables_initializer())
op = sess.graph.get_tensor_by_name('crzout:0')
path = '/data1/crz/superresolution/BIc_interp_testimage/test/'
outpath = '/data1/crz/superresolution/BIc_interp_testimage/edsr_test/'
images_name = [x for x in listdir(path) ]
for image_name in images_name:
    x = cv2.imread(path+image_name)
    print("inimage",image_name)
    #x=cv2.resize(x,(50,50))
    x=np.asarray(x)
    inputs = x
    x=np.expand_dims(x,axis=0)
    input_x = sess.graph.get_tensor_by_name('crzin:0')
    start=time.time()
    outputs=sess.run(op,feed_dict={input_x:x})
    #crz=np.asarray(outputs)
    crz=np.squeeze(outputs)


    #crz=(crz-crz.min())/(crz.max()-crz.min())*255
    #np.clip(crz,0.0,255.0)
    #print("crz.min",crz.min())
    #print("outputs",crz)
    #print("outputs",np.shape(crz))
    outimage=outpath+image_name
    print("outimage",outimage)
    #cv2.imwrite(outimage,inputs)
    cv2.imwrite(outimage,crz)
if args.image:
	scipy.misc.imsave(args.outdir+"/input_"+args.image,inputs)
	scipy.misc.imsave(args.outdir+"/output_"+args.image,crz)



