from model import EDSR
import scipy.misc
import argparse
import data
import os
import cv2
import time
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework import graph_util
parser = argparse.ArgumentParser()
parser.add_argument("--dataset",default="/data1/crz/superresolution/EDSR/EDSR-Tensorflow-master/testdata")
parser.add_argument("--imgsize",default=100,type=int)
parser.add_argument("--scale",default=2,type=int)
parser.add_argument("--layers",default=8,type=int)
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
savedir='/data1/crz/superresolution/EDSR/ooo_modified/EDSR-Tensorflow-master/saved_models'
network = EDSR(down_size,args.layers,args.featuresize,scale=args.scale)
network.resume_pb(args.savedir)
#network.resume(args.savedir)
'''
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	#tf.train.write_graph(sess.graph_def,"saved_models/","EDSR.pb",as_text=False)
	output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), ['crzout'])
	#tf.train.write_graph(output_graph_def, "saved_models/", "crz"+'.pb', as_text=False)
	with open("saved_models/crz"+'.pb',"wb") as f:
		f.write(output_graph_def.SerializeToString()) 


'''

'''
converter = tf.lite.TFLiteConverter.from_saved_model(savedir)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	tf.train.write_graph(sess.graph_def,"saved_models/","EDSR.pb",as_text=False)
	tf.train.Saver.save(sess,"saved_models/EDSR.ckpt")

'''

