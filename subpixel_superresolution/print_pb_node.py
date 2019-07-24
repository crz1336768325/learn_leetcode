import tensorflow as tf
import os
from tensorflow.python.platform import gfile
model_name="/data1/crz/superresolution/EDSR/ooo_modified/EDSR-Tensorflow-master/saved_models/crz.pb"
def create_graph():
    with gfile.FastGFile(model_name,'rb') as f:
        graph_def=tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def,name='')
create_graph()
tensor_name_list=[tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
#for tensor_name in tensor_name_list:
#    print(tensor_name,'\n')
for op in tf.get_default_graph().get_operations():
    print(op.name,'\n')
    print(op.values(),'\n')
