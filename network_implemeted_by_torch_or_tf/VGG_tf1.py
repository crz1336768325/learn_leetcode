import tensorflow as tf 
import numpy  as np

# x:[8,224,224,3]
#第一个参数input：指需要做卷积的输入图像，它要求是一个Tensor，
# 具有[batch, in_height, in_width, in_channels]这样的shape，
# 具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，
# 注意这是一个4维的Tensor，要求类型为float32和float64其中之一

#第二个参数filter：相当于CNN中的卷积核，它要求是一个Tensor，
# 具有[filter_height, filter_width, in_channels, out_channels]这样的shape，
# 具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，
# 要求类型与参数input相同，有一个地方需要注意，第三维in_channels，就是参数input的第四维
def get_weight_variable(shape):
    return tf.Variable(tf.random.normal(shape))

def conv(input,filter,layer_name):
    with tf.variable_scope(layer_name):
        x=tf.nn.conv2d(input,filter,strides=[1,1,1,1],padding='SAME')
        x=tf.nn.relu(x)
        return x
def pooling(input,layer_name):
    with tf.variable_scope(layer_name):
        x=tf.nn.max_pool(input,[1,2,2,1],[1,2,2,1],padding='SAME')
        return x
def vgg(x):
    init=tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        # print(x.eval().shape)
        n,h,w,c=x.eval().shape
        weight=get_weight_variable([3,3,c,64])
        x=conv(x,weight,"c11")
        weight=get_weight_variable([3,3,64,64])
        x=conv(x,weight,"c12")
        x=pooling(x,"max1")
        weight=get_weight_variable([3,3,64,128])
        x=conv(x,weight,"c21")
        weight=get_weight_variable([3,3,128,128])
        x=conv(x,weight,"c22")
        x=pooling(x,"max2")
        weight=get_weight_variable([3,3,128,256])
        x=conv(x,weight,"c31")
        weight=get_weight_variable([3,3,256,256])
        x=conv(x,weight,"c32")
        weight=get_weight_variable([3,3,256,256])
        x=conv(x,weight,"c33")
        x=pooling(x,"max3")
        weight=get_weight_variable([3,3,256,512])
        x=conv(x,weight,"c41")
        weight=get_weight_variable([3,3,512,512])
        x=conv(x,weight,"c42")
        weight=get_weight_variable([3,3,512,512])
        x=conv(x,weight,"c43")
        x=pooling(x,"max4")
        weight=get_weight_variable([3,3,512,512])
        x=conv(x,weight,"c51")
        weight=get_weight_variable([3,3,512,512])
        x=conv(x,weight,"c52")
        weight=get_weight_variable([3,3,512,512])
        x=conv(x,weight,"c53")
        x=pooling(x,"max5")
        # pool5_flatten_dims = int(np.prod(x.get_shape().as_list()[1:]))
        x= tf.reshape(x,[-1,25088])
        print(x.shape)
        x=tf. layers.dense(x,4096)
        x=tf. layers.dense(x,4096)
        x=tf. layers.dense(x,1000)
        # x=conv(x,filter)
        print(x.shape)
        return x

batch_img=tf.Variable(tf.random.normal([8, 224,224,3]))
output=vgg(batch_img)
print("out",output)

