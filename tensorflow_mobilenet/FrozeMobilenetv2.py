import tensorflow as tf

import numpy as np

Detection_or_Classifier = 'classifier'#'detection','classifier'

class Mobilenetv2():
    
    def __init__(self,num_classes,learning_rate=0.045):
        self.num_classes = num_classes
        
        self.ssd_default_box_size=[6,6,6,6,6,6]
        self.learning_rate = learning_rate
        
        self.loss = Loss()
        
        self.__build()
    
    def __build(self):
        self.norm = 'bias'#batch_norm,bias
        self.activate = 'relu'#relu,relu6
        self.BlockInfo = {#scale /8
                          '1':[1,16,1,1],
                          '2':[6,24,1,2],
                          '3':[6,24,1,1],#ssd
                          '4':[6,32,1,2],
                          '5':[6,32,2,1],#ssd
                          '6':[6,64,1,2],
                          '7':[6,64,3,1],#ssd
                          '8':[6,96,3,1],#ssd
                          '9':[6,160,1,2],
                          '10':[6,160,2,1],#ssd
                          '11':[6,320,1,1],#ssd
                          '12':[1,1280,1,1]}
    
        self.__init_global_epoch()
        self.__init_global_step()
        self.__init_input()

        with tf.variable_scope('zsc_feature'):
            #none,none,none,3
            x = PrimaryConv('PrimaryConv',self.input_image,32,self.norm,self.activate)
            #skip_0 = x
            #none,none/2,none/2,32
            
            index = '1'
            x = Mobilenetv2Block('Mobilenetv2Block_'+index,x,
                                 self.BlockInfo[index][0],self.BlockInfo[index][1],self.BlockInfo[index][2],self.BlockInfo[index][3],
                                 self.norm,self.activate)
            #skip_1 = x
            #none,none/2,none/2,16
            
            index = '2'
            x = Mobilenetv2Block('Mobilenetv2Block_'+index,x,
                                 self.BlockInfo[index][0],self.BlockInfo[index][1],self.BlockInfo[index][2],self.BlockInfo[index][3],
                                 self.norm,self.activate)
            #skip_2 = x
            #none,none/4,none/4,24
            
            index = '3'
            x = Mobilenetv2Block('Mobilenetv2Block_'+index,x,
                                 self.BlockInfo[index][0],self.BlockInfo[index][1],self.BlockInfo[index][2],self.BlockInfo[index][3],
                                 self.norm,self.activate)
            #skip_3 = x
            #none,none/4,none/4,24
            
            index = '4'
            x = Mobilenetv2Block('Mobilenetv2Block_'+index,x,
                                 self.BlockInfo[index][0],self.BlockInfo[index][1],self.BlockInfo[index][2],self.BlockInfo[index][3],
                                 self.norm,self.activate)
            #skip_4 = x
            #none,none/8,none/8,32
            
            index = '5'
            x = Mobilenetv2Block('Mobilenetv2Block_'+index,x,
                                 self.BlockInfo[index][0],self.BlockInfo[index][1],self.BlockInfo[index][2],self.BlockInfo[index][3],
                                 self.norm,self.activate)
            #skip_5 = x
            #none,none/8,none/8,32
            
            index = '6'
            x = Mobilenetv2Block('Mobilenetv2Block_'+index,x,
                                 self.BlockInfo[index][0],self.BlockInfo[index][1],self.BlockInfo[index][2],self.BlockInfo[index][3],
                                 self.norm,self.activate)
            #skip_6 = x
            #none,none/16,none/16,64
            
            index = '7'
            x = Mobilenetv2Block('Mobilenetv2Block_'+index,x,
                                 self.BlockInfo[index][0],self.BlockInfo[index][1],self.BlockInfo[index][2],self.BlockInfo[index][3],
                                 self.norm,self.activate)
            #skip_7 = x
            #none,none/16,none/16,64
            
            index = '8'
            x = Mobilenetv2Block('Mobilenetv2Block_'+index,x,
                                 self.BlockInfo[index][0],self.BlockInfo[index][1],self.BlockInfo[index][2],self.BlockInfo[index][3],
                                 self.norm,self.activate)
            #skip_8 = x
            #none,none/16,none/16,96
            
            index = '9'
            x = Mobilenetv2Block('Mobilenetv2Block_'+index,x,
                                 self.BlockInfo[index][0],self.BlockInfo[index][1],self.BlockInfo[index][2],self.BlockInfo[index][3],
                                 self.norm,self.activate)
            #skip_9 = x
            #none,none/16,none/16,160
            
            index = '10'
            x = Mobilenetv2Block('Mobilenetv2Block_'+index,x,
                                 self.BlockInfo[index][0],self.BlockInfo[index][1],self.BlockInfo[index][2],self.BlockInfo[index][3],
                                 self.norm,self.activate)
            #skip_10 = x
            #none,none/32,none/32,160
            
            index = '11'
            x = Mobilenetv2Block('Mobilenetv2Block_'+index,x,
                                 self.BlockInfo[index][0],self.BlockInfo[index][1],self.BlockInfo[index][2],self.BlockInfo[index][3],
                                 self.norm,self.activate)
            #skip_11 = x
            #none,none/32,none/32,320
            
        if Detection_or_Classifier=='classifier':
            with tf.variable_scope('zsc_classifier'):
                x = _conv_block('FinalConv',x,self.num_classes,1,1,'SAME',self.norm,self.activate)
                x = tf.nn.avg_pool(x,[1,7,7,1],[1,1,1,1],'VALID')
                self.classifier_logits = tf.reshape(
                                                    x,
                                                    [-1,self.num_classes]
                                                   )
        elif Detection_or_Classifier=='detection':
            pass
        
        self.__init__output()
        
        if Detection_or_Classifier=='classifier':
            pass
        elif Detection_or_Classifier=='detection':
            pass
    def __init__output(self):
        with tf.variable_scope('output'):
            regularzation_loss = self.loss.regularzation_loss()
            
            if Detection_or_Classifier=='classifier':
                self.all_loss = self.loss.sparse_softmax_loss(self.classifier_logits,self.y)
                self.all_loss += regularzation_loss
                
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    learning_rate = tf.train.exponential_decay(self.learning_rate,global_step=self.global_epoch_tensor,decay_steps=5,decay_rate=0.9995)
                    self.optimizer = tf.train.AdamOptimizer(learning_rate)
                    self.train_op = self.optimizer.minimize(self.all_loss,global_step=self.global_epoch_tensor)
                
                self.y_out_softmax = tf.nn.softmax(self.classifier_logits,name='zsc_output')
                
                self.y_out_argmax = tf.cast(tf.argmax(self.y_out_softmax, axis=-1),tf.int32)
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y, self.y_out_argmax), tf.float32))
                
                self.accuracy_top_5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.y_out_softmax,self.y,5),tf.float32))

            elif Detection_or_Classifier=='detection':
                pass
        
    def __init_input(self):
        if Detection_or_Classifier=='classifier':
            with tf.variable_scope('input'):
                self.input_image = tf.placeholder(tf.float32,[None,224,224,3],name='zsc_input')#训练、测试用
                self.y = tf.placeholder(tf.int32, [None],name='zsc_input_target')#训练、测试用
        elif Detection_or_Classifier=='detection':
            pass
    def __init_global_epoch(self):
        with tf.variable_scope('global_epoch'):
            self.global_epoch_tensor = tf.Variable(-1, trainable=False, name='global_epoch')
            self.global_epoch_input = tf.placeholder('int32', None, name='global_epoch_input')
            self.global_epoch_assign_op = self.global_epoch_tensor.assign(self.global_epoch_input)
    def __init_global_step(self):
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            self.global_step_input = tf.placeholder('int32', None, name='global_step_input')
            self.global_step_assign_op = self.global_step_tensor.assign(self.global_step_input)

from tensorflow.python.ops import array_ops
##LOSS
class Loss():
    def __init__(self):
        pass
    #regularzation loss
    def regularzation_loss(self):
        return sum(tf.get_collection("regularzation_loss"))
    
    #sparse softmax loss
    def sparse_softmax_loss(self, logits, labels):
        labels = tf.to_int32(labels)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
            logits=logits, name='xentropy')
        return tf.reduce_mean(cross_entropy, name='xentropy_mean')
    
    #focal loss
    def focal_loss(self, prediction_tensor, target_tensor, alpha=0.25, gamma=2):
        #prediction_tensor [batch,num_anchors,num_classes]
        #target_tensor     [batch,num_anchors,num_classes]
        sigmoid_p = tf.nn.sigmoid(prediction_tensor)
        zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
        
        pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)
        
        neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
        per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                              - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
        return tf.reduce_sum(per_entry_cross_ent,2)
    
    #smooth_L1
    def smooth_L1(self, x):
        return tf.where(tf.less_equal(tf.abs(x),1.0), tf.multiply(0.5, tf.pow(x, 2.0)), tf.subtract(tf.abs(x), 0.5))
    
    def ssd_loss(self, num_classes, pred, ground_truth, positive, negative, use_focal_loss=True):
        #pred [batch,num_anchors,num_classes+4]
        #ground_truth [batch,num_anchors,1+4]
        #positive [batch,num_anchors]
        #negative [batch,num_anchors]
        ground_truth_count = tf.add(positive,negative)
        if use_focal_loss:
            loss_class = self.focal_loss(pred[:,:,1:-4],tf.one_hot(tf.cast(ground_truth[:,:,0],tf.int32),num_classes))
        else:
            loss_class = self.sparse_softmax_loss(pred[:,:,1:-4],tf.cast(ground_truth[:,:,0],tf.int32))
        self.loss_location = tf.truediv(
                                        tf.reduce_sum(
                                                      tf.multiply(
                                                                  tf.reduce_sum(
                                                                                self.smooth_L1(
                                                                                               tf.subtract(
                                                                                                           ground_truth[:,:,1:], 
                                                                                                           pred[:,:,-4:]
                                                                                                           )
                                                                                               ),
                                                                                2
                                                                                ), 
                                                                  positive
                                                                  ),
                                                      1), 
                                        tf.reduce_sum(positive,1)
                                        )
        self.loss_class = tf.truediv(
                                     tf.reduce_sum(
                                                   tf.multiply(
                                                               loss_class,
                                                               ground_truth_count),
                                                   1), 
                                     tf.reduce_sum(ground_truth_count,1)
                                     )
        self.loss_confidence = tf.truediv(
                                        tf.reduce_sum(
                                                      tf.multiply(
                                                                 self.smooth_L1(
                                                                                tf.subtract(
                                                                                            positive, 
                                                                                            pred[:,:,0]
                                                                                            )
                                                                                ),
                                                                  positive
                                                                  ),
                                                      1), 
                                        tf.reduce_sum(positive,1)
                                        )
        self.loss_unconfidence = tf.truediv(
                                            tf.reduce_sum(
                                                          tf.multiply(
                                                                     self.smooth_L1(
                                                                                    tf.subtract(
                                                                                                negative, 
                                                                                                pred[:,:,0]
                                                                                                )
                                                                                    ),
                                                                      negative
                                                                      ),
                                                          1), 
                                            tf.reduce_sum(ground_truth_count,1)
                                            )
        return self.loss_class,self.loss_location,self.loss_confidence,self.loss_unconfidence
################################################################################################################
################################################################################################################
################################################################################################################
##Mobilenetv2Block
def Mobilenetv2Block(name,x,ratio=6,num_filters=16,repeat=1,stride=1,norm='batch_norm',activate='selu'):
    with tf.variable_scope(name):
        if stride==1:
            for i in range(repeat):
                x = DepthwiseBlock('depthwiseblock_{}'.format(i),x,ratio,num_filters,norm,activate)
        else:
            x = Transition('Transition',x,ratio,num_filters,norm,activate)
        
        return x
def DepthwiseBlock(name,x,ratio,num_filters=16,norm='batch_norm',activate='selu'):
    with tf.variable_scope(name):
        input = x
        
        x = _conv_block('conv_0',x,ratio*num_filters,1,1,'SAME',norm,activate)
        x = _depthwise_conv2d('depthwise',x,1,3,1,'SAME',norm,activate)
        x = _conv_block('conv_1',x,num_filters,1,1,'SAME',norm,None)
        
        if input.get_shape().as_list()[-1]==x.get_shape().as_list()[-1]:
            pass
        else:
            input = _conv_block('conv_2',input,num_filters,1,1,'SAME',norm,activate)
        x += input
        
        return x
def Transition(name,x,ratio=6,num_filters=16,norm='batch_norm',activate='selu'):
    with tf.variable_scope(name):
        x = _conv_block('conv_0',x,ratio*num_filters,1,1,'SAME',norm,activate)
        x = _depthwise_conv2d('depthwise',x,1,3,2,'SAME',norm,activate)
        x = _conv_block('conv_1',x,num_filters,1,1,'SAME',norm,None)
        return x
##primary_conv
def PrimaryConv(name,x,num_filters=32,norm='batch_norm',activate='selu'):
    with tf.variable_scope(name):
        #none,none,none,3
        x = _conv_block('conv_0',x,num_filters,3,2,'SAME',norm,activate)#none,none/2,none/2,num_filters
        return x
##_conv_block
def _conv_block(name,x,num_filters=16,kernel_size=3,stride=2,padding='SAME',norm='batch_norm',activate='selu'):
    with tf.variable_scope(name):
        w = GetWeight('weight',[kernel_size,kernel_size,x.shape.as_list()[-1],num_filters])
        x = tf.nn.conv2d(x,w,[1,stride,stride,1],padding=padding,name='conv')
        
        if norm=='batch_norm':
            x = bn(x, name='batchnorm')
        elif norm=='bias':
            b = tf.get_variable('bias',[num_filters],tf.float32,initializer=tf.constant_initializer(0.001))
            x += b
        else:
            pass
        if activate=='relu':
            x = tf.nn.relu(x,name='relu')
        elif activate=='relu6':
            x = relu6(x,name='relu6')
        else:
            pass

        return x
##_depthwise_conv2d
def _depthwise_conv2d(name,x,scale=1,kernel_size=3,stride=1,padding='SAME',norm='batch_norm',activate='selu'):
    with tf.variable_scope(name) as scope:
        w = GetWeight('weight',[kernel_size,kernel_size,x.shape.as_list()[-1],scale])
        x = tf.nn.depthwise_conv2d(x, w, [1,stride,stride,1], padding)
        
        if norm=='batch_norm':
            x = bn(x, name='batchnorm')
        elif norm=='bias':
            b = tf.get_variable('bias',[int(x.shape.as_list()[-1])],tf.float32,initializer=tf.constant_initializer(0.001))
            x += b
        else:
            pass
        if activate=='relu':
            x = tf.nn.relu(x,name='relu')
        elif activate=='relu6':
            x = relu6(x,name='relu6')
        else:
            pass
        return x

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import math_ops
##batch_norm
def bn(x, name='batchnorm'):
    with tf.variable_scope(name):
        epsilon = 1e-3
        
        size = int(x.shape.as_list()[-1])
        
        beta = tf.get_variable('beta', [size], initializer=tf.zeros_initializer())
        scale = tf.get_variable('scale', [size], initializer=tf.ones_initializer())

        moving_mean = tf.get_variable('mean', [size], initializer=tf.zeros_initializer(), trainable=False)
        moving_variance = tf.get_variable('variance', [size], initializer=tf.ones_initializer(), trainable=False)

        inv = math_ops.rsqrt(moving_variance + epsilon)
        inv *= scale 
        return x * inv + (beta - moving_mean * inv)
##weight variable
def GetWeight(name,shape,weights_decay = 0.00004):
    with tf.variable_scope(name):
        #w = tf.get_variable('weight',shape,tf.float32,initializer=VarianceScaling())
        w = tf.get_variable('weight',shape,tf.float32,initializer=glorot_uniform_initializer())
        weight_decay = tf.multiply(tf.nn.l2_loss(w), weights_decay, name='weight_loss')
        tf.add_to_collection('regularzation_loss', weight_decay)
        return w
##initializer
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops
import math
def glorot_uniform_initializer(seed=None, dtype=dtypes.float32):
    return VarianceScaling(scale=1.0,
                          mode="fan_avg",
                          distribution="uniform",
                          seed=seed,
                          dtype=dtype)
def glorot_normal_initializer(seed=None, dtype=dtypes.float32):
    return VarianceScaling(scale=1.0,
                          mode="fan_avg",
                          distribution="normal",
                          seed=seed,
                          dtype=dtype)
def _compute_fans(shape):
    if len(shape) < 1:
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        receptive_field_size = 1.
        for dim in shape[:-2]:
            receptive_field_size *= dim
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
    return fan_in, fan_out
class VarianceScaling():
    def __init__(self, scale=1.0,
                 mode="fan_in",
                 distribution="normal",
                 seed=None,
                 dtype=dtypes.float32):
      if scale <= 0.:
          raise ValueError("`scale` must be positive float.")
      if mode not in {"fan_in", "fan_out", "fan_avg"}:
          raise ValueError("Invalid `mode` argument:", mode)
      distribution = distribution.lower()
      if distribution not in {"normal", "uniform"}:
          raise ValueError("Invalid `distribution` argument:", distribution)
      self.scale = scale
      self.mode = mode
      self.distribution = distribution
      self.seed = seed
      self.dtype = dtype

    def __call__(self, shape, dtype=None, partition_info=None):
      if dtype is None:
          dtype = self.dtype
      scale = self.scale
      scale_shape = shape
      if partition_info is not None:
          scale_shape = partition_info.full_shape
      fan_in, fan_out = _compute_fans(scale_shape)
      if self.mode == "fan_in":
          scale /= max(1., fan_in)
      elif self.mode == "fan_out":
          scale /= max(1., fan_out)
      else:
          scale /= max(1., (fan_in + fan_out) / 2.)
      if self.distribution == "normal":
          stddev = math.sqrt(scale)
          return random_ops.truncated_normal(shape, 0.0, stddev,
                                             dtype, seed=self.seed)
      else:
          limit = math.sqrt(3.0 * scale)
          return random_ops.random_uniform(shape, -limit, limit,
                                           dtype, seed=self.seed)
##relu6
def relu6(x, name='relu6'):
    with tf.variable_scope(name):
        x = tf.nn.relu(x,name='relu')
        x = math_ops.maximum(x,6.0,name='relu6')
        return x
################################################################################################################
################################################################################################################
################################################################################################################

if __name__=='__main__':
    import time
    from functools import reduce
    from operator import mul
    
    import numpy as np

    def get_num_params():
        num_params = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            num_params += reduce(mul, [dim.value for dim in shape], 1)
        return num_params

    model = Mobilenetv2(num_classes=18)
    
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        
        num_params = get_num_params()
        print('all params:{}'.format(num_params))
        
        '''
        feed_dict={model.input_image:np.random.randn(1,224,224,3),
                   model.ground_truth:np.concatenate([np.ones((1,26460,1)),np.random.randn(1,26460,4)],axis=-1),
                   model.positive:np.ones((2,26460)),
                   model.negative:np.zeros((2,26460)),
                   model.original_wh:[[256,256]]}
        '''
        feed_dict={model.input_image: np.random.randn(1,224,224,3),
                   model.y: [1]}
        
        start = time.time()
        out = sess.run(model.classifier_logits,feed_dict=feed_dict)
        print('Spend Time:{}'.format(time.time()-start))
        
        print(out.shape)
