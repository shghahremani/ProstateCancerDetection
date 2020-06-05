from keras.layers import  Conv2D
from keras.layers.core import  Activation
from keras import backend as K
import tensorflow as tf
from keras.layers import Lambda
from keras.layers.merge import concatenate

def Convertothreeclass( **kwargs):
    def layer(x):
        x= x[:,:,:,1:9]

        return tf.expand_dims(tf.reduce_max(x,axis=3), axis=3)
    return Lambda(layer, **kwargs)

def firstchannel( **kwargs):
    def layer(x):
        x= x[:,:,:,0]

        return tf.expand_dims(x, axis=3)
    return Lambda(layer, **kwargs)

def lastchannel( **kwargs):
    def layer(x):
        x= x[:,:,:,8]

        return tf.expand_dims(x, axis=3)
    return Lambda(layer, **kwargs)


def build(inp, nc):
    if K.image_dim_ordering() == 'th':
        inputs = inp
        axis = 1
    else:
        inputs =inp
        axis = 3

    input=inp
    OUTPUT_MASK_CHANNELS = nc
    wound_feature = Convertothreeclass()(input)
    #print('wound_feature is {0}'.format(wound_feature.shape))
    Firstchannel = firstchannel()(input)
    #print('wound_feature is {0}'.format(Firstchannel.shape))
    Lastchannel = lastchannel()(input)
   # print('wound_feature is {0}'.format(Lastchannel.shape))
    conv_final = concatenate([Firstchannel, wound_feature,Lastchannel], axis=axis)
    # print('conv_final sig is {0}'.format(conv_final.shape))
    return conv_final