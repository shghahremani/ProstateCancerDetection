from keras.layers import  Conv2D
from keras.layers.core import  Activation
from keras import backend as K
from keras.layers import  Conv2D , Lambda, Multiply, Reshape, Dense, Flatten
from kwae.tools.metrics import f1 as f1_score
from keras import regularizers
import tensorflow as tf
from keras.layers.merge import concatenate


def getindice( **kwargs):
    def layer(x):
        x= x[:,:,:,1]

        return tf.expand_dims(x, axis=3)
    return Lambda(layer, **kwargs)

def woundchannels( **kwargs):
    def layer(x):
        x= x[:,:,:,1:9]

        return x
    return Lambda(layer, **kwargs)

#def build(config,inp,inp_indice,nc,w, h):
def build(WounSkin, inp):
    if K.image_dim_ordering() == 'th':
        inputs = inp
        axis = 1
    else:
        inputs =inp
        axis = 3

    input=inp
    #OUTPUT_MASK_CHANNELS = nc
    #indice = generate_wound_indice(config=config, nc=nc, w=w, h=h)(inp_indice)
    WoundSkin = getindice()(WounSkin)
    print('WoundSkin is {0}'.format(WoundSkin.shape))
    wound_chanels = woundchannels()(input)
    input_signal_shape = wound_chanels.shape
    print('wound_chanels is {0}'.format(wound_chanels.shape))
    upsampler_conc = concatenate([WoundSkin, WoundSkin], axis=axis)
    for i in range(input_signal_shape[3] - 2):
        upsampler_conc = concatenate([upsampler_conc, WoundSkin], axis=axis)
    wound_indices = Multiply()([upsampler_conc, wound_chanels])
    wound_indices = Flatten()(wound_indices)
    wound_labels = Dense(units=7, activation='sigmoid', kernel_regularizer=regularizers.l1(0.01))(wound_indices)
    print('wound_labels sig is {0}'.format(wound_labels.shape))
    return wound_labels