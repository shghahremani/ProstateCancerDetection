# coding=utf-8
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers.core import Activation
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import tensorflow as tf
from keras.layers.merge import concatenate
from keras.layers import Lambda, Add, Multiply

def bottleneck(encoder, output, upsample=False, reverse_module=False):
    internal = output // 4

    x = Conv2D(internal, (1, 1), use_bias=False)(encoder)
    x = BatchNormalization(momentum=0.1)(x)
    x = Activation('relu')(x)
    if not upsample:
        x = Conv2D(internal, (3, 3), padding='same', use_bias=True)(x)
    else:
        x = Conv2DTranspose(filters=internal, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization(momentum=0.1)(x)
    x = Activation('relu')(x)

    x = Conv2D(output, (1, 1), padding='same', use_bias=False)(x)

    other = encoder
    if encoder.get_shape()[-1] != output or upsample:
        other = Conv2D(output, (1, 1), padding='same', use_bias=False)(other)
        other = BatchNormalization(momentum=0.1)(other)
        if upsample and reverse_module is not False:
            other = UpSampling2D(size=(2, 2))(other)
        
    if upsample and reverse_module is False:
        decoder = x
    else:
        x = BatchNormalization(momentum=0.1)(x)
        decoder = add([x, other])
        decoder = Activation('relu')(decoder)

    return decoder
def grid_attention(input_signal,gating_signal,inter_channels):
    if K.image_dim_ordering() == 'th':
        axis = 1
    else:
        axis = 3
    theta_x = Conv2D(inter_channels,
                     kernel_size=(1,1), strides=(2,2), padding='valid',
                               use_bias=False)(input_signal)
    theta_x_size = theta_x.shape
    phi_g = Conv2D(inter_channels,
                             kernel_size=(1,1), padding='valid', use_bias=True)(gating_signal)

    cancat =Add()([phi_g, theta_x])
    f =  Activation('relu')(cancat)
    psi = Conv2D(filters=1, kernel_size=(1,1), padding='valid', use_bias=True)(f)
    sigm_psi_f = Activation('sigmoid')(psi)

    input_signal_shape=input_signal.shape
    # upsample the attentions and multiply
    #print('sigm_psi_f is {0}'.format(sigm_psi_f))
    scale_factor=2
    upsampler = UpSampling2DBilinear(stride=scale_factor)(sigm_psi_f)
    #print('upsampled is {0}'.format(upsampler.shape))
    upsampler_conc = concatenate([upsampler, upsampler], axis = axis)
    for i in range(input_signal_shape[3]-2):
        upsampler_conc = concatenate([upsampler_conc, upsampler], axis=axis)
    #print('upsampler_conc  is {0}'.format(upsampler_conc .shape))
    y = Multiply()([input_signal,upsampler_conc])
    #final_dimention=input_signal_shape[3]
    W_y =Conv2D (inter_channels, kernel_size=(1,1), strides=(1,1), padding='valid')(y)
    w_y = BatchNormalization(axis=axis)(W_y)

    return w_y
def UpSampling2DBilinear(stride, **kwargs):
    def layer(x):
        input_shape = K.int_shape(x)
        output_shape = (stride * input_shape[1], stride * input_shape[2])
        return tf.image.resize_bilinear(x, output_shape, align_corners=True)
    return Lambda(layer, **kwargs)



def build(enet_gate,enet_att_input, nc):


    g_conv = grid_attention(enet_att_input, enet_gate, inter_channels=64)
    up_14 = concatenate([UpSampling2D(size=(2, 2))(enet_gate), g_conv], axis=3)
    enet = bottleneck(up_14, 64, reverse_module=True)  # bottleneck 4.0
    enet = bottleneck(enet, 64)  # bottleneck 4.1
    enet = bottleneck(enet, 64)  # bottleneck 4.2
    enet = bottleneck(enet, 16, upsample=True, reverse_module=True)  # bottleneck 5.0
    enet = bottleneck(enet, 16)  # bottleneck 5.1

    enet = Conv2DTranspose(filters=nc, kernel_size=(2, 2), strides=(2, 2), padding='same')(enet)
    return enet
