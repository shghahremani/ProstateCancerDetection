import numpy as np
from keras.layers import  Conv2D, MaxPooling2D, UpSampling2D, Add, Multiply,Subtract
from keras.layers.normalization import BatchNormalization
from keras.layers.core import SpatialDropout2D, Activation
from keras import backend as K
import tensorflow as tf
from keras.layers.merge import concatenate
from keras import Sequential
from keras.layers import Lambda

# Number of output masks (1 in case you predict only one type of objects)
INPUT_CHANNELS = 3
# Pretrained weights

def double_conv_layer(x, size,layer_name, dropout=0.0, batch_norm=True):
    if K.image_dim_ordering() == 'th':
        axis = 1
    else:
        axis = 3
    conv = Conv2D(size, (3, 3), padding='same',name=layer_name)(x)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(size, (3, 3), padding='same')(conv)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    if dropout > 0:
        conv = SpatialDropout2D(dropout)(conv)
    return conv
def UpSampling2DBilinear(stride, **kwargs):
    def layer(x):
        input_shape = K.int_shape(x)
        output_shape = (stride * input_shape[1], stride * input_shape[2])
        return tf.image.resize_bilinear(x, output_shape, align_corners=True)
    return Lambda(layer, **kwargs)
def DownSampling2DBilinear(stride, **kwargs):
    def layer(x):
        input_shape = K.int_shape(x)
        output_shape = (np.int( input_shape[1]/stride), np.int(input_shape[2]/stride) )
        return tf.image.resize_bilinear(x, output_shape, align_corners=True)
    return Lambda(layer, **kwargs)
def reverse_sign( **kwargs):
    def layer(x):

        return tf.negative(x)
    return Lambda(layer, **kwargs)


def dsv(x, size,scale_factor, dropout=0.0, batch_norm=True):
    if K.image_dim_ordering() == 'th':
        axis = 1
    else:
        axis = 3
    conv = Conv2D(size, (3, 3), padding='same')(x)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(size, (3, 3), padding='same')(conv)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    if dropout > 0:
        conv = SpatialDropout2D(dropout)(conv)

    upsampler = UpSampling2DBilinear(stride=scale_factor)(conv)

    return upsampler
def gating(x, size, dropout=0.0, batch_norm=True):
    if K.image_dim_ordering() == 'th':
        axis = 1
    else:
        axis = 3
    conv = Conv2D(size, (1, 1), padding='valid')(x)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)

    if dropout > 0:
        conv = SpatialDropout2D(dropout)(conv)
    return conv
def neg_grid_attention(input_signal_1=None,input_signal_2=None,input_signal_3=None,input_signal_4=None,scale_factor=None,inter_channels=None):
    if K.image_dim_ordering() == 'th':
        axis = 1
    else:
        axis = 3
    if input_signal_1!=None:
        Down_1 = DownSampling2DBilinear(stride=scale_factor)(input_signal_1)
        Down_1=reverse_sign()(Down_1)
        input_signal_1=Conv2D(inter_channels,kernel_size=(1, 1), padding='valid', use_bias=True)(Down_1)
        cancat=input_signal_1
    if input_signal_2!=None:
        Down_2 = DownSampling2DBilinear(stride=scale_factor/2)(input_signal_2)
        Down_2 = reverse_sign()(Down_2)
        input_signal_2=Conv2D(inter_channels,kernel_size=(1, 1), padding='valid', use_bias=True)(Down_2)
        cancat = Add()([input_signal_1, input_signal_2])
    if input_signal_3!=None:
        Down_3 = DownSampling2DBilinear(stride=scale_factor/4)(input_signal_3)
        Down_3 = reverse_sign()(Down_3)
        input_signal_3=Conv2D(inter_channels,kernel_size=(1, 1), padding='valid', use_bias=True)(Down_3)
        cancat = Add()([input_signal_1, input_signal_2, input_signal_3])
    if input_signal_4!=None:
        Down_4 = DownSampling2DBilinear(stride=scale_factor / 8)(input_signal_4)
        Down_4 =reverse_sign()(Down_4)
        input_signal_4 = Conv2D(inter_channels, kernel_size=(1, 1), padding='valid', use_bias=True)(Down_4)
        cancat = Add()([input_signal_1,input_signal_2,input_signal_3,input_signal_4])

    f =  Activation('relu') (cancat)
    psi = Conv2D(filters=1, kernel_size=(1,1), padding='valid', use_bias=True)(f)
    sigm_psi_f = Activation('sigmoid')(psi)
    return  sigm_psi_f

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


    return upsampler
def grid_attention_mult(input_signal,improved_atten_coeff,inter_channels):
    if K.image_dim_ordering() == 'th':
        axis = 1
    else:
        axis = 3

    improved_atten_coeff_conc = concatenate([improved_atten_coeff, improved_atten_coeff], axis = axis)
    input_signal_shape = input_signal.shape
    for i in range(input_signal_shape[3]-2):
        improved_atten_coeff_conc = concatenate([improved_atten_coeff_conc, improved_atten_coeff], axis=axis)
    #print('upsampler_conc  is {0}'.format(upsampler_conc .shape))
    y = Multiply()([input_signal,improved_atten_coeff_conc])
    #final_dimention=input_signal_shape[3]
    W_y =Conv2D (inter_channels, kernel_size=(1,1), strides=(1,1), padding='valid')(y)
    w_y = BatchNormalization(axis=axis)(W_y)

    return w_y


def build(inp, nc):
    if K.image_dim_ordering() == 'th':
        inputs = inp
        axis = 1
    else:
        inputs =inp
        axis = 3
    filters = 32
    OUTPUT_MASK_CHANNELS = nc

    dropout_val = 0.2
    layer_name = 'first_conv_layer'
    conv_224 = double_conv_layer(inputs, filters,layer_name)
    pool_112 = MaxPooling2D(pool_size=(2, 2))(conv_224)
    print('pool_112 is {0}'.format(pool_112.shape))
    layer_name = 'second_conv_layer'
    conv_112 = double_conv_layer(pool_112, 2 * filters,layer_name)
    pool_56 = MaxPooling2D(pool_size=(2, 2))(conv_112)

    print('pool_56 is {0}'.format(pool_56.shape))
    layer_name = 'third_conv_layer'
    conv_56 = double_conv_layer(pool_56, 4 * filters,layer_name)
    pool_28 = MaxPooling2D(pool_size=(2, 2))(conv_56)
    print('pool_28 is {0}'.format(pool_28.shape))
    layer_name = 'fourth_conv_layer'
    conv_28 = double_conv_layer(pool_28, 8 * filters,layer_name)
    pool_14 = MaxPooling2D(pool_size=(2, 2))(conv_28)
    print('pool_14 is {0}'.format(pool_14.shape))
    layer_name = 'fifth_conv_layer'
    conv_14 = double_conv_layer(pool_14, 16 * filters,layer_name)
    print('conv_14 is {0}'.format(conv_14.shape))
    pool_7 = MaxPooling2D(pool_size=(2, 2))(conv_14)
    print('pool_7 is {0}'.format(pool_7.shape))
    layer_name = 'six_conv_layer'
    center = double_conv_layer(pool_7,32* filters,layer_name)
    print('center is {0}'.format(center.shape))
    gate_signal= gating(center,32*filters)
    print('gate_signal is {0}'.format(gate_signal.shape))
    attn_coff_1 = grid_attention(conv_14, gate_signal,inter_channels=16*filters)
    neg_attn_coff_1=neg_grid_attention(conv_224,conv_112,conv_56,conv_28,scale_factor=16,inter_channels=16*filters)
    improv_atten_coeff_1 = Subtract()([attn_coff_1, neg_attn_coff_1])
    g_conv2=grid_attention_mult(conv_14,improv_atten_coeff_1,inter_channels=16*filters)
    print('g_conv2 is {0}'.format(g_conv2.shape))
    up_14 = concatenate([UpSampling2D(size=(2, 2))(center), g_conv2], axis=axis)
    layer_name = 'first_upconv_layer'
    up_14 = double_conv_layer(up_14, filters*16,layer_name)
    print('up_14 is {0}'.format(up_14.shape))
    attn_coff_2 = grid_attention(conv_28, up_14,inter_channels=8*filters)
    neg_attn_coff_2 = neg_grid_attention(conv_224, conv_112, conv_56,scale_factor=8,inter_channels=8*filters)
    improv_atten_coeff_2 = Subtract()([attn_coff_2, neg_attn_coff_2])
    g_conv3 =grid_attention_mult(conv_28,improv_atten_coeff_2,inter_channels=8*filters)
    up_28 = concatenate([UpSampling2D(size=(2, 2))(up_14), g_conv3], axis=axis)
    layer_name = 'second_upconv_layer'
    up_28 = double_conv_layer(up_28, filters*8,layer_name)
    print('up_28 is {0}'.format(up_28.shape))
    attn_coff_3 = grid_attention(conv_56, up_28,inter_channels=4*filters)
    neg_attn_coff_3 = neg_grid_attention(conv_224, conv_112,scale_factor=4,inter_channels=4*filters)
    improv_atten_coeff_3 = Subtract()([attn_coff_3, neg_attn_coff_3])
    g_conv4 =grid_attention_mult(conv_56,improv_atten_coeff_3,inter_channels=4*filters)
    up_56 = concatenate([UpSampling2D(size=(2, 2))(up_28), g_conv4], axis=axis)
    layer_name = 'fourth_upconv_layer'
    up_56 = double_conv_layer(up_56, filters*4,layer_name)
    print('up_56 is {0}'.format(up_56.shape))
    attn_coff_4 = grid_attention(conv_112, up_56, inter_channels=2*filters)
    neg_attn_coff_4 = neg_grid_attention(conv_224,scale_factor=2,inter_channels=2*filters)
    improv_atten_coeff_4 = Subtract()([attn_coff_4, neg_attn_coff_4])
    g_conv5 =grid_attention_mult(conv_112,improv_atten_coeff_4,inter_channels=2*filters)
    up_112 = concatenate([UpSampling2D(size=(2, 2))(up_56), g_conv5], axis=axis)
    layer_name = 'fifth_upconv_layer'
    up_112 = double_conv_layer(up_112, filters*2,layer_name)
    print('up_112 is {0}'.format(up_112.shape))
    up_224 = concatenate([UpSampling2D(size=(2, 2))(up_112), conv_224], axis=axis)
    layer_name = 'sixth_upconv_layer'
    up_224 = double_conv_layer(up_224, filters,layer_name)
    print('up_224 is {0}'.format(up_224.shape))

    up_conv_14 = dsv(up_14, OUTPUT_MASK_CHANNELS,16)
    up_conv_28 = dsv(up_28, OUTPUT_MASK_CHANNELS,8)
    up_conv_56 = dsv(up_56, OUTPUT_MASK_CHANNELS,4)
    up_conv_112 = dsv(up_112, OUTPUT_MASK_CHANNELS,2)
    up_conv_224 = dsv(up_224, OUTPUT_MASK_CHANNELS,1, dropout_val)
    final = concatenate([up_conv_14, up_conv_28,up_conv_56,up_conv_112,up_conv_224], axis=axis)

    conv_final = Conv2D(OUTPUT_MASK_CHANNELS, (1, 1))(final)

    print('conv_final is {0}'.format(conv_final.shape))
    conv_final = Activation('sigmoid')(conv_final)
    print('conv_final sig is {0}'.format(conv_final.shape))
    return conv_final