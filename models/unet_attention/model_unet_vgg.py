from keras.models import *
from keras.layers import *

from .model_utils import get_segmentation_model
from .vgg16 import get_vgg_encoder
from .mobilenet import get_mobilenet_encoder
from .basic_models import vanilla_encoder
from .resnet50 import get_resnet50_encoder
import tensorflow as tf

IMAGE_ORDERING_CHANNELS_LAST = "channels_last"
IMAGE_ORDERING_CHANNELS_FIRST = "channels_first"

# Default IMAGE_ORDERING = channels_last
IMAGE_ORDERING = IMAGE_ORDERING_CHANNELS_LAST
if IMAGE_ORDERING == 'channels_first':
    MERGE_AXIS = 1
elif IMAGE_ORDERING == 'channels_last':
    MERGE_AXIS = -1


def UpSampling2DBilinear(stride, **kwargs):
    def layer(x):
        input_shape = K.int_shape(x)
        output_shape = (stride * input_shape[1], stride * input_shape[2])
        return tf.image.resize_bilinear(x, output_shape, align_corners=True)
    return Lambda(layer, **kwargs)

def grid_attention(input_signal,gating_signal,inter_channels):
    if K.image_dim_ordering() == 'th':
        axis = 1
    else:
        axis = 3
    #print(input_signal.shape)
    #print(gating_signal.shape)
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

    return w_y, sigm_psi_f
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

def unet_mini(n_classes, input_height=360, input_width=480):

    if IMAGE_ORDERING == 'channels_first':
        img_input = Input(shape=(3, input_height, input_width))
    elif IMAGE_ORDERING == 'channels_last':
        img_input = Input(shape=(input_height, input_width, 3))

    conv1 = Conv2D(32, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(img_input)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2), data_format=IMAGE_ORDERING)(conv1)

    conv2 = Conv2D(64, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2), data_format=IMAGE_ORDERING)(conv2)

    conv3 = Conv2D(128, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(conv3)

    up1 = concatenate([UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(
        conv3), conv2], axis=MERGE_AXIS)
    conv4 = Conv2D(64, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(conv4)

    up2 = concatenate([UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(
        conv4), conv1], axis=MERGE_AXIS)
    conv5 = Conv2D(32, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(conv5)

    o = Conv2D(n_classes, (1, 1), data_format=IMAGE_ORDERING,
               padding='same')(conv5)

    model = get_segmentation_model(img_input, o)
    model.model_name = "unet_mini"
    return model


def _unet(n_classes, encoder, l1_skip_conn=True, input_height=480,
          input_width=640):
    filters = 16
    axis = 3
    dropout_val = 0.2
    OUTPUT_MASK_CHANNELS = n_classes

    img_input, levels = encoder(
        input_height=input_height, input_width=input_width)
    [f1, f2, f3, f4, f5, f1_center, f2_center, f3_center, f4_center, f5_center] = levels

    print(f5_center.shape)
    center = double_conv_layer(f5_center, 32 * filters, 'center')
    # print('center is {0}'.format(center.shape))
    gate_signal = gating(center, 32 * filters)
    print(gate_signal.shape)
    # print('gate_signal is {0}'.format(gate_signal.shape))
    g_conv2, att2 = grid_attention(f5, gate_signal, inter_channels=16 * filters)
    print(g_conv2.shape)

    # print('g_conv2 is {0}'.format(g_conv2.shape))
    up_14 = concatenate([UpSampling2D(size=(2, 2))(center), g_conv2], axis=axis)
    layer_name = 'first_upconv_layer'
    up_14 = double_conv_layer(up_14, filters * 16, layer_name)
    # print('up_14 is {0}'.format(up_14.shape))
    g_conv3, att3 = grid_attention(f4, up_14, inter_channels=8 * filters)
    up_28 = concatenate([UpSampling2D(size=(2, 2))(up_14), g_conv3], axis=axis)
    layer_name = 'second_upconv_layer'
    up_28 = double_conv_layer(up_28, filters * 8, layer_name)
    print(layer_name,up_28.shape)

    # print('up_28 is {0}'.format(up_28.shape))
    g_conv4, att4 = grid_attention(f3, up_28, inter_channels=4 * filters)
    up_56 = concatenate([UpSampling2D(size=(2, 2))(up_28), g_conv4], axis=axis)
    layer_name = 'fourth_upconv_layer'
    up_56 = double_conv_layer(up_56, filters * 4, layer_name)
    print(layer_name,up_56.shape)

    # print('up_56 is {0}'.format(up_56.shape))
    g_conv5, att5 = grid_attention(f2, up_56, inter_channels=2 * filters)
    up_112 = concatenate([UpSampling2D(size=(2, 2))(up_56), g_conv5], axis=axis)
    layer_name = 'fifth_upconv_layer'
    print(layer_name, up_112.shape)
    up_112 = double_conv_layer(up_112, filters * 2, layer_name)
    # print('up_112 is {0}'.format(up_112.shape))
    up_224 = concatenate([UpSampling2D(size=(2, 2))(up_112), f1], axis=axis)
    layer_name = 'sixth_upconv_layer'
    up_224 = double_conv_layer(up_224, filters, layer_name)
    print(layer_name,up_224.shape)

    # print('up_224 is {0}'.format(up_224.shape))

    up_conv_14 = dsv(up_14, OUTPUT_MASK_CHANNELS, 16)
    up_conv_28 = dsv(up_28, OUTPUT_MASK_CHANNELS, 8)
    up_conv_56 = dsv(up_56, OUTPUT_MASK_CHANNELS, 4)
    up_conv_112 = dsv(up_112, OUTPUT_MASK_CHANNELS, 2)
    up_conv_224 = dsv(up_224, OUTPUT_MASK_CHANNELS, 1, dropout_val)
    final = concatenate([up_conv_14, up_conv_28, up_conv_56, up_conv_112, up_conv_224], axis=axis)

    conv_final = Conv2D(OUTPUT_MASK_CHANNELS, (1, 1))(final)
    # print('conv_final is {0}'.format(conv_final.shape))
    conv_final = Activation('sigmoid')(conv_final)
    # print('conv_final sig is {0}'.format(conv_final.shape))

    # print('conv_final sig is {0}'.format(conv_final.shape))

    return img_input, conv_final


def unet(n_classes, input_height=416, input_width=608, encoder_level=3):

    model = _unet(n_classes, vanilla_encoder,
                  input_height=input_height, input_width=input_width)
    model.model_name = "unet"
    return model


def vgg_unet(n_classes, input_height=416, input_width=608, encoder_level=3):

    model = _unet(n_classes, get_vgg_encoder,
                  input_height=input_height, input_width=input_width)
    model.model_name = "vgg_unet"
    return model


def resnet50_unet(n_classes, input_height=416, input_width=608,
                  encoder_level=3):

    model = _unet(n_classes, get_resnet50_encoder,
                  input_height=input_height, input_width=input_width)
    model.model_name = "resnet50_unet"
    return model


def mobilenet_unet(n_classes, input_height=224, input_width=224,
                   encoder_level=3):

    model = _unet(n_classes, get_mobilenet_encoder,
                  input_height=input_height, input_width=input_width)
    model.model_name = "mobilenet_unet"
    return model


# if __name__ == '__main__':
#     m = unet_mini(101)
#     m = _unet(101, vanilla_encoder)
#     # m = _unet( 101 , get_mobilenet_encoder ,True , 224 , 224  )
#     m = _unet(101, get_vgg_encoder)
#     m = _unet(101, get_resnet50_encoder)