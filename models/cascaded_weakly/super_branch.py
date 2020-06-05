from keras.layers import  Conv2D
from keras.layers.core import  Activation


def build(inp, nc):


    input=inp
    OUTPUT_MASK_CHANNELS = nc
    conv_final = Conv2D(OUTPUT_MASK_CHANNELS, (1, 1))(input)
    # print('conv_final is {0}'.format(conv_final.shape))
    conv_final = Activation('sigmoid')(conv_final)
    # print('conv_final sig is {0}'.format(conv_final.shape))
    return conv_final