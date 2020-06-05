from __future__ import absolute_import, print_function

from keras.engine.topology import Input
from keras.layers.core import Activation
from keras.layers import  Conv2D
from keras.models import Model
from . import Base_Unet_model

from tools.metrics import f1 as f1_score
import tensorflow as tf

def transfer_weights(model, weights=None):
    """
    Always trains from scratch; never transfers weights
    :param model:
    :param weights:
    :return:
    """
    print('ENet has found no compatible pretrained weights! Skipping weight transfer...')
    return model


def build(nc, w, h,loss='categorical_crossentropy',optimizer='adam'):
    data_shape = w * h if None not in (w, h) else -1  # TODO: -1 or None?
    tissue_inp = Input(shape=(h, w, 3))
    print('tissue_inp is {0}'.format(tissue_inp.shape))
    weakly_inp = Input(shape=(h, w, 3))
    print('weakly_inp is {0}'.format(weakly_inp.shape))

    unet_1 = Base_Unet_model.build(tissue_inp, nc)
    unet_2 = Base_Unet_model.build(weakly_inp, nc)

    Radboud_Branch = Conv2D(6, (1, 1))(unet_1)
    # print('conv_final is {0}'.format(conv_final.shape))
    Radboud_Branch = Activation('sigmoid')(Radboud_Branch)
    # print('conv_final sig is {0}'.format(conv_final.shape))
    ##########
    Karolinska_Branch = Conv2D(3, (1, 1))(unet_2)
    # print('conv_final is {0}'.format(conv_final.shape))
    Karolinska_Branch = Activation('sigmoid')(Karolinska_Branch)

    model = Model(inputs=[tissue_inp,weakly_inp],outputs=[Radboud_Branch,Karolinska_Branch])

    # model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', 'mean_squared_error', f1_score])
    name = 'Unet_Prostate'

    return model, name
