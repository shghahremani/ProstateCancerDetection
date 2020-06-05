from __future__ import absolute_import, print_function

from keras.engine.topology import Input
from keras.layers.core import Activation, Reshape
from keras.layers import  Conv2D , Lambda, Multiply, Reshape, Dense
from keras.layers.core import  Activation
from keras.models import Model
from keras import backend as K
from . import base_unet
from . import super_branch
from . import semisuper_branch
from . import Label_branch
from kwae.tools.metrics import f1 as f1_score
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



def build(nc, w, h,cfg):
    data_shape = w * h if None not in (w, h) else -1  # TODO: -1 or None?
    tissue_inp = Input(shape=(h, w, 3))
    print('tissue_inp is {0}'.format(tissue_inp.shape))
    weakly_inp = Input(shape=(h, w, 3))
    print('weakly_inp is {0}'.format(weakly_inp.shape))
    #WoundSkin = Input(shape=(h, w, 3))
    #print('WoundSkin_input is {0}'.format(WoundSkin.shape))
    unet_1 = base_unet.build(tissue_inp, nc)
    unet_2 = base_unet.build(weakly_inp, nc)
    #extract 3 class output

    seg_branch = super_branch.build(unet_1, nc)

    #wealky 9 class branch

    #####################upper loss
    #weakly_branch=semisuper_branch.build(conv_final, nc)
    #label_branch = label_branch.build(config=cfg,inp=conv_final,inp_indice=weakly_inp,nc= nc,w=w, h=h)
    label_branch=Label_branch.build(unet_2)

    ##### LOAD 3 CLASS TRAINED MODEL WEIGHT






    #enet = decoder.build(enet, nc=nc)
    name = 'unet_naive_upsampling'

    # enet = Reshape((data_shape, nc))(enet)  # TODO: need to remove data_shape for multi-scale training
    # with tf.name_scope('output'):
    #     enet = Activation('softmax',name="predictions")(enet)
    #unet = Activation('softmax',name="segmentation_map")(unet)
    model = Model(inputs=[tissue_inp,weakly_inp],outputs=[seg_branch,label_branch])

    # model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', 'mean_squared_error', f1_score])

    return model, name
