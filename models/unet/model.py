from __future__ import absolute_import, print_function

from keras.engine.topology import Input
from keras.layers.core import Activation, Reshape
from keras.models import Model
from . import encoder

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


def build(nc, w, h,
          loss='categorical_crossentropy',
          optimizer='adam'):
    data_shape = w * h if None not in (w, h) else -1  # TODO: -1 or None?
    inp = Input(shape=(h, w, 3))

    unet = encoder.build(inp, nc)
    #enet = decoder.build(enet, nc=nc)
    name = 'unet_naive_upsampling'

    # enet = Reshape((data_shape, nc))(enet)  # TODO: need to remove data_shape for multi-scale training
    # with tf.name_scope('output'):
    #     enet = Activation('softmax',name="predictions")(enet)
    #unet = Activation('softmax',name="segmentation_map")(unet)
    model = Model(inputs=inp, outputs=unet)

    # model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', 'mean_squared_error', f1_score])

    return model, name