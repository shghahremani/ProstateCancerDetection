from __future__ import absolute_import, print_function

from keras.engine.topology import Input
from keras.layers.core import Activation, Reshape
from keras.models import Model
from . import encoder
import os
import json
from keras.utils.data_utils import get_file

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


def build(nc, w, h,):

    solver_json = '../../config/deep_lab'

    solver = json.load(open(os.path.abspath(solver_json)))

    data_shape = w * h if None not in (w, h) else -1  # TODO: -1 or None?
    inp = Input(shape=(h, w, 3))

    deeplab = encoder.build(inp,solver,nc)
    #enet = decoder.build(enet, nc=nc)
    name = 'deeplabv3+'
    model = Model(inputs=inp, outputs=deeplab)

    # load weights
    weights = solver['weights']
    backbone = solver['backbone']
    WEIGHTS_PATH_X = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5"
    WEIGHTS_PATH_MOBILE = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5"
    if weights == 'pascal_voc':
        if backbone == 'xception':
            weights_path = get_file('deeplabv3_xception_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH_X,
                                    cache_subdir='models')
        else:
            weights_path = get_file('deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH_MOBILE,
                                    cache_subdir='models')
        model.load_weights(weights_path, by_name=True)

    return model, name
