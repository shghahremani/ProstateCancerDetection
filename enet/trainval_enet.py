# global scope
import json
import shutil
import sys

import numpy as np
import os

from jsmin import jsmin
from keras import backend as K
import keras
import tensorflow as tf

# project scope
from models import select_model
from dataset import select_dataset
from keras.optimizers import *

from tools.callbacks import TFCheckPointCallback, SaveKerasModel, SelectLRCallBack, TensoBoardCallback
from tools.metrics import f1 as f1_score


def train(cfg):
    print('Preparing to train {} on {} data...'.format(cfg['main']['model_name'],
                                                       cfg['main']['dataset_name']))

    np.random.seed(1337)  # for reproducibility

    print('Tensorflow backend detected; Applying memory usage constraints')
    ss = K.tf.Session(config=K.tf.ConfigProto(gpu_options=K.tf.GPUOptions(allow_growth=True)))
    K.set_session(ss)
    ss.run(K.tf.global_variables_initializer())
    K.set_learning_phase(1)

    Prostate_dataset = select_dataset(cfg)
    if not cfg["validation"]["do_validation"]:
        image_gen = Prostate_dataset.create_generator()
        val_data = None
    else:
        image_gen, val_data = Prostate_dataset.create_generator_with_validation()
    model = select_model(cfg, Prostate_dataset)
    # model.summary()
    myAdam = keras.optimizers.Adam(lr=1e-4)

    model.compile(optimizer=Adam(lr=1e-4),
                      loss={'Radboud_Branch': 'categorical_crossentropy', 'Karolinska_Branch': 'categorical_crossentropy'}, loss_weights=[0.5, 0.5], metrics=['accuracy'])

    print('Created image generator from dataset {}!'.format(cfg['main']['dataset_name']))
    tf_saver = tf.train.Saver(max_to_keep=2)

    checkpoint_callback = TFCheckPointCallback(tf_saver, ss, cfg)
    modelsaver_callback = SaveKerasModel(cfg)
    learningRate_callback = SelectLRCallBack(cfg)
    tbCallBack = TensoBoardCallback(cfg)



    model.fit_generator(image_gen,
                        steps_per_epoch=Prostate_dataset.sample_number / cfg['training']['batch_size'],
                        epochs=cfg['training']['epochs'],
                        callbacks=[checkpoint_callback, learningRate_callback, modelsaver_callback, tbCallBack],
                        validation_data=val_data)







if __name__ == '__main__':
    # solver_json = sys.argv[1]
    solver_json='../config/Prostate.json'
    print('solver json: {}'.format(os.path.abspath(solver_json)))

    with open(solver_json) as json_file:
        config = jsmin(json_file.read())
        config = json.loads(config)
    #Copies the file src to the file or directory dst
    shutil.copy(solver_json, os.path.join(config['training']['model_save_dir'],
                                  config['main']['config_name'],
                                  config['main']['model_name']))

    train(cfg=config)
