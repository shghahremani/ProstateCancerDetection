import keras
import keras.backend as K
import os
from tools.clr_callback import CyclicLR as cyclic


class TFCheckPointCallback(keras.callbacks.Callback):
    def __init__(self, saver, sess, config):
        self.saver = saver
        self.sess = sess
        self.config = config
        self.checkpoint_dir = os.path.join(config['training']['model_save_dir'],
                                           config['main']['config_name'],
                                           config['main']['model_name'])
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def on_epoch_end(self, epoch, logs=None):
        K.set_learning_phase(0)
        self.saver.save(self.sess, self.checkpoint_dir + '/freeze/checkpoint', global_step=epoch)


class SaveKerasModel(keras.callbacks.Callback):
    def __init__(self, config):
        self.folder = os.path.join(config['training']['model_save_dir'],
                                   config['main']['config_name'],
                                   config['main']['model_name'])
        self.config = config

    def on_epoch_end(self, epoch, logs=None):
        if (epoch % self.config['training']['model_save_frequency'] == 0) or \
                (epoch == 0) or (epoch == (self.config['training']['epochs'] - 1)):
            self.model.save(self.folder + '/weights__epoch' + str(epoch) + '.h5', overwrite=True)


def SelectLRCallBack(config):
    LR_type = config['training']['learning_rate']['type']
    if LR_type == 'cyclic':
        return cyclic(base_lr=config['training']['learning_rate']['base'],
                      max_lr=config['training']['learning_rate']['max'],
                      step_size=config['training']['learning_rate']['step'],
                      mode=config['training']['learning_rate']['mode'], )


def TensoBoardCallback(config):
    return keras.callbacks.TensorBoard(log_dir=config['training']['tensorboard_dir'],
                                       histogram_freq=0,
                                       write_graph=True,
                                       write_images=True)


def BestModelCallback(config):
    checkpoint_dir = os.path.join(config['training']['model_save_dir'],
                                  config['main']['config_name'],
                                  config['main']['model_name'])
    return keras.callbacks.ModelCheckpoint(checkpoint_dir + '/' + 'weights.{epoch:02d}-{val_f1:.2f}.hdf5'
                                           , monitor='val_f1', verbose=0, save_best_only=True,
                                           save_weights_only=False, mode='max', period=10)
