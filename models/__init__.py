# coding=utf-8
from tensorflow.contrib.keras import utils


def select_model(config,dataset):
    model_name= config['main']['model_name']
    if model_name == 'enet_unpooling':
        from .enet import model as raw_model
        model, _ = raw_model.build(nc=dataset.nc,
                                   w=dataset.dw,
                                   h=dataset.dh)
    elif  model_name == 'unet':
        from .unet import model as raw_model
        #if net=="first":
        model, _ = raw_model.build(nc=dataset.nc,
                                              w=dataset.dw,
                                              h=dataset.dh)
        #else:
           # model, _ = raw_model.build(nc=dataset.nc,
                                      # w=config['training']['ROI_dw'],
                                     #  h=config['training']['ROI_dh'])
    elif  model_name == 'unet_improved_attention':
        from .unet_imprv_att import model as raw_model
        #if net=="first":
        model, _ = raw_model.build(nc=dataset.nc,
                                              w=dataset.dw,
                                              h=dataset.dh)
    elif  model_name == 'unet_attention':
        from .unet_attention import model as raw_model
        #if net=="first":
        model, _ = raw_model.build(nc=dataset.nc,
                                              w=dataset.dw,
                                              h=dataset.dh)
    elif  model_name == 'weakly_sup':
        from .weakly_sup import model as raw_model
        #if net=="first":
        model, _ = raw_model.build(nc=dataset.nc,
                                              w=dataset.dw,
                                              h=dataset.dh, cfg=config)
    elif  model_name == 'cascaded_weakly':
        from .cascaded_weakly import model as raw_model
        #if net=="first":
        model, _ = raw_model.build(nc=dataset.nc,
                                              w=dataset.dw,
                                              h=dataset.dh, cfg=config)
    elif  model_name == 'deeplab':
        from .deeplab import model as raw_model
        model, _ = raw_model.build(nc=dataset.nc,
                                              w=config['training']['dw'],
                                              h=config['training']['dh'])
    elif model_name == 'mobilenet':
        from .mobilenet import mobilenet as model
    elif model_name=='simple':
        from .simple_convnet import simple_conv as model
    elif model_name=='mobilenet_v2':
        from .mobilenet import mobilenets as raw_model
        model = raw_model.MobileNetV2(input_shape=dataset.input_shape,
                                      alpha=1.0,
                                      depth_multiplier=1,
                                      dropout=1e-3,
                                      include_top=True,
                                      weights=None,
                                      input_tensor=None,
                                      pooling=None,
                                      classes=dataset.nc)

    else:
        raise ValueError('Unknown model {}'.format(model_name))
    return model


def plot(model_name):
    model = select_model(model_name=model_name)
    autoencoder, name = model.build(nc=2, w=512, h=512)
    utils.plot_model(autoencoder, to_file='{}.png'.format(name), show_shapes=True)

