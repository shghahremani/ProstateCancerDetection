import numpy as np
import os
import tensorflow as tf


class _GridAttentionBlockND(nn.Module):
    def __init__(self, input_signal,gating_signal, gating_channels, dimension=3, mode='concatenation',
                 sub_sample_factor=(2,2,2)):
        super(_GridAttentionBlockND, self).__init__()

        assert dimension in [2, 3]
        assert mode in ['concatenation', 'concatenation_debug', 'concatenation_residual']

        # Downsampling rate for the input featuremap
        if isinstance(sub_sample_factor, tuple): self.sub_sample_factor = 2
        elif isinstance(sub_sample_factor, list): self.sub_sample_factor = tuple(sub_sample_factor)
        else: self.sub_sample_factor = tuple([sub_sample_factor]) * dimension

        # Default parameter set
        self.mode = mode
        self.dimension = dimension
        self.sub_sample_kernel_size = self.sub_sample_factor

        # Number of channels (pixel dimensions)
        self.input_signal = input_signal
        self.gating_signal=gating_signal
        self.gating_channels = gating_channels
        #self.inter_channels = inter_channels

        #if self.inter_channels is None:
         #   self.inter_channels = in_channels // 2
         #   if self.inter_channels == 0:
         #       self.inter_channels = 1


        input_size = self.input_signal.size()
        batch_size = input_size[0]
        assert batch_size == gating_signal.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = tf.layers.conv2d(inputs=input_signal, filters=self.inter_channels,
                             kernel_size=self.sub_sample_kernel_size, strides=self.sub_sample_factor, padding='valid', use_bias=False,kernel_initializer=
                            tf.contrib.layers.variance_scaling_initializer())
        theta_x_size = theta_x.shape
        phi_g=tf.layers.conv2d(inputs=gating_signal, filters=self.inter_channels,
                           kernel_size=1, strides=1, padding='valid', use_bias=True,kernel_initializer=
                            tf.contrib.layers.variance_scaling_initializer())
        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = tf.image.resize_bilinear(image=phi_g, size=theta_x_size[1:2])
        cancat=phi_g+theta_x
        f = tf.nn.relu(cancat)
        psi=tf.layers.conv2d(inputs=f, filters=1, kernel_size=1, strides=1, padding='valid', use_bias=True,kernel_initializer=
                            tf.contrib.layers.variance_scaling_initializer())
        sigm_psi_f=tf.sigmoid(psi)
        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        #sigm_psi_f = F.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = tf.image.resize_bilinear(image=sigm_psi_f, size=theta_x_size[1:2])
        print(input_signal.shape)
        print(tf.tile(tf.expand_dims(sigm_psi_f, 3), input_signal.shape).shape)
        y = tf.tile(tf.expand_dims(sigm_psi_f, 3), input_signal.shape) * input_signal
        W_y = tf.layers.conv2d(inputs=y, filters=self.in_channels, kernel_size=1, strides=1, padding='valid')
        w_y=tf.layers.batch_normalization(W_y,axis=3)

        return w_y, sigm_psi_f



class GridAttentionBlock2D(_GridAttentionBlockND):
    def __init__(self, in_channels, gating_channels, inter_channels=None, mode='concatenation',
                 sub_sample_factor=(2,2)):
        super(GridAttentionBlock2D, self).__init__(in_channels,
                                                   inter_channels=inter_channels,
                                                   gating_channels=gating_channels,
                                                   dimension=2, mode=mode,
                                                   sub_sample_factor=sub_sample_factor,
                                                   )
