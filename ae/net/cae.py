#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import chainer
import chainer.links as L
import chainer.functions as F
from tensorboard import name_scope, within_name_scope

class CAE(chainer.Chain):
    def __init__(self, data_obj):
        self.data_obj = data_obj
        super(CAE, self).__init__()
        with self.init_scope():
            # parts of netowork
            # autoencoder with cnn
            if self.data_obj.name == 'mnist':
                # self.len1 = L.Convolution2D(in_channels=1, out_channels=32, ksize=4, stride=2, pad=1)
                # self.len2 = L.Convolution2D(in_channels=32, out_channels=64, ksize=4, stride=2, pad=1)
                # self.len3 = L.Linear(7*7*64, 100)
                # self.lde0 = L.Linear(100, 7*7*64)
                # self.lde1 = L.Deconvolution2D(in_channels=64, out_channels=32, ksize=4, stride=2, pad=1)
                # self.lde2 = L.Deconvolution2D(in_channels=32, out_channels=1, ksize=4, stride=2, pad=1)
                self.len1 = L.Convolution2D(in_channels=1, out_channels=32, ksize=5, stride=1, pad=0)
                self.len2 = L.Convolution2D(in_channels=32, out_channels=64, ksize=5, stride=1, pad=0)  # 20x20
                self.len3 = L.Linear(20*20*64, 100)
                self.conv_linear_shape = (-1,20*20*64)
                self.deconv_linear_shape = (-1, 64, 20, 20)
                self.lde1 = L.Linear(100, 20*20*64)
                self.lde2 = L.Deconvolution2D(in_channels=64, out_channels=32, ksize=5, stride=1, pad=0)
                self.lde3 = L.Deconvolution2D(in_channels=32, out_channels=1, ksize=5, stride=1, pad=0)

            elif self.data_obj.name == 'celebA':
                self.len1 = L.Convolution2D(in_channels=3, out_channels=32, ksize=5, stride=1, pad=0)
                self.len2 = L.Convolution2D(in_channels=32, out_channels=64, ksize=5, stride=1, pad=0)
                self.lde1 = L.Deconvolution2D(in_channels=64, out_channels=32, ksize=5, stride=1, pad=0)
                self.lde2 = L.Deconvolution2D(in_channels=32, out_channels=3, ksize=5, stride=1, pad=0)
            elif self.data_obj.name == 'dsprites':
                self.len1 = L.Convolution2D(in_channels=1, out_channels=32, ksize=5, stride=1, pad=0)
                self.len2 = L.Convolution2D(in_channels=32, out_channels=64, ksize=5, stride=1, pad=0)
                self.len3 = L.Linear(56*56*64, 100)
                self.conv_linear_shape = (-1,56*56*64)
                self.deconv_linear_shape = (-1, 64, 56, 56)
                self.lde1 = L.Linear(100, 56*56*64)
                self.lde2 = L.Deconvolution2D(in_channels=64, out_channels=32, ksize=5, stride=1, pad=0)
                self.lde3 = L.Deconvolution2D(in_channels=32, out_channels=1, ksize=5, stride=1, pad=0)

    def encode(self, x):
        with name_scope('conv1', self.len1.params()):
            x = F.relu(self.len1(x))
        with name_scope('conv2', self.len2.params()):
            x = F.relu(self.len2(x))
        with name_scope('linear1', self.len3.params()):
            x = self.len3(x.reshape(self.conv_linear_shape))
        return x

    def decode(self, z):
        with name_scope('linear2', self.lde1.params()):
            z = F.relu(self.lde1(z))
            z = z.reshape(self.deconv_linear_shape)
        with name_scope('conv3', self.lde2.params()):
            z = F.relu(self.lde2(z))
        with name_scope('conv4', self.lde3.params()):
            z = F.sigmoid(self.lde3(z))
        return z

    @within_name_scope('CAE')
    def __call__(self, x):
        x = x.reshape((-1, self.data_obj.data_shape[0], self.data_obj.data_shape[1], self.data_obj.data_shape[2]))
        x = x.transpose(0, 3, 1, 2)
        with name_scope('encode', self.len1.params()):
            h = self.encode(x)
        with name_scope('decode', self.lde1.params()):
            h = self.decode(h)
        return h.transpose(0,2,3,1)

    def get_loss_func(self):
        def loss_func(x):
            x = x.reshape((-1, self.data_obj.data_shape[0], self.data_obj.data_shape[1], self.data_obj.data_shape[2]))
            x = x.transpose(0,3,1,2)
            rec_x = self.decode(self.encode(x))
            self.loss = F.mean_squared_error(x, rec_x)
            return self.loss
        return loss_func