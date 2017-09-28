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


class AE(chainer.Chain):
    def __init__(self, data_obj):
        self.data_obj = data_obj
        super(AE, self).__init__()
        with self.init_scope():
            # parts of netowork
            self.len1 = L.Linear(None, 100)
            self.lde1 = L.Linear(None, data_obj.total_dim)

    @within_name_scope('AE')
    def encode(self, x):
        with name_scope('linear1', self.len1.params()):
            return F.relu(self.len1(x))

    @within_name_scope('AE')
    def decode(self, z):
        with name_scope('linear2', self.lde1.params()):
            return F.relu(self.lde1(z))

    def __call__(self, x):
        x = x.reshape(-1,self.data_obj.total_dim)
        h1 = self.encode(x)
        h2 = self.decode(h1)
        return h2.reshape(self.data_obj.batch_data_shape)

    def get_loss_func(self):
        def loss_func(x):
            x = x.reshape(-1, self.data_obj.total_dim)
            rec_x = self.decode(self.encode(x))
            rec_x = rec_x.reshape(-1, self.data_obj.total_dim)
            self.loss = F.mean_squared_error(x, rec_x)
            return self.loss
        return loss_func
