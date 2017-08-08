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

class AE(chainer.Chain):
    def __init__(self, data_obj):
        super(AE, self).__init__()
        with self.init_scope():
            # parts of netowork
            self.len1 = L.Linear(data_obj.total_dim, 100)
            self.lde1 = L.Linear(100, data_obj.total_dim)

    def encode(self, x):
        return F.relu(self.len1(x))

    def decode(self, z):
        return F.relu(self.lde1(z))

    def __call__(self, x):
        return self.decode(self.encode(x))

    def get_loss_func(self):
        def loss_func(x):
            rec_x = self.decode(self.encode(x))
            self.loss = F.mean_squared_error(x, rec_x)
            return self.loss
        return loss_func