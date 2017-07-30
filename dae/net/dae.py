#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.links as L
import chainer.functions as F

class DAE(chainer.Chain):
    def __init__(self):
        super(DAE, self).__init__()
        with self.init_scope():
            # parts of netowork
            # encoder
            self.len1 = L.Linear(28*28, 100)
            self.lde1 = L.Linear(100, 28*28)

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