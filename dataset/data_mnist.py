#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dataset.dataset
import chainer

class MnistDataset(dataset.dataset.Dataset):
    def __init__(self):
        self.namda = 'mnist'
        self.data_shape = [1, 28, 28]
        self.witdth = 28
        self.height = 28
        self.train_size = 50000
        self.test_size = 10000
        self.range = [0., 1.]
        self.train_idx = 0
        self.test_idx = 0

        self.x_data = None
        self.label_data = None
        train, test = chainer.datasets.get_mnist()
        self.train_array = train
        self.test_array = test