#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import matplotlib
matplotlib.use('Pdf')
import matplotlib.pyplot as plt
import chainer
import numpy as np
import six

import dataset.dataset


class MnistDataset(dataset.dataset.Dataset):
    def __init__(self):
        super(MnistDataset, self).__init__()
        self.name = 'mnist'
        self.data_shape = [28, 28, 1]
        self.batch_data_shape = (-1, 28, 28, 1)
        self.width = 28
        self.height = 28
        self.train_size = 50000
        self.test_size = 10000
        self.range = [0., 1.]
        self.total_dim = 28*28

        self.x_data = None
        self.label_data = None
        train, test = chainer.datasets.get_mnist()
        self.train_array = train
        self.test_array = test

    def get_test_iter(self, batch_size=None, noise_type=None):
        """ test_iterを返す
        batchsize指定: batchsieで区切ったiter 端数は無視
        指定なし: 全部のtestを1つに固めて返す
        """

        test_iter = []
        if batch_size is None:
            data, label = self.test_array._datasets
            if noise_type is not None:
                data = self.add_noise(data[:self.test_size], noise_type)
            test_iter.append((data[:self.test_size], label[:self.test_size]))

        elif batch_size is not None:
            data, label = self.test_array._datasets[:self.test_size]
            for i in six.moves.range(0, self.test_size, batch_size):
                split_data, split_label = data[i: i + batch_size], label[i: i + batch_size]
                if noise_type is not None:
                    split_data = self.add_noise(split_data, noise_type)
                test_iter.append((split_data, split_label))
        return test_iter

    def get_train_iter(self, batch_size, noise_type=None):
        """batchsizeのiterを返す(端数は無視)
        順序はシャッフルされる
        """
        perm = np.random.permutation(self.train_size)
        train_iter = []
        data, label = self.train_array._datasets # need for random sampling in all data
        for i in six.moves.range(0, self.train_size, batch_size):
            split_data, split_label = data[perm[i: i + batch_size]], label[perm[i: i + batch_size]]
            if noise_type is not None:
                split_data = self.add_noise(split_data, noise_type)
            train_iter.append((split_data, split_label))
        return train_iter

    def display(self, image):
        return image

if __name__ == '__main__':
    data = MnistDataset()
    while True:
        batch = data.get_train_iter(batch_size=100)[0]
        print(batch.shape)
        plt.imshow(data.display(batch[0]))
        plt.show()