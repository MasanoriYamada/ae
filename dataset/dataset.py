#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import six
from sklearn.model_selection import train_test_split

class Dataset(object):
    def __init__(self):
        self.namda = 'abstract'
        self.data_shape = []
        self.witdth = -1
        self.height = -1
        self.train_size = -1
        self.test_size = -1
        self.range = [0., 1.]
        self.train_idx = 0
        self.test_idx = 0

        self.x_data = None
        self.label_data = None
        self.train_array = None
        self.test_array = None

    def split_train_test(self, split_rate=0.8):
        train_x, test_x, train_label, test_label = train_test_split(self.x_data, self.label_data,
                                                                    test_size = split_rate, random_state = 42)
        self.train_array = (train_x, train_label)
        self.test_array = (test_x, test_label)
        self.train_size = len(train_label)
        self.test_size = len(test_label)

    def get_test_iter(self, size=None, batch_size=None):
        """ test_iterを返す
        batchsize指定: batchsieで区切ったiter 端数は無視
        size指定: sizeだけの長さを返す(順序固定)
        指定なし: 全部のtestを1つに固めて返す
        """
        test_iter = []
        if batch_size == None and size == None:
            return self.test_array

        elif batch_size != None and size != None:
            print('Error get_test_iter in dataset.py')

        elif size != None:
            data, label = self.test_array._datasets
            test_iter.append((data[:size], label[:size]))
            return test_iter

        elif batch_size != None:
            for i in six.moves.range(0, self.test_size, batch_size):
                data, label = self.test_array
                split_data, split_label = data[i: i + batch_size], label[i: i + batch_size]
                test_iter.append((split_data, split_label))
            return np.array(test_iter)

    def get_train_iter(self, batch_size):
        """batchsizeのiterを返す(端数は無視)
        順序はシャッフルされる
        """
        perm = np.random.permutation(self.train_size)
        train_iter = []
        data, label = self.train_array._datasets
        for i in six.moves.range(0, self.train_size, batch_size):
            split_data, split_label = data[perm[i: i + batch_size]], label[perm[i: i + batch_size]]
            train_iter.append((split_data, split_label))
        return np.array(train_iter)

    def add_noise(self, original):
        # add salt and pepper noise
        noisy_input = np.multiply(original, np.random.binomial(n=1, p=0.9, size=original.shape)) \
        + np.random.binomial(n=1, p=0.1, size=original.shape)
        # add gaussian noise
        noisy_input += np.random.normal(scale=0.1, size=original.shape)
        return np.clip(noisy_input, a_min=self.range[0], a_max=self.range[1]) # clipping

    def display(self, image):
        return image

    def reset(self):
        self.handle_unsupported_op()

    def handle_unsupported_op(self):
        pass

