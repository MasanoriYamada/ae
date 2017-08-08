#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import six
from sklearn.model_selection import train_test_split
from abc import ABCMeta, abstractmethod


class Dataset(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        self.name = 'abstract'
        self.data_shape = []
        self.batch_data_shape = ()
        self.width = -1
        self.height = -1
        self.train_size = -1
        self.test_size = -1
        self.range = [0., 1.]
        self.train_idx = 0
        self.test_idx = 0
        self.total_dim = 0

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

    def get_test_iter(self, batch_size=None, noise_type=None):
        """ test_iterを返す
        batchsize指定: batchsieで区切ったiter 端数は無視
        指定なし: 全部のtestを1つに固めて返す
        """

        test_iter = []
        if batch_size is None:
            data, label = self.test_array._datasets
            if noise_type is None:
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

    def reset(self):
        self.handle_unsupported_op()

    def handle_unsupported_op(self):
        pass

    def add_noise(self, original, noise_type):
        noisy_input = original
        if noise_type == 'gauss':
            # Add Gaussian noise
            noisy_input += np.random.normal(scale=0.1, size=original.shape)

        # Add salt and pepper noise
        elif noise_type == 'salt':
            noisy_input = np.multiply(original, np.random.binomial(n=1, p=0.9, size=original.shape)) + \
                      np.random.binomial(n=1, p=0.1, size=original.shape)

        return np.clip(noisy_input, a_min=self.range[0], a_max=self.range[1]) # clipping

