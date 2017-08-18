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


class DspritesDataset(dataset.dataset.Dataset):
    def __init__(self, db_path='~/lab/dat/dsprites_data'):
        super(DspritesDataset, self).__init__()
        self.name = 'dsprites'
        self.data_shape = [64, 64, 1]
        self.batch_data_shape = (-1, 64, 64, 1)
        self.width = 64
        self.height = 64
        self.train_size = 32*32*6*40*3
        self.test_size = 32*32*6*40*3
        self.range = [0., 1.]
        self.total_dim = 64*64

        # original
        self.db_path = db_path
        dataset_zip = np.load(db_path + '/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', encoding='latin1')
        self.imgs = dataset_zip['imgs']
        metadata = dataset_zip['metadata'][()]
        # Define number of values per latents and functions to convert to indices
        self.latents_sizes = metadata['latents_sizes']
        self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:],
                                        np.array([1, ])))
        self.nc = self.latents_sizes[0]
        self.nl = self.latents_sizes[1]
        self.ns = self.latents_sizes[2]
        self.nr = self.latents_sizes[3]
        self.nx = self.latents_sizes[4]
        self.ny = self.latents_sizes[5]

        self.x_data = self.set_train_test_data()
        self.label_data = np.zeros(shape=(self.train_size,1))
        self.train_array = None
        self.test_array = None
        self.train_size = -1
        self.test_size = -1
        self.split_train_test(split_rate=0.8)  # set train and test array

    def _latent_to_index(self, latents):
      return np.dot(latents, self.latents_bases).astype(int)
    def _2d_to_1d(self, imgs):
        shape = imgs.shape
        return imgs.reshape(shape[0], -1)

    def _sample_latent(self, x_start, x_goal, y_start, y_goal, scale_start, scale_goal, rot_start, rot_goal, shape_lst):
        samples = []
        for ci in range(self.nc):
            for shapei in shape_lst:
                for scalei in range(scale_start, scale_goal):
                    for roti in range(rot_start, rot_goal):
                        for xi in range(x_start, x_goal):
                            for yi in range(y_start,y_goal):
                                samples.append([ci, shapei, scalei, roti,xi, yi])
        return np.array(samples)

    def _random_sample_latent(self, size=1):
        samples = np.zeros((size, self.latents_sizes.size))
        for lat_i, lat_size in enumerate(self.latents_sizes):
            samples[:, lat_i] = np.random.randint(lat_size, size=size)
        return samples

    def set_train_test_data(self):
        # Sample latents randomly
        latents_sampled = self._sample_latent(0, 32, 0, 32, 0, 6, 0, 40, [0,1,2])
        # Select images
        id = self._latent_to_index(latents_sampled)
        # calc index in arg conditions
        return self._2d_to_1d(self.imgs[id])

    def display(self, image):
        return image

if __name__ == '__main__':
    data = DspritesDataset()
    while True:
        batch = data.get_train_iter(batch_size=100)[0]
        print(batch.shape)
        plt.imshow(data.display(batch[0]))
        plt.show()