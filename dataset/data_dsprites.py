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
        dataset_zip = np.load(self.db_path + '/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', encoding='latin1')
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

    def get_pos_test(self, x_start, x_goal, y_start, y_goal, shape_lst=[0, 1, 2]):
        assert x_goal <= self.nx, "Nx_pos must x_goal <= Nx_pos: actual Nx_pos= {}, x_goal = {}".format(self.nx, x_goal)
        assert y_goal <= self.ny, "Ny_pos must y_goal <= Ny_pos: actual Ny_pos= {}, y_goal = {}".format(self.ny, y_goal)
        latents_sampled = self._sample_latent(x_start, x_goal, y_start, y_goal, scale_start=0, scale_goal=self.ns,
                                              rot_start=0, rot_goal=self.nr, shape_lst=shape_lst)
        id = self._latent_to_index(latents_sampled)
        label_id = np.array([latents_sampled[:, 4], latents_sampled[:, 5]])
        return {'data': self._2d_to_1d(self.imgs[id]), 'label': label_id.T}

    def get_scale_test(self, scale_start, scale_goal, shape_lst=[0, 1, 2]):
        latents_sampled = self._sample_latent(x_start=0, x_goal=self.nx, y_start=0, y_goal=self.ny,
                                              scale_start=scale_start, scale_goal=scale_goal,
                                              rot_start=0, rot_goal=self.nr, shape_lst=shape_lst)
        id = self._latent_to_index(latents_sampled)
        label_id = np.array(latents_sampled[:, 2])
        return {'data': self._2d_to_1d(self.imgs[id]), 'label': label_id}

    def get_scale_half_test(self, scale_start, scale_goal, shape_lst=[0, 1, 2]):
        latents_sampled = self._sample_latent(x_start=int(self.nx / 4), x_goal=int(self.nx / 4 * 3),
                                              y_start=int(self.ny / 4), y_goal=int(self.ny / 4 * 3),
                                              scale_start=scale_start, scale_goal=scale_goal,
                                              rot_start=0, rot_goal=self.nr, shape_lst=shape_lst)
        id = self._latent_to_index(latents_sampled)
        label_id = np.array(latents_sampled[:, 2])
        return {'data': self._2d_to_1d(self.imgs[id]), 'label': label_id}

    def get_rot_test(self, rot_start, rot_goal, shape_lst=[0, 1, 2]):
        latents_sampled = self._sample_latent(x_start=0, x_goal=self.nx, y_start=0, y_goal=self.ny, scale_start=0,
                                              scale_goal=self.ns,
                                              rot_start=rot_start, rot_goal=rot_goal, shape_lst=shape_lst)
        id = self._latent_to_index(latents_sampled)
        label_id = np.array(latents_sampled[:, 3])
        return {'data': self._2d_to_1d(self.imgs[id]), 'label': label_id}

    def get_rot_half_test(self, rot_start, rot_goal, shape_lst=[0, 1, 2]):
        latents_sampled = self._sample_latent(x_start=int(self.nx / 4), x_goal=int(self.nx / 4 * 3),
                                              y_start=int(self.ny / 4), y_goal=int(self.ny / 4 * 3), scale_start=0,
                                              scale_goal=self.ns,
                                              rot_start=rot_start, rot_goal=rot_goal, shape_lst=shape_lst)
        id = self._latent_to_index(latents_sampled)
        label_id = np.array(latents_sampled[:, 3])
        return {'data': self._2d_to_1d(self.imgs[id]), 'label': label_id}

    def get_pos_single_test(self, x_start, x_goal, y_start, y_goal, shape_lst=[0, 1, 2]):
        assert x_goal <= self.nx, "Nx_pos must x_goal <= Nx_pos: actual Nx_pos= {}, x_goal = {}".format(self.nx, x_goal)
        assert y_goal <= self.ny, "Ny_pos must y_goal <= Ny_pos: actual Ny_pos= {}, y_goal = {}".format(self.ny, y_goal)
        latents_sampled = self._sample_latent(x_start, x_goal, y_start, y_goal, scale_start=2, scale_goal=3,
                                              rot_start=0, rot_goal=1, shape_lst=shape_lst)
        id = self._latent_to_index(latents_sampled)
        label_id = np.array([latents_sampled[:, 4], latents_sampled[:, 5]])
        return {'data': self._2d_to_1d(self.imgs[id]), 'label': label_id.T}

    def get_scale_single_test(self, scale_start, scale_goal, shape_lst=[0, 1, 2]):
        latents_sampled = self._sample_latent(x_start=15, x_goal=16, y_start=15, y_goal=16, scale_start=scale_start,
                                              scale_goal=scale_goal,
                                              rot_start=0, rot_goal=1, shape_lst=shape_lst)
        id = self._latent_to_index(latents_sampled)
        label_id = np.array(latents_sampled[:, 2])
        return {'data': self._2d_to_1d(self.imgs[id]), 'label': label_id}

    def get_rot_single_test(self, rot_start, rot_goal, shape_lst=[0, 1, 2]):
        latents_sampled = self._sample_latent(x_start=15, x_goal=16, y_start=15, y_goal=16, scale_start=2, scale_goal=3,
                                              rot_start=rot_start, rot_goal=rot_goal, shape_lst=shape_lst)
        id = self._latent_to_index(latents_sampled)
        label_id = np.array(latents_sampled[:, 3])
        return {'data': self._2d_to_1d(self.imgs[id]), 'label': label_id}

if __name__ == '__main__':
    data = DspritesDataset()
    while True:
        batch = data.get_train_iter(batch_size=100)[0]
        print(batch.shape)
        plt.imshow(data.display(batch[0]))
        plt.show()