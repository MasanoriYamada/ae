#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import os
from glob import glob
import numpy as np
import scipy.misc as misc
import matplotlib
matplotlib.use('Pdf')
import matplotlib.pyplot as plt


from dataset import dataset

class CelebADataset(dataset.Dataset):
    def __init__(self, db_path="~/lab/dat/celebA", crop=True, data_size=None):
        super(CelebADataset, self).__init__()

        self.name = 'celebA'
        self.data_shape = [64, 64, 3]
        self.batch_data_shape = (-1, 64, 64, 3)
        self.width = 64
        self.height = 64
        self.range = [-1., 1.]
        self.total_dim = 3*64*64
        self.is_crop = crop
        self.data_files = glob(os.path.join(db_path, "*.png"))
        self.data_files = self.data_files[:data_size] if data_size is not None else self.data_files
        if len(self.data_files) < 10: # 100000:
            print("Only %d images found for celebA, is this right?" % len(self.data_files))
            exit(-1)
        self.x_data = self.set_train_test_data()
        self.label_data = np.zeros(shape=(len(self.data_files),1))

        self.train_array = None
        self.test_array = None
        self.train_size = -1
        self.test_size = -1
        self.split_train_test(split_rate=0.8)

    def set_train_test_data(self):
        sample_files = self.data_files
        sample = [self.get_image(sample_file, self.is_crop) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)
        return sample_images

    @staticmethod
    def get_image(image_path, is_crop=True):
        image = CelebADataset.transform(misc.imread(image_path).astype(np.float), is_crop=is_crop)
        return image

    @staticmethod
    def center_crop(x, crop_h, crop_w=None, resize_w=64):
        if crop_w is None:
            crop_w = crop_h
        h, w = x.shape[:2]
        j = int(round((h - crop_h) / 2.))
        i = int(round((w - crop_w) / 2.))
        return misc.imresize(x[j:j + crop_h, i:i + crop_w],
                                   [resize_w, resize_w])

    @staticmethod
    def full_crop(x):
        if x.shape[0] <= x.shape[1]:
            lb = int((x.shape[1] - x.shape[0]) / 2)
            ub = lb + x.shape[0]
            x = misc.imresize(x[:, lb:ub], [64, 64])
        else:
            lb = int((x.shape[0] - x.shape[1]) / 2)
            ub = lb + x.shape[1]
            x = misc.imresize(x[lb:ub, :], [64, 64])
        return x

    @staticmethod
    def transform(image, npx=108, is_crop=True, resize_w=64):
        # npx : # of pixels width/height of image
        if is_crop:
            cropped_image = CelebADataset.center_crop(image, npx, resize_w=resize_w)
        else:
            cropped_image = CelebADataset.full_crop(image)
        return np.array(cropped_image) / 127.5 - 1.
    """ Transform image to displayable to 0~1"""
    def display(self, image):
        rescaled = np.divide(image + 1.0, 2.0)
        return np.clip(rescaled, 0.0, 1.0)

if __name__ == '__main__':

    data = CelebADataset()
    while True:
        batch = data.get_train_iter(batch_size=100)[0]
        print(batch.shape)
        plt.imshow(data.display(batch[0]))
        plt.show()