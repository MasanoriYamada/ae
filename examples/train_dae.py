#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Denoising auto encoder in mnist
"""

__author__ = "Masanori Yamada"
__date__ = "30 Jul 2017"

from collections import OrderedDict

import numpy as np
import six

import chainer

import dataset.data_mnist
from dae.net import dae
from dae import image


def main():

    # parametor
    epoch_num = 100
    batch_size = 128
    ana_freq = 10
    xp = np

    # plot_dict
    plt_tuple = (('test_rec_x',[]),
                 ('test_x', []),
                 ('head', 'results/'),)
    plt_dict = OrderedDict(plt_tuple)

    # read data
    global data_obj
    data_obj = dataset.data_mnist.MnistDataset()

    # model and optimizer
    model = dae.DAE()
    opt = chainer.optimizers.Adam()
    opt.setup(model)

    # Learning loop
    for epoch in six.moves.range(epoch_num):
        train_iter = data_obj.get_train_iter(batch_size)
        total_loss = 0.
        for x, t in train_iter:
            x = chainer.Variable(np.array(x, dtype=xp.float32))
            opt.update(model.get_loss_func(), x)
            local_loss = model.loss.data * len(x.data)
            total_loss += local_loss

        print(total_loss/batch_size)
        # evaluate
        if epoch % ana_freq == 0:
            with chainer.using_config('train', False):
                test_iter = data_obj.get_test_iter(size=9)
                for x, t in test_iter:
                    rec_x = model(x)
                    plt_dict['test_rec_x'].append(rec_x.data)
                    plt_dict['test_x'].append(rec_x.data)
                analysis(plt_dict)

def analysis(plt_dict):
    rec_x = plt_dict['test_rec_x']
    image.save_images(rec_x[0], plt_dict['head'] + '/rec_x.pdf', data_obj)

    test_x = plt_dict['test_x']
    image.save_images(test_x[0], plt_dict['head'] + '/test_x.pdf', data_obj)




if __name__ == '__main__':
    main()