#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Auto encoder in mnist
"""
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

from collections import OrderedDict
import numpy as np
import logging.config
from logging import getLogger

import chainer

import dataset.data_mnist
import dataset.data_celeba
from ae.net import ae
from ae import image
from ae import util


def main():

    # parametor
    epoch_num = 100
    batch_size = 128
    ana_freq = 10
    gpu = -1

    # set logger
    logging.config.fileConfig('./log/log.conf')
    logger = getLogger(__name__)

    logger.info('file = {}'.format(__file__))
    logger.info('epoch_num = {}'.format(epoch_num))
    logger.info('batch_size = {}'.format(batch_size))
    logger.info('ana_freq = {}'.format(ana_freq))
    logger.info('gpu = {}'.format(gpu))

    # plot_dict
    plt_tuple = (
        ('epoch', 0),
        ('test_rec_x',[]),
        ('test_x', []),
        ('head', 'results/'),)
    plt_dict = OrderedDict(plt_tuple)

    # read data
    global data_obj
    # data_obj = dataset.data_celeba.CelebADataset(db_path='./dataset/celebA', data_size=1000)
    data_obj = dataset.data_mnist.MnistDataset()
    data_obj.train_size = 200  # adjust train data size for speed
    data_obj.test_size = 9

    # model and optimizer
    model = ae.AE(data_obj)
    opt = chainer.optimizers.Adam()
    opt.setup(model)

    # gpu setup
    xp = np
    if gpu >= 0:
        import cupy
        xp = cupy
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    # Learning loop
    for epoch in range(epoch_num):
        train_iter = data_obj.get_train_iter(batch_size)
        total_loss = 0.
        for i, (x, t) in enumerate(train_iter):
            x = chainer.Variable(xp.array(x, dtype=xp.float32))
            opt.update(model.get_loss_func(), x)
            local_loss = model.loss.data * len(x.data)
            total_loss += local_loss
            logger.debug('{}/{} in epoch = {} , train local loss = {}'.format(i, len(train_iter), epoch,
                                                                              local_loss / batch_size))
        logger.info('epoch = {}, train loss = {}'.format(epoch, total_loss/batch_size))
        print('epoch = {}, train loss = {}'.format(epoch, total_loss/batch_size))
        # evaluate
        if epoch % ana_freq == 0:
            with chainer.using_config('train', False):
                plt_dict['epoch'] = epoch
                plt_dict['test_rec_x'] = []
                plt_dict['test_x'] = []
                test_iter = data_obj.get_test_iter()
                for x, t in test_iter:
                    x = chainer.Variable(xp.array(x, dtype=xp.float32))
                    rec_x = model(x)
                    plt_dict['test_rec_x'].append(rec_x.data)
                    plt_dict['test_x'].append(x.data)
                if gpu >=0:
                    plt_dict = util.dict_to_cpu(plt_dict)
                analysis(plt_dict)

def analysis(plt_dict):
    rec_x = plt_dict['test_rec_x']
    image.save_images_tile(rec_x[0], plt_dict['head'] + '/rec_x_{}.pdf'.format(plt_dict['epoch']), data_obj)

    test_x = plt_dict['test_x']
    image.save_images_tile(test_x[0], plt_dict['head'] + '/test_x_{}.pdf'.format(plt_dict['epoch']), data_obj)

if __name__ == '__main__':
    main()