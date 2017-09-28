#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Denoising convolutional auto encoder in mnist
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
from datetime import datetime

import chainer
from tensorboard import SummaryWriter

import dataset.data_mnist
import dataset.data_celeba
import dataset.data_dsprites
from ae.net import cae
from ae import image
from ae import util


def main():

    # parametor
    epoch_num = 100
    batch_size = 128
    ana_freq = 1
    gpu = -1

    # set logger
    logging.config.fileConfig('./log/log.conf')
    logger = getLogger(__name__)

    logger.info('file = {}'.format(__file__))
    logger.info('epoch_num = {}'.format(epoch_num))
    logger.info('batch_size = {}'.format(batch_size))
    logger.info('ana_freq = {}'.format(ana_freq))
    logger.info('gpu = {}'.format(gpu))

    # set writer
    writer = SummaryWriter('results/' + datetime.now().strftime('%B%d  %H:%M:%S'))

    # plot_dict
    plt_tuple = (
    ('epoch', 0),
    ('test_rec_x', []),
    ('test_x', []),
    ('head', 'results/'),)
    plt_dict = OrderedDict(plt_tuple)

    # read data
    global data_obj
    data_obj = dataset.data_dsprites.DspritesDataset(db_path='/Users/yamada/lab/dat/dsprites')
    # data_obj = dataset.data_celeba.CelebADataset(db_path='./dataset/celebA', data_size=10000)
    # data_obj = dataset.data_mnist.MnistDataset()
    data_obj.train_size = 100 # adjust train data size for speed
    data_obj.test_size = 9

    # model and optimizer
    model = cae.CAE(data_obj)
    opt = chainer.optimizers.Adam()
    opt.setup(model)

    # gpu setup
    xp = np
    if gpu >= 0:
        import cupy
        xp = cupy
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    tmp_loss = model(chainer.Variable(np.random.rand(1, data_obj.total_dim).astype(np.float32)))
    writer.add_graph([tmp_loss])

    # Learning loop
    for epoch in range(epoch_num):
        train_iter = data_obj.get_train_iter(batch_size, noise_type='salt')
        total_loss = 0.
        for i, (x, t) in enumerate(train_iter):
            x = chainer.Variable(xp.array(x, dtype=xp.float32))
            opt.update(model.get_loss_func(), x)
            local_loss = model.loss * len(x.data)
            total_loss += local_loss
            logger.debug('{}/{} in epoch = {} , train local loss = {}'.format(i, len(train_iter), epoch, local_loss.data / batch_size))
        logger.info('epoch = {}, train loss = {}'.format(epoch, total_loss.data/batch_size))
        writer.add_all_parameter_histograms([total_loss], epoch, pattern='.*CAE.*')
        writer.add_scalar('train_loss', total_loss.data/batch_size, epoch)

        # evaluate
        if epoch % ana_freq == 0:
            with chainer.using_config('train', False):
                plt_dict['epoch'] = epoch
                plt_dict['test_rec_x'] = []
                plt_dict['test_x'] = []
                test_iter = data_obj.get_test_iter(batch_size=9, noise_type=None)
                for x, t in test_iter:
                    x = chainer.Variable(xp.array(x, dtype=xp.float32))
                    rec_x = model(x)
                    plt_dict['test_rec_x'].append(rec_x.data)
                    x = x.reshape(
                        (-1, data_obj.data_shape[0], data_obj.data_shape[1], data_obj.data_shape[2]))
                    plt_dict['test_x'].append(x.data)
                if gpu >=0:
                    plt_dict = util.dict_to_cpu(plt_dict)
                analysis(plt_dict)
    writer.close()

def analysis(plt_dict):
    rec_x = plt_dict['test_rec_x']
    image.save_images_tile(rec_x[0], plt_dict['head'] + '/rec_x_{}.pdf'.format(plt_dict['epoch']), data_obj)

    test_x = plt_dict['test_x']
    image.save_images_tile(test_x[0], plt_dict['head'] + '/test_x_{}.pdf'.format(plt_dict['epoch']), data_obj)

if __name__ == '__main__':
    main()