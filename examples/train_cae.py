#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Convolutional auto encoder in mnist
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
from ae.trainer import Trainer


def main():

    # parametor
    epoch_num = 30
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

    # read data
    # data_obj = dataset.data_dsprites.DspritesDataset(db_path='~/lab/dat/dsprites')
    # data_obj = dataset.data_celeba.CelebADataset(db_path='./dataset/celebA', data_size=10000)
    data_obj = dataset.data_mnist.MnistDataset()
    data_obj.train_size = 80#00  # adjust train data size for speed
    data_obj.test_size = 9

    # model and optimizer
    model = cae.CAE(data_obj)
    opt = chainer.optimizers.Adam()

    trainer = Trainer(model=model, optimizer=opt, writer=writer, gpu=gpu)
    trainer.fit(data_obj, epoch_num=epoch_num, batch_size=batch_size, ana_freq=ana_freq)


if __name__ == '__main__':
    main()