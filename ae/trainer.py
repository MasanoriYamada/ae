#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import numpy as np
import copy
import chainer
from logging import getLogger
from ae import image
from ae import util


class Trainer(object):
    def __init__(self, writer, model, optimizer, gpu=-1):
        self.logger = getLogger(__name__)
        self.writer = writer
        self.model = model
        self.best_model = None
        self.opt = optimizer
        self.best_opt = None
        self.gpu = gpu
        self.best_loss = float('inf')

        # gpu setup
        self.xp = np
        if self.gpu >= 0:
            import cupy
            self.xp = cupy
            chainer.cuda.get_device_from_id(gpu).use()
            self.model.to_gpu()  # Copy the model to the GPU

    def fit(self, data_obj, epoch_num=100, batch_size=128, ana_freq=1, noise_type=None):
        '''
        :param data_obj: input data
        :param epoch_num:
        :param batch_size:
        :param gpu: -1 is no use gpu
        :param noise_type: None is no use noise, 'salt', 'gauss'
        :return:
        '''

        self.opt.setup(self.model)
        tmp_loss = self.model(chainer.Variable(np.random.rand(1, data_obj.total_dim).astype(np.float32)))
        self.writer.add_graph([tmp_loss])

        # Learning loop
        for epoch in range(epoch_num):
            if noise_type is None:
                train_iter = data_obj.get_train_iter(batch_size)
            elif noise_type == 'salt':
                train_iter = data_obj.get_train_iter(batch_size, noise_type='salt')
            elif noise_type == 'gauss':
                train_iter = data_obj.get_train_iter(batch_size, noise_type='gauss')
            else:
                self.logger.critical('noise_type error: {}'.format(noise_type))

            total_loss = 0.
            for i, (x, t) in enumerate(train_iter):
                x = chainer.Variable(self.xp.array(x, dtype=self.xp.float32))
                self.opt.update(self.model.get_loss_func(), x)
                local_loss = self.model.loss * len(x.data)
                total_loss += local_loss
                self.logger.debug('{}/{} in epoch = {} , train local loss = {}'.format(i, len(train_iter), epoch,
                                                                                  local_loss.data / batch_size))
            self.logger.info('epoch = {}, train loss = {}'.format(epoch, total_loss.data / batch_size))
            # writer.add_all_variable_images([total_loss], pattern='.*AE.*')
            self.writer.add_all_parameter_histograms([total_loss], epoch)
            self.writer.add_scalar('train_loss', total_loss.data / batch_size, epoch)
            if total_loss.data < self.best_loss:
                self.best_model = copy.deepcopy(self.model)
                self.best_opt = copy.deepcopy(self.opt)
                self.best_loss = total_loss.data
                self.logger.debug('model update')
            if epoch % ana_freq == 0:
                self.eval(data_obj, epoch, noise_type)
        self.writer.close()
        return self.best_model

    def eval(self, data_obj, epoch, noise_type=None):
        plt_dict = {}
        plt_dict['head'] = './results/'
        plt_dict['epoch'] = epoch
        plt_dict['test_rec_x'] = []
        plt_dict['test_x'] = []
        # evaluate
        with chainer.using_config('train', False):
            if noise_type is None:
                test_iter = data_obj.get_test_iter()
            elif noise_type == 'salt':
                test_iter = data_obj.get_test_iter(noise_type='salt')
            elif noise_type == 'gauss':
                test_iter = data_obj.get_test_iter(noise_type='gauss')
            else:
                self.logger.critical('noise_type error: {}'.format(noise_type))

            for x, t in test_iter:
                x = chainer.Variable(self.xp.array(x, dtype=self.xp.float32))
                rec_x = self.best_model(x)
                plt_dict['test_rec_x'].append(rec_x.data)
                plt_dict['test_x'].append(x.data)

            if self.gpu >= 0:
                plt_dict = util.dict_to_cpu(plt_dict)
                self.logger.debug('plt_dict gpu to cpu')

            rec_x = plt_dict['test_rec_x']
            image.save_images_tile(rec_x[0], plt_dict['head'] + '/rec_x_{}.pdf'.format(plt_dict['epoch']), data_obj)

            test_x = plt_dict['test_x']
            image.save_images_tile(test_x[0], plt_dict['head'] + '/test_x_{}.pdf'.format(plt_dict['epoch']), data_obj)



