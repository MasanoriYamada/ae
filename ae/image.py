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
import math
from logging import getLogger


def save_images_tile(x, filename, data):
    """ save image
    x.shape = (datasize, width, height) width and height get from data.data_shape
    x.shape = (datasize, channel, width, height) width and height get from data.data_shape

    """
    logger = getLogger(__name__)

    # check data size
    data_len = len(x)
    if data_len > 81:
        logger.info('data is too large data len = {}'.format(data_len))
    # calc raw and col
    raw = math.ceil(math.sqrt(data_len))
    col = data_len // raw
    logger.debug('data len = {}'.format(data_len))
    logger.debug('output raw = {}, col = {}'.format(raw,col))
    if data.data_shape[-1] == 3:
        color_is = True
        data_shape = (data.data_shape[0], data.data_shape[1], data.data_shape[2])
    elif data.data_shape[-1] == 1:
        color_is = False
        data_shape = (data.data_shape[0], data.data_shape[1])
    logger.debug('color is = {}'.format(color_is))

    fig, ax = plt.subplots(col, raw, figsize=(9, 9), dpi=100)
    for ai, xi in zip(ax.flatten(), x):
        # image rescale to 0~1
        xi = data.display(xi)
        ai.imshow(xi.reshape(data_shape), shape=data_shape) if color_is else ai.imshow(xi.reshape(data_shape), shape=data_shape)
    fig.savefig(filename)
    plt.close()