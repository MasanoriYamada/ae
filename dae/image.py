#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('Pdf')
import matplotlib.pyplot as plt

def save_images(x, filename, data):

    if data.data_shape[0] == 3:
        color_is = True
    elif data.data_shape[0] == 1:
        color_is = False
    fig, ax = plt.subplots(3, 3, figsize=(9, 9), dpi=100)
    for ai, xi in zip(ax.flatten(), x):
        ai.imshow(xi.reshape(data.data_shape)) if color_is else ai.imshow(xi.reshape((data.data_shape[1], data.data_shape[2])))
    fig.savefig(filename)
    plt.close()