#!/usr/bin/env python
# -*- coding: utf-8 -*-
import chainer
from chainer import cuda
from chainer import serializers
import chainer.computational_graph
import time
import os
import logging, logging.handlers

def dict_to_cpu(x_dict):
    """from gpu to cpu in dict(only ndarray)
    :param dict:
    :return dict:
    """
    for key in x_dict:
        tmp = x_dict[key]
        if isinstance(tmp, cuda.cupy.ndarray):
            # print("debug done cupy {}\t{}".format(tmp, key))
            x_dict[key] = chainer.cuda.to_cpu(tmp)
        elif isinstance(tmp, dict):
            # print("debug done dict {}\t{}".format(tmp, key))
            x_dict[key] = dict_to_cpu(tmp)
        elif isinstance(tmp, list):
            # print("debug done list {}\t{}".format(tmp, key))
            x_dict[key] = list_to_cpu(tmp)
        else:
            # print("debug done other {}\t{}".format(tmp, key))
            x_dict[key] = tmp
    return x_dict

def list_to_cpu(x_lst):
    """from gpu to cpu in lst(only ndarray)
    :param lst:
    :return lst:
    """
    for id, data in enumerate(x_lst):
        if isinstance(data, cuda.cupy.ndarray):
            x_lst[id] = chainer.cuda.to_cpu(data)
        elif isinstance(data, dict):
            x_lst[id] = dict_to_cpu(data)
        elif isinstance(data, list):
            x_lst[id] = list_to_cpu(data)
        else:
            x_lst[id] = data
    return x_lst

def model_save(path, file_name,  model, optimizer):
    if not os.path.exists(path):
        os.makedirs(path)
    # Save the model and the optimizer
    # print('save the model')
    serializers.save_npz(path + '/'+ file_name + '.model', model)
    # print('save the optimizer')
    serializers.save_npz(path + '/' + file_name + '.state', optimizer)

class Timer(object):
    """Timer  object  write now and diff"""
    def __init__(self,file):
        self.file_name = file
        self.time_lst = []
        self.time_diff_lst = []
        if os.path.exists(self.file_name):
            os.remove(self.file_name)

    def detect_time(self, counter):
        self.time_lst.append(time.time())
        if len(self.time_lst) < 2:
            with open(self.file_name, 'a') as time_out:
                time_out.write("{}\t{}\n".format(counter, self.time_lst[-1]))
        else:
            self.time_diff_lst.append(self.time_lst[-1] - self.time_lst[-2])
            with open(self.file_name, 'a') as time_out:
                time_out.write("{}\t{}\t{}\n".format(counter, self.time_lst[-1], self.time_diff_lst[-1]))

def split_int(x):
    a = x // 2
    b = x - a
    return (a, b)

def warm_up(max_kl, current_epoch, max_epoch=500.):
    beta =  (current_epoch / max_epoch) * max_kl
    return beta if max_kl >= beta else max_kl

def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)
    wrapper.has_run = False
    return wrapper

@run_once
def graph_save(target_path_dict):
    """save computation graph. for example
    util.graph_save(
    {head + "total_graph.dot":model.loss,
    head + "kl_graph.dot": model.kl_loss,
    head + "rec_graph.dot": model.rec_loss})
    """
    for path in target_path_dict:
        with open(path, 'w') as o:
            variable_style = {'shape': 'octagon', 'fillcolor': '#E0E0E0',
                              'style': 'filled'}
            function_style = {'shape': 'record', 'fillcolor': '#6495ED',
                              'style': 'filled'}
            g = chainer.computational_graph.build_computational_graph(
                (target_path_dict[path], ),
                variable_style=variable_style,
                function_style=function_style,
                rankdir='BT')
            o.write(g.dump())
    print('graph generated')