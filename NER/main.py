#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import tensorboardX
from tensorboardX import SummaryWriter

import os
import logging
import time

import utils.data as data
import utils.configuration as configuration
import utils.train as train
import models.BiLSTM_CRF as bilstm_crf


if __name__ == '__main__':
    config = configuration.gen_config()

    train.train(config)






