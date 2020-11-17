#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import tensorboardX
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR


import os
import logging
import time
import numpy as np

import utils.data as data
import utils.configuration as configuration
import utils.train as train
import models.BiLSTM_CRF as bilstm_crf
import models.TENER as tener
import utils.scheduler as scheduler


def tagseq_to_dict(seq, tagseq, tag_vocab):
    tagseq = tagseq.squeeze().numpy()
    res_dict = {'label':{}}
    for i in range(len(tagseq)):
        ix = tagseq[i]
        if tag_vocab[ix] == 'O':
            continue
        if tag_vocab[ix][0] == 'B':
            tag = tag_vocab[ix][2:]
            if tag not in res_dict['label']:
                res_dict['label'][tag] = {}
            start = i
            j = i
            while j + 1 < len(tagseq) and tag_vocab[tagseq[j + 1]][0] == 'I':
                j += 1
            end = j
            tag_pair = dict({seq[start: end + 1]:[[start, end]]})
            entity = seq[start: end + 1]
            if entity not in res_dict['label'][tag]:
                res_dict['label'][tag][entity] = []
            res_dict['label'][tag][entity].append([start, end])
    return res_dict


            


if __name__ == '__main__':

    # config = configuration.gen_config()

    # train.train(config)

    # model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
    #
    # optimizer = optim.Adam(model, lr=1e-4, eps=1e-8, betas=[0.9, 0.999])
    #
    # # lr_sch = CosineAnnealingLR(optimizer, T_max=90, eta_min=1e-5)
    # lr_sch = StepLR(optimizer, 10, gamma=0.5)
    # scheduler_warmup = scheduler.GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=lr_sch)
    #
    # optimizer.zero_grad()
    # optimizer.step()
    # scheduler_warmup.step()
    # lr = []
    # for epoch in range(100):
    #     current_lr = optimizer.param_groups[0]['lr']
    #     print(current_lr)
    #     lr.append(current_lr)
    #     optimizer.step()
    #     scheduler_warmup.step()

    config = configuration.load_config('checkpoints/TENER_2020-11-15-12-49.json')
    train_data, word_to_ix, _, train_raw_sen, train_raw_lab = data.json_to_input('data/train.json', config.tag_to_ix, ratio=[0.8, 0.2], if_padding=False, pre_trained_emb_path='data/gigaword_chn.all.a2b.uni.ite50.vec')
    test_data, _, _, test_raw_sen, test_raw_lab = data.json_to_input('data/test.json', config.tag_to_ix, ratio=[0, 1], if_padding=False, pre_trained_emb_path='data/gigaword_chn.all.a2b.uni.ite50.vec', pre_word_to_ix=word_to_ix)
    tag_vocab = dict([(v, k) for (k, v) in config.tag_to_ix.items()])
    # model = tener.TENER(tag_vocab=tag_vocab, embed=word_embeds,
    #                 num_layers=config.num_layers, d_model=config.d_model, 
    #                 n_head=config.n_heads, feedforward_dim=config.dim_feedforward, 
    #                 dropout=config.dropout, after_norm=config.after_norm, attn_type=config.attn_type, 
    #                 fc_dropout=config.fc_dropout, pos_embed=config.pos_embed,
    #                 scale=False, dropout_attn=None)
    model = train.load_model('checkpoints/TENER_2020-11-15-12-49.pth', 0)
    word_embeds = model.embed

    sentence, label = test_data
    pred_label = []
    for i in range(len(sentence)):
        seq = torch.LongTensor(sentence[i]).view(1, 1, -1)
        tag = torch.LongTensor(label[i]).view(1, 1, -1)
        raw_s = test_raw_sen[i]
        raw_l = test_raw_lab[i]
        seq, tag = data.prepare_data(seq, tag, next(model.parameters()).device)
        _, pred_seq = model(seq)
        res_dict = tagseq_to_dict(raw_s, pred_seq, tag_vocab)
        pred_label.append(res_dict)
    
    data.save_json('data/test_pred.json', pred_label)




