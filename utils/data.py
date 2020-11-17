#!/usr/bin/env python
# coding: utf-8

import os
import json
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.data as Data


def load_json(data_path):
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_json(save_path, data):
    with open(save_path, 'a+', encoding='utf-8') as json_file:
        for d in data:
            json_str = json.dumps(d, ensure_ascii=False) + '\n'
            json_file.write(json_str)



def json_to_input(data_path, tag_to_ix, tag_type='BIO', emb_dim=128, if_padding=True, ratio=[0.8, 0.2],
                  pre_trained_emb_path='../data/gigaword_chn.all.a2b.uni.ite50.vec', pre_word_to_ix=None):
    if pre_word_to_ix is None:
        word_to_ix = {'<PAD>': 0, '<UNK>': 1}
    else:
        word_to_ix = pre_word_to_ix
    pre_emb = {}
    pre_emb_dim = emb_dim
    if pre_trained_emb_path is not None and pre_word_to_ix is None:
        with open(pre_trained_emb_path, 'r') as f:
            for line in f.readlines():
                line = line.split()
                char = line[0]
                emb = list(map(float, line[1:]))
                pre_emb[char] = emb
                pre_emb_dim = len(emb)
                word_to_ix[char] = len(word_to_ix)
    
    data = load_json(data_path)
    raw_sen = []
    raw_lab = []
    sentence = []
    label = []
    max_len = 0
    for i, d in enumerate(data):
        _sentence = [w for w in d['text']]
        raw_sen.append(d['text'])
        raw_lab.append(d['label'])
        _sen_ix = []
        for w in _sentence:
            if i < len(data) * ratio[0]:
                if w not in word_to_ix:
                    word_to_ix[w] = len(word_to_ix)
            if w in word_to_ix:
                _sen_ix.append(word_to_ix[w])
            else:
                _sen_ix.append(word_to_ix['<UNK>'])
        sentence.append(_sen_ix)
        sen_len = len(d['text'])
        max_len = max(sen_len, max_len)
        if tag_type == 'BIO':
            _label = [tag_to_ix['O']] * sen_len
            for entity in d['label']:
                dic = d['label'][entity]
                for obj in dic:
                    index = dic[obj]
                    for ind in index:
                        _label[ind[0]] = tag_to_ix['B_' + entity]
                        _label[ind[0] + 1: ind[1] + 1] = [tag_to_ix['I_' + entity]] * (ind[1] - ind[0])
            label.append(_label)

    # for i in range(10):
    #     print(data[i], sentence[i], label[i])
    # print('over')

    if if_padding:
        for i in range(len(sentence)):
            sentence[i] = sentence[i] + [word_to_ix['<PAD>']] * (max_len - len(sentence[i]))
            label[i] = label[i] + [tag_to_ix['<PAD>']] * (max_len - len(label[i]))

    emb = nn.Embedding(len(word_to_ix), pre_emb_dim)
    for char in pre_emb:
        emb.weight.data[word_to_ix[char]] = torch.Tensor(pre_emb[char])

    return (sentence, label), word_to_ix, emb, raw_sen, raw_lab


class NER_Dataset(Data.Dataset):
    def __init__(self, data, train=True):
        sentence, label = data
        self.x_data = sentence
        self.y_data = label
    
    def __getitem__(self, index):
        return torch.LongTensor(self.x_data[index]).unsqueeze(0), \
               torch.LongTensor(self.y_data[index]).unsqueeze(0)
    
    def __len__(self):
        return len(self.x_data)


def kfold_cross_data(dataset, k, ith):
    assert k > 1
    total_datalen = len(dataset)
    kfold_datalen = total_datalen // k
    indices = list(range(total_datalen))
    valid_indices = indices[ith*kfold_datalen: (ith + 1)*kfold_datalen]
    train_indices = [i for i in indices if i not in valid_indices]
    
    train_sampler = Data.SubsetRandomSampler(train_indices)
    valid_sampler = Data.SubsetRandomSampler(valid_indices)
    
    train_loader = Data.DataLoader(dataset, batch_size=1, sampler=train_sampler)
    valid_loader = Data.DataLoader(dataset, batch_size=1, sampler=valid_sampler)
    
    return train_loader, valid_loader


def hold_out_data(dataset, ratio, batch_size=1):
    assert sum(ratio) == 1
    total_datalen = len(dataset)
    indices = list(range(total_datalen))

    if len(ratio) == 2:
        valid_indices = indices[int(total_datalen * (1 - ratio[1])):]
        train_indices = [i for i in indices if i not in valid_indices]

        train_sampler = Data.SubsetRandomSampler(train_indices)
        valid_sampler = Data.SubsetRandomSampler(valid_indices)

        train_loader = Data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        valid_loader = Data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

        return train_loader, valid_loader

    if len(ratio) == 3:
        valid_indices = indices[-total_datalen * (ratio[1] + ratio[2]): -total_datalen * ratio[2]]
        test_indices = indices[-total_datalen * ratio[2]]
        non_train_indices = valid_indices + test_indices
        train_indices = [i for i in indices if i not in non_train_indices]

        train_sampler = Data.SubsetRandomSampler(train_indices)
        valid_sampler = Data.SubsetRandomSampler(valid_indices)
        test_sampler = Data.SubsetRandomSampler(test_indices)

        train_loader = Data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        valid_loader = Data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
        test_loader = Data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

        return train_loader, valid_loader, test_loader


# def gen_vocab_emb(data_loader, emb_dim, pre_trained_emb_path='data/gigaword_chn.all.a2b.uni.ite50.vec'):
#     word_to_ix = {'<PAD>': 0, '<UNK>': 1}
#     pre_emb = {}
#     pre_emb_dim = emb_dim
#     if pre_trained_emb_path is not None:
#         with open(pre_trained_emb_path, 'r') as f:
#             for line in f.readlines():
#                 line = line.split()
#                 char = line[0]
#                 emb = list(map(float, line[1:]))
#                 pre_emb[char] = emb
#                 pre_emb_dim = len(emb)
#                 word_to_ix[char] = len(word_to_ix)
#
#     for d in data_loader:
#         sentence, label = d
#         for word in sentence:
#             w = word[0]
#             if w not in word_to_ix:
#                 word_to_ix[w] = len(word_to_ix)
#
#     emb = nn.Embedding(len(word_to_ix), pre_emb_dim)
#     for char in pre_emb:
#         emb.weight.data[word_to_ix[char]] = torch.Tensor(pre_emb[char])
#
#     return word_to_ix, emb


def freq_count(data_loader):
    freq_dict = {}
    for d in data_loader:
        sentence, label = d
        for word in sentence:
            w = word[0]
            freq_dict[w] = freq_dict[w] + 1 if w in freq_dict else 1
    return freq_dict


def sentence_to_ix(sentence, word_to_ix):
    s_ix = [word_to_ix[w[0]] if w in word_to_ix else word_to_ix['<UNK>'] for w in sentence]
    if torch.cuda.is_available():
        return torch.LongTensor(s_ix).cuda()
    return torch.LongTensor(s_ix)


def prepare_data(sentence, label, device, if_train=True, random_z=0.8375):
    # s_ix = []
    # if if_train:
    #     for word in sentence:
    #         w = word[0]
    #         if freq_dict[w] > 2:
    #             prob = random_z / (random_z + freq_dict[w])
    #             random_num = np.random.uniform(0, 1)
    #             if random_num <= prob:
    #                 s_ix.append(word_to_ix['<UNK>'])
    #             else:
    #                 s_ix.append(word_to_ix[w])
    #         else:
    #             s_ix.append(word_to_ix[w])
    # else:
    #     s_ix = [word_to_ix[w[0]] if w in word_to_ix else word_to_ix['<UNK>'] for w in sentence]

    seq = sentence.squeeze(1).to(device)
    lab = label.squeeze(1).to(device)
    return seq, lab


if __name__ == '__main__':
    data = json_to_input('../data/train.json', tag_to_ix)



