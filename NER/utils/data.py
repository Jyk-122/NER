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

# def save_json(save_path, data, json_type):


def json_to_input(data_path, tag_to_ix, tag_type='BIO'):
    data = load_json(data_path)
    sentence = []
    label = []
    for d in data:
        _sentence = [w for w in d['text']]
        sentence.append(_sentence)
        if tag_type == 'BIO':
            sen_len = len(d['text'])
            _label = torch.full((1, sen_len), tag_to_ix['O'], dtype=torch.long)
            for entity in d['label']:
                dic = d['label'][entity]
                for obj in dic:
                    index = dic[obj]
                    _label[0][index[0][0]] = tag_to_ix['B_' + entity]
                    _label[0][index[0][0] + 1: index[0][1] + 1] = tag_to_ix['I_' + entity]
            label.append(_label)
            
    return sentence, label

            
tag_to_ix = {'START_TAG':0, 'STOP_TAG':1, 'O':2,
             'B_address':3, 'I_address':4,
             'B_book':5, 'I_book':6, 
             'B_company':7, 'I_company':8, 
             'B_game':9, 'I_game':10, 
             'B_government':11, 'I_government':12,
             'B_movie':13, 'I_movie':14, 
             'B_name':15, 'I_name':16, 
             'B_organization':17, 'I_organization':18, 
             'B_position':19, 'I_position':20, 
             'B_scene':21, 'I_scene':22}
# json_to_input('../data/test.json', tag_to_ix)


class NER_Dataset(Data.Dataset):
    def __init__(self, data, train=True):
        sentence, label = data
        self.x_data = sentence
        self.y_data = label
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
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


def hold_out_data(dataset, ratio):
    assert sum(ratio) == 1
    total_datalen = len(dataset)
    indices = list(range(total_datalen))

    if len(ratio) == 2:
        valid_indices = indices[-total_datalen * ratio[1]:]
        train_indices = [i for i in indices if i not in valid_indices]

        train_sampler = Data.SubsetRandomSampler(train_indices)
        valid_sampler = Data.SubsetRandomSampler(valid_indices)

        train_loader = Data.DataLoader(dataset, batch_size=1, sampler=train_sampler)
        valid_loader = Data.DataLoader(dataset, batch_size=1, sampler=valid_sampler)

        return train_loader, valid_loader

    if len(ratio) == 3:
        valid_indices = indices[-total_datalen * (ratio[1] + ratio[2]): -total_datalen * ratio[2]]
        test_indices = indices[-total_datalen * ratio[2]]
        non_train_indices = valid_indices + test_indices
        train_indices = [i for i in indices if i not in non_train_indices]

        train_sampler = Data.SubsetRandomSampler(train_indices)
        valid_sampler = Data.SubsetRandomSampler(valid_indices)
        test_sampler = Data.SubsetRandomSampler(test_indices)

        train_loader = Data.DataLoader(dataset, batch_size=1, sampler=train_sampler)
        valid_loader = Data.DataLoader(dataset, batch_size=1, sampler=valid_sampler)
        test_loader = Data.DataLoader(dataset, batch_size=1, sampler=test_sampler)

        return train_loader, valid_loader, test_loader


def gen_vocab(data_loader):
    word_to_ix = {'<UNK>': 0}
    for d in data_loader:
        sentence, label = d
        for word in sentence:
            w = word[0]
            if w not in word_to_ix:
                word_to_ix[w] = len(word_to_ix)
    return word_to_ix


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


def prepare_data(sentence, word_to_ix, freq_dict, label, cuda_device=0, if_train=True, random_z=0.8375):
    s_ix = []
    if if_train:
        for word in sentence:
            w = word[0]
            if freq_dict[w] > 2:
                prob = random_z / (random_z + freq_dict[w])
                random_num = np.random.uniform(0, 1)
                if random_num <= prob:
                    s_ix.append(word_to_ix['<UNK>'])
                else:
                    s_ix.append(word_to_ix[w])
            else:
                s_ix.append(word_to_ix[w])
    else:
        s_ix = [word_to_ix[w[0]] if w in word_to_ix else word_to_ix['<UNK>'] for w in sentence]

    # s_ix = [word_to_ix[w[0]] if w in word_to_ix else word_to_ix['<UNK>'] for w in sentence]
    s_ix = torch.LongTensor(s_ix).view(-1)
    targets = torch.LongTensor(label).view(-1)
    if torch.cuda.is_available():
        s_ix = s_ix.cuda(cuda_device)
        targets = targets.cuda(cuda_device)
    return s_ix, targets


if __name__ == '__main__':
    data = json_to_input('../data/train.json', tag_to_ix)
    NER_dataset = NER_Dataset(data)
    train_loader, valid_loader = kfold_cross_data(NER_dataset, 10, 0)
    word_to_ix = gen_vocab(train_loader)
    vocab_size = len(word_to_ix)
    print(vocab_size)
    embedding_dim = 128
    word_embeds = nn.Embedding(vocab_size, embedding_dim)

    for data in train_loader:
        s, l = data
        print(s, l)
        print(len(s), len(l[0][0]))
        s = sentence_to_ix(s, word_to_ix)
        print(s.shape)
        break


