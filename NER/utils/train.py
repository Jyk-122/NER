import torch
import torch.nn as nn
import torch.optim as optim
import tensorboardX
from tensorboardX import  SummaryWriter

import os
import logging
import time
from tqdm import tqdm

import utils.data as data
import utils.configuration as configuration
import models.BiLSTM_CRF as bilstm_crf


def save_model(model, path):
    torch.save(model, path)


def load_model(path, cuda_device):
    if torch.cuda.is_available():
        model = torch.load(path, map_location=lambda storage, loc: storage.cuda(cuda_device))
    else:
        model = torch.load(path)
    model.eval()
    return model


def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def load_checkpoint(model, optimizer, path, cuda_device=0):
    checkpoint = torch.load(path)
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict'])
    if torch.cuda.is_available():
        model = model.cuda(cuda_device)
    return model, optimizer, epoch, loss


def solver(config, model, train_loader, valid_loader, optimizer):
    if os.path.exists(config.checkpoint_save_path):
        model, optimizer, epoch_current, loss = load_checkpoint(model, optimizer, config.checkpoint_save_path)

    word_to_ix = data.gen_vocab(train_loader)
    freq_dict = data.freq_count(train_loader)

    writer = SummaryWriter('runs/' + config.train_log_filename)
    global_train_step = 0

    for epoch in range(config.epochs_current, config.epochs_total):
        total_train_loss = 0
        total_train_accu = 0
        total_valid_loss = 0
        total_valid_accu = 0

        for d in tqdm(train_loader):
            model.zero_grad()

            sentence, label = d
            sentence_in, targets = data.prepare_data(sentence=sentence, word_to_ix=word_to_ix,
                                                     freq_dict=freq_dict, label=label,
                                                     cuda_device=config.cuda_device, if_train=True)

            loss = model.neg_log_likelihood(sentence_in, targets)
            print(loss.item())

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                score, pred_seq = model(sentence_in)
                accu = torch.sum(targets == pred_seq).item() / targets.numel()

            global_train_step += 1
            writer.add_scalar('train_step_loss', loss.item(), global_step=global_train_step)
            writer.add_scalar('train_step_accu', accu, global_step=global_train_step)
            total_train_loss += loss.item()
            total_train_accu += accu

        writer.add_scalar('train_epoch_loss', total_train_loss / len(train_loader), global_step=epoch)
        writer.add_scalar('train_epoch_accu', total_train_accu / len(train_loader), global_step=epoch)

        save_checkpoint(model, optimizer, epoch, total_train_loss / len(train_loader), config.checkpoint_save_path)

        with torch.no_grad():
            for i, d in enumerate(valid_loader):
                sentence, label = d
                sentence_in, targets = data.prepare_data(sentence=sentence, word_to_ix=word_to_ix,
                                                         freq_dict=freq_dict, label=label,
                                                         cuda_device=config.cuda_device, if_train=False)

                loss = model.neg_log_likelihood(sentence_in, targets)
                score, pred_seq = model(sentence_in)
                accu = torch.sum(targets == pred_seq).item() / targets.numel()

                total_valid_loss += loss
                total_valid_accu += accu
            writer.add_scalar('valid_epoch_loss', total_valid_loss / len(valid_loader), global_step=epoch)
            writer.add_scalar('valid_epoch_accu', total_valid_accu / len(valid_loader), global_step=epoch)


def train(config):
    # parameters adjustments and records
    tag_to_ix = config.tag_to_ix
    START_TAG = config.START_TAG
    STOP_TAG = config.STOP_TAG
    config.train_log_filename = 'Log_' + config.model_type + '_train'
    config.valid_log_filename = 'Log_' + config.model_type + '_valid'

    # processing data
    train_data = data.json_to_input(config.train_data_path, tag_to_ix)
    NER_dataset = data.NER_Dataset(train_data)
    print('Data is loaded.')

    if config.train_type == 'cross_validation':
        k = config.k_fold
        for ith in range(k):
            train_loader, valid_loader = data.kfold_cross_data(NER_dataset, config.k_fold, ith)
            word_to_ix = data.gen_vocab(train_loader)

            model = bilstm_crf.BiLSTM_CRF(word_to_ix, tag_to_ix, config.embedding_dim, config.hidden_dim)
            if torch.cuda.is_available():
                model.cuda(config.cuda_device)

            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, betas=[config.beta1, config.beta2],
                                   eps=1e-8)

            save_name = config.model_type + '_' + time.strftime('%Y-%m-%d-%H-%M', time.localtime())
            config.checkpoint_save_path = 'checkpoints/' + save_name + '.tar'

            solver(config, model, train_loader, valid_loader, optimizer)

            configuration.save_config(config, 'checkpoints/' + save_name + '.json')

    elif config.train_type == 'hold_out':
        if len(config.ratio) == 2:
            train_loader, valid_loader = data.hold_out_data(NER_dataset, config.ratio)
        elif len(config.ratio) == 3:
            train_loader, valid_loader, test_loader = data.hold_out_data(NER_dataset, config.ratio)

        word_to_ix = data.gen_vocab(train_loader)
        model = bilstm_crf.BiLSTM_CRF(word_to_ix, tag_to_ix, config.embedding_dim, config.hidden_dim)
        if torch.cuda.is_available():
            model.cuda(config.cuda_device)

        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, betas=[config.beta1, config.beta2],
                               eps=1e-8)

        save_name = config.model_type + '_' + time.strftime('%Y-%m-%d-%H-%M', time.localtime())
        config.checkpoint_save_path = 'checkpoints/' + save_name + '.tar'

        solver(config, model, train_loader, valid_loader, optimizer)

        configuration.save_config(config, 'checkpoints/' + save_name + '.json')


if __name__ == '__main__':
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    tag_to_ix = {START_TAG: 0, STOP_TAG: 1, 'O': 2,
                 'B_address': 3, 'I_address': 4,
                 'B_book': 5, 'I_book': 6,
                 'B_company': 7, 'I_company': 8,
                 'B_game': 9, 'I_game': 10,
                 'B_government': 11, 'I_government': 12,
                 'B_movie': 13, 'I_movie': 14,
                 'B_name': 15, 'I_name': 16,
                 'B_organization': 17, 'I_organization': 18,
                 'B_position': 19, 'I_position': 20,
                 'B_scene': 21, 'I_scene': 22}

    dataset = data.json_to_input('../data/train.json', tag_to_ix)
    NER_dataset = data.NER_Dataset(dataset)
    train_loader, valid_loader = data.kfold_cross_data(NER_dataset, 5, 0)
    word_to_ix = data.gen_vocab(train_loader)
    vocab_size = len(word_to_ix)
    print(vocab_size)
    embedding_dim = 128
    word_embeds = nn.Embedding(vocab_size, embedding_dim)

    for i, d in enumerate(valid_loader):
        s, l = d
        for i in s:
            if i[0] not in word_to_ix:
                print(i[0])
