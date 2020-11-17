import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.optim.sgd import SGD
import tensorboardX
from tensorboardX import  SummaryWriter

import os
import logging
import time
from tqdm import tqdm

import utils.data as data
import utils.configuration as configuration
import models.BiLSTM_CRF as bilstm_crf
import models.TENER as tener
import utils.scheduler as scheduler


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


def solver(config, model, train_loader, valid_loader, optimizer, sch_warmup=None):
    if os.path.exists(config.checkpoint_save_path):
        model, optimizer, epoch_current, loss = load_checkpoint(model, optimizer, config.checkpoint_save_path)
        print("Pre-trained models loaded.")

    # freq_dict = data.freq_count(train_loader)

    writer = SummaryWriter('runs/' + config.train_log_filename)
    global_train_step = 0
    global_valid_accu = 0

    if sch_warmup is not None:
        optimizer.zero_grad()
        optimizer.step()
        sch_warmup.step()

    for epoch in range(config.epochs_current, config.epochs_total):
        model.train()
        total_train_loss = 0
        total_train_accu = 0
        total_valid_loss = 0
        total_valid_accu = 0

        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('lr', current_lr, global_step=epoch)
        print(epoch, current_lr)

        for d in tqdm(train_loader):
            model.zero_grad()

            sentence, label = d
            sentence_in, targets = data.prepare_data(sentence=sentence, label=label,
                                                     device=next(model.parameters()).device)

            loss = model.neg_log_likelihood(sentence_in, targets)
            # print( loss.item())

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

        if sch_warmup is not None:
            sch_warmup.step()

        writer.add_scalar('train_epoch_loss', total_train_loss / len(train_loader), global_step=epoch)
        writer.add_scalar('train_epoch_accu', total_train_accu / len(train_loader), global_step=epoch)

        save_checkpoint(model, optimizer, epoch, total_train_loss / len(train_loader), config.checkpoint_save_path)

        with torch.no_grad():
            model.eval()

            for i, d in enumerate(valid_loader):
                sentence, label = d
                sentence_in, targets = data.prepare_data(sentence=sentence, label=label,
                                                         device=next(model.parameters()).device)

                loss = model.neg_log_likelihood(sentence_in, targets)
                score, pred_seq = model(sentence_in)
                accu = torch.sum(targets == pred_seq).item() / targets.numel()

                total_valid_loss += loss
                total_valid_accu += accu

            writer.add_scalar('valid_epoch_loss', total_valid_loss / len(valid_loader), global_step=epoch)
            writer.add_scalar('valid_epoch_accu', total_valid_accu / len(valid_loader), global_step=epoch)
        
        if total_valid_accu > global_valid_accu:
            save_model(model, config.model_save_path)
            global_valid_accu = total_valid_accu


def train(config):
    # parameters adjustments and records
    tag_to_ix = config.tag_to_ix
    tag_vocab = dict([(v, k) for (k, v) in tag_to_ix.items()])
    START_TAG = config.START_TAG
    STOP_TAG = config.STOP_TAG

    # processing data
    total_data, word_to_ix, word_embeds, _ = data.json_to_input(config.train_data_path, tag_to_ix,
                                                             tag_type=config.tag_type, emb_dim=config.embedding_dim,
                                                             if_padding=config.if_padding, ratio=config.ratio,
                                                             pre_trained_emb_path='data/gigaword_chn.all.a2b.uni.ite50.vec')
    NER_dataset = data.NER_Dataset(total_data)
    print('Data is loaded.')

    if config.train_type == 'cross_validation':
        k = config.k_fold
        for ith in range(k):
            train_loader, valid_loader = data.kfold_cross_data(NER_dataset, config.k_fold, ith)

            model = bilstm_crf.BiLSTM_CRF(word_to_ix, tag_to_ix, config.embedding_dim, config.hidden_dim)
            if torch.cuda.is_available():
                model.cuda(config.cuda_device)

            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, betas=[config.beta1, config.beta2],
                                   eps=1e-8)

            save_name = config.model_type + '_' + time.strftime('%Y-%m-%d-%H-%M', time.localtime())
            config.checkpoint_save_path = 'checkpoints/' + save_name + '.tar'
            config.model_save_path = 'checkpoints/' + save_name + '.pth'
            config.train_log_filename = 'Log_train_' + save_name
            config.valid_log_filename = 'Log_valid_' + save_name

            configuration.save_config(config, 'checkpoints/' + save_name + '.json')

            solver(config, model, train_loader, valid_loader, optimizer)

    elif config.train_type == 'hold_out':
        if len(config.ratio) == 2:
            train_loader, valid_loader = data.hold_out_data(NER_dataset, config.ratio, config.batch_size)
        elif len(config.ratio) == 3:
            train_loader, valid_loader, test_loader = data.hold_out_data(NER_dataset, config.ratio, config.batch_size)

        # model = bilstm_crf.BiLSTM_CRF(word_to_ix, tag_to_ix, config.embedding_dim, config.hidden_dim)
        model = tener.TENER(tag_vocab=tag_vocab, embed=word_embeds,
                            num_layers=config.num_layers, d_model=config.d_model, 
                            n_head=config.n_heads, feedforward_dim=config.dim_feedforward, 
                            dropout=config.dropout, after_norm=config.after_norm, attn_type=config.attn_type, 
                            fc_dropout=config.fc_dropout, pos_embed=config.pos_embed,
                            scale=False, dropout_attn=None)
        if torch.cuda.is_available():
            model.cuda(config.cuda_device)

        # optimizer = SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, betas=[config.beta1, config.beta2], eps=1e-8)
        
        # lr_sch = CosineAnnealingLR(optimizer, T_max=config.epochs_total - 10, eta_min=1e-5)
        # scheduler_warmup = scheduler.GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=lr_sch)

        lr_sch = StepLR(optimizer, 10, gamma=0.5)
        scheduler_warmup = scheduler.GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=lr_sch)

        save_name = config.model_type + '_' + time.strftime('%Y-%m-%d-%H-%M', time.localtime())
        config.checkpoint_save_path = 'checkpoints/' + save_name + '.tar'
        config.model_save_path = 'checkpoints/' + save_name + '.pth'
        config.train_log_filename = 'Log_train_' + save_name
        config.valid_log_filename = 'Log_valid_' + save_name

        configuration.save_config(config, 'checkpoints/' + save_name + '.json')

        solver(config, model, train_loader, valid_loader, optimizer, scheduler_warmup)


if __name__ == '__main__':
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
