#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)

START_TAG = "<START>"
STOP_TAG = "<STOP>"


def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.LongTensor(idxs)


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):
    def __init__(self, word_to_ix, tag_to_ix, embedding_dim, hidden_dim, use_gpu=True, cuda_device=0):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = len(word_to_ix)
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.use_gpu = use_gpu
        self.cuda_device = cuda_device
        
        self.word_embeds = nn.Embedding(len(word_to_ix), embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000
        
        self.hidden = self.init_hidden()
    
    def init_hidden(self):
        if self.use_gpu:
            return (torch.randn(2, 1, self.hidden_dim // 2).cuda(self.cuda_device),
                    torch.randn(2, 1, self.hidden_dim // 2).cuda(self.cuda_device))
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))
    
    def _forward_alg(self, feats):
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        if self.use_gpu:
            init_alphas = init_alphas.cuda(self.cuda_device)
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0
        
        forward_var = init_alphas
        
        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tagset_size):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha
    
    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats
    
    def _score_sentence(self, feats, tags):
        score = torch.zeros(1)
        # tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]), tags])
        prefix = torch.LongTensor([self.tag_to_ix[START_TAG]])
        if self.use_gpu:
            score = score.cuda(self.cuda_device)
            prefix = prefix.cuda(self.cuda_device)
        tags = torch.cat([prefix, tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score
    
    def _viterbi_decode(self, feats):
        backpointers = []
        
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0
        if self.use_gpu:
            init_vvars = init_vvars.cuda(self.cuda_device)
        
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []
            viterbivars_t = []
            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)
            
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]
        
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]
        best_path.reverse()
        return path_score, best_path
    
    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score
    
    def forward(self, sentence):
        lstm_feats = self._get_lstm_features(sentence)
        score, tag_seq = self._viterbi_decode(lstm_feats)
        if self.use_gpu:
            tag_seq = torch.LongTensor(tag_seq).cuda(self.cuda_device)
        return score, tag_seq


if __name__ == '__main__':

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

    word_to_ix = {'a':0, 'b':1, 'c':2, 'd':2,'e':2,'f':2,'g':2,'h':2,'i':2,'j':2,}

    model = BiLSTM_CRF(word_to_ix, tag_to_ix=tag_to_ix, embedding_dim=128, hidden_dim=128, use_gpu=False)
    x = torch.randint(0, 10, (30,))
    y = torch.randint(0, 20, (30,))
    word_embeds = nn.Embedding(10, 128)
    # x = word_embeds(x)
    loss = model.neg_log_likelihood(x, y)

