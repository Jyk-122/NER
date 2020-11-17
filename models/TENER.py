import torch 
import torch.nn as nn
import torch.nn.functional as F

import math
from copy import deepcopy
from fastNLP.modules import ConditionalRandomField, allowed_transitions
from torchcrf import CRF

import utils.data as data


class RelativeEmbedding(nn.Module):
    def forward(self, input):
        """
        Input is expected to be of size [bsz x seqlen].
        """
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + seq_len
        if max_pos > self.origin_shift:
            # recompute/expand embeddings if needed
            weights = self.get_embedding(
                max_pos*2,
                self.embedding_dim,
                self.padding_idx,
            )
            weights = weights.to(self._float_tensor)
            del self.weights
            self.origin_shift = weights.size(0)//2
            self.register_buffer('weights', weights)

        positions = torch.arange(-seq_len, seq_len).to(input.device).long() + self.origin_shift  # 2*seq_len
        embed = self.weights.index_select(0, positions.long()).detach()
        return embed


class RelativeSinusoidalPositionalEmbedding(RelativeEmbedding):
    """
    This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1568):
        """
        :param embedding_dim: 每个位置的dimension
        :param padding_idx:
        :param init_size:
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        assert init_size % 2 == 0
        weights = self.get_embedding(
            init_size+1,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer('weights', weights)
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    def get_embedding(self, num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(-num_embeddings//2, num_embeddings//2, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        self.origin_shift = num_embeddings//2 + 1
        return emb


class RelativeMultiHeadAttn(nn.Module):
    def __init__(self, d_model, n_head, dropout, r_w_bias=None, r_r_bias=None, scale=False):
        """
        :param int d_model:
        :param int n_head:
        :param dropout: 对attention map的dropout
        :param r_w_bias: n_head x head_dim or None, 如果为dim
        :param r_r_bias: n_head x head_dim or None,
        :param scale:
        :param rel_pos_embed:
        """
        super().__init__()
        self.qkv_linear = nn.Linear(d_model, d_model * 3, bias=False)
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.dropout_layer = nn.Dropout(dropout)

        self.pos_embed = RelativeSinusoidalPositionalEmbedding(d_model//n_head, 0, 1200)

        if scale:
            self.scale = math.sqrt(d_model // n_head)
        else:
            self.scale = 1

        if r_r_bias is None or r_w_bias is None:  # Biases are not shared
            self.r_r_bias = nn.Parameter(nn.init.xavier_normal_(torch.zeros(n_head, d_model // n_head)))
            self.r_w_bias = nn.Parameter(nn.init.xavier_normal_(torch.zeros(n_head, d_model // n_head)))
        else:
            self.r_r_bias = r_r_bias  # r_r_bias就是v
            self.r_w_bias = r_w_bias  # r_w_bias就是u

    def forward(self, x, mask):
        """
        :param x: batch_size x max_len x d_model
        :param mask: batch_size x max_len
        :return:
        """

        batch_size, max_len, d_model = x.size()
        pos_embed = self.pos_embed(mask)  # l x head_dim

        qkv = self.qkv_linear(x)  # batch_size x max_len x d_model3
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)
        q = q.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        k = k.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        v = v.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)  # b x n x l x d

        rw_head_q = q + self.r_r_bias[:, None]
        AC = torch.einsum('bnqd,bnkd->bnqk', [rw_head_q, k])  # b x n x l x d, n是head

        D_ = torch.einsum('nd,ld->nl', self.r_w_bias, pos_embed)[None, :, None]  # head x 2max_len, 每个head对位置的bias
        B_ = torch.einsum('bnqd,ld->bnql', q, pos_embed)  # bsz x head  x max_len x 2max_len，每个query对每个shift的偏移
        E_ = torch.einsum('bnqd,ld->bnql', k, pos_embed)  # bsz x head x max_len x 2max_len, key对relative的bias
        BD = B_ + D_  # bsz x head x max_len x 2max_len, 要转换为bsz x head x max_len x max_len
        BDE = self._shift(BD) + self._transpose_shift(E_)
        attn = AC + BDE

        attn = attn / self.scale

        attn = attn.masked_fill(mask[:, None, None, :].eq(0), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout_layer(attn)
        v = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, max_len, d_model)  # b x n x l x d

        return v

    def _shift(self, BD):
        """
        类似
        -3 -2 -1 0 1 2
        -3 -2 -1 0 1 2
        -3 -2 -1 0 1 2
        转换为
        0   1  2
        -1  0  1
        -2 -1  0
        :param BD: batch_size x n_head x max_len x 2max_len
        :return: batch_size x n_head x max_len x max_len
        """
        bsz, n_head, max_len, _ = BD.size()
        zero_pad = BD.new_zeros(bsz, n_head, max_len, 1)
        BD = torch.cat([BD, zero_pad], dim=-1).view(bsz, n_head, -1, max_len)  # bsz x n_head x (2max_len+1) x max_len
        BD = BD[:, :, :-1].view(bsz, n_head, max_len, -1)  # bsz x n_head x 2max_len x max_len
        BD = BD[:, :, :, max_len:]
        return BD
    
    def _transpose_shift(self, E):
        """
        类似
          -3   -2   -1   0   1   2
         -30  -20  -10  00  10  20
        -300 -200 -100 000 100 200
        转换为
          0  -10   -200
          1   00   -100
          2   10    000
        :param E: batch_size x n_head x max_len x 2max_len
        :return: batch_size x n_head x max_len x max_len
        """
        bsz, n_head, max_len, _ = E.size()
        zero_pad = E.new_zeros(bsz, n_head, max_len, 1)
        # bsz x n_head x -1 x (max_len+1)
        E = torch.cat([E, zero_pad], dim=-1).view(bsz, n_head, -1, max_len)
        indice = (torch.arange(max_len)*2+1).to(E.device)
        E = E.index_select(index=indice, dim=-2).transpose(-1,-2)  # bsz x n_head x max_len x max_len

        return E


class MultiHeadAttn(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1, scale=False):
        """
        :param d_model:
        :param n_head:
        :param scale: 是否scale输出
        """
        super().__init__()
        assert d_model%n_head==0

        self.n_head = n_head
        self.qkv_linear = nn.Linear(d_model, 3*d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout_layer = nn.Dropout(dropout)

        if scale:
            self.scale = math.sqrt(d_model//n_head)
        else:
            self.scale = 1

    def forward(self, x, mask):
        """
        :param x: bsz x max_len x d_model
        :param mask: bsz x max_len
        :return:
        """
        batch_size, max_len, d_model = x.size()
        x = self.qkv_linear(x)
        q, k, v = torch.chunk(x, 3, dim=-1)
        q = q.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        k = k.view(batch_size, max_len, self.n_head, -1).permute(0, 2, 3, 1)
        v = v.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)

        attn = torch.matmul(q, k)  # batch_size x n_head x max_len x max_len
        attn = attn/self.scale
        attn.masked_fill_(mask=mask[:, None, None].eq(0), value=float('-inf'))

        attn = F.softmax(attn, dim=-1)  # batch_size x n_head x max_len x max_len
        attn = self.dropout_layer(attn)
        v = torch.matmul(attn, v)  # batch_size x n_head x max_len x d_model//n_head
        v = v.transpose(1, 2).reshape(batch_size, max_len, -1)
        v = self.fc(v)

        return v


class TransformerLayer(nn.Module):
    def __init__(self, d_model, self_attn, feedforward_dim, after_norm, dropout):
        """
        :param int d_model: 一般512之类的
        :param self_attn: self attention模块，输入为x:batch_size x max_len x d_model, mask:batch_size x max_len, 输出为
            batch_size x max_len x d_model
        :param int feedforward_dim: FFN中间层的dimension的大小
        :param bool after_norm: norm的位置不一样，如果为False，则embedding可以直接连到输出
        :param float dropout: 一共三个位置的dropout的大小
        """
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn = self_attn

        self.after_norm = after_norm

        self.ffn = nn.Sequential(nn.Linear(d_model, feedforward_dim),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(feedforward_dim, d_model),
                                 nn.Dropout(dropout))

    def forward(self, x, mask):
        """
        :param x: batch_size x max_len x hidden_size
        :param mask: batch_size x max_len, 为0的地方为pad
        :return: batch_size x max_len x hidden_size
        """
        residual = x
        if not self.after_norm:
            x = self.norm1(x)

        x = self.self_attn(x, mask)
        x = x + residual
        if self.after_norm:
            x = self.norm1(x)
        residual = x
        if not self.after_norm:
            x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        if self.after_norm:
            x = self.norm2(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, n_head, feedforward_dim, dropout, after_norm=True, attn_type='naive',
                 scale=False, dropout_attn=None, pos_embed=None):
        super().__init__()
        if dropout_attn is None:
            dropout_attn = dropout
        self.d_model = d_model

        if pos_embed is None:
            self.pos_embed = None
        elif pos_embed == 'sin':
            self.pos_embed = SinusoidalPositionalEmbedding(d_model, 0, init_size=1024)
        elif pos_embed == 'fix':
            self.pos_embed = LearnedPositionalEmbedding(1024, d_model, 0)

        if attn_type == 'transformer':
            self_attn = MultiHeadAttn(d_model, n_head, dropout_attn, scale=scale)
        elif attn_type == 'adatrans':
            self_attn = RelativeMultiHeadAttn(d_model, n_head, dropout_attn, scale=scale)

        self.layers = nn.ModuleList([TransformerLayer(d_model, deepcopy(self_attn), feedforward_dim, after_norm, dropout)
                       for _ in range(num_layers)])

    def forward(self, x, mask):
        """
        :param x: batch_size x max_len
        :param mask: batch_size x max_len. 有value的地方为1
        :return:
        """
        if self.pos_embed is not None:
            x = x + self.pos_embed(mask)

        for layer in self.layers:
            x = layer(x, mask)
        return x


def make_positions(tensor, padding_idx):
    """
    Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (
        torch.cumsum(mask, dim=1).type_as(mask) * mask
    ).long() + padding_idx


class SinusoidalPositionalEmbedding(nn.Module):
    """
    This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1568):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """
        Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input):
        """
        Input is expected to be of size [bsz x seqlen].
        """
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        if max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.to(self._float_tensor)

        positions = make_positions(input, self.padding_idx)
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            padding_idx: int,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)

    def forward(self, input):
        # positions: batch_size x max_len, 把words的index输入就好了
        positions = make_positions(input, self.padding_idx)
        return super().forward(positions)


def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.LongTensor(idxs).to(seq.device)


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class TENER(nn.Module):
    def __init__(self, tag_vocab, embed, num_layers, d_model, n_head, feedforward_dim, dropout,
                 after_norm=True, attn_type='adatrans',
                 fc_dropout=0.3, pos_embed=None, scale=False, dropout_attn=None, 
                 START_TAG='<SOS>', STOP_TAG='<EOS>', PAD_TAG = '<PAD>'):
        """
        :param tag_vocab: fastNLP Vocabulary
        :param embed: fastNLP TokenEmbedding
        :param num_layers: number of self-attention layers
        :param d_model: input size
        :param n_head: number of head
        :param feedforward_dim: the dimension of ffn
        :param dropout: dropout in self-attention
        :param after_norm: normalization place
        :param attn_type: adatrans, naive
        :param rel_pos_embed: position embedding的类型，支持sin, fix, None. relative时可为None
        :param bi_embed: Used in Chinese scenerio
        :param fc_dropout: dropout rate before the fc layer
        """
        super().__init__()

        self.embed = embed
        self.embed_size = self.embed.embedding_dim
        self.embed.weight.requires_grad = True

        self.in_fc = nn.Linear(self.embed_size, d_model)

        self.transformer = TransformerEncoder(num_layers, d_model, n_head, feedforward_dim, dropout,
                                              after_norm=after_norm, attn_type=attn_type,
                                              scale=scale, dropout_attn=dropout_attn,
                                              pos_embed=pos_embed)
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.out_fc = nn.Linear(d_model, len(tag_vocab))

        # trans = allowed_transitions(tag_vocab, include_start_end=True)
        # self.crf = ConditionalRandomField(len(tag_vocab), include_start_end_trans=True, allowed_transitions=trans)

        # self.crf = CRF(len(tag_vocab), batch_first=True)
        self.START_TAG = START_TAG
        self.STOP_TAG = STOP_TAG
        self.PAD_TAG = PAD_TAG
        self.tag_to_ix = dict([(v, k) for (k, v) in tag_vocab.items()])
        self.tagset_size = len(self.tag_to_ix)
        self.transitions = nn.Parameter(torch.randn(len(self.tag_to_ix), len(self.tag_to_ix)))
        self.transitions.data[self.tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, self.tag_to_ix[STOP_TAG]] = -10000


    # def _forward(self, chars, target=None, bigrams=None):
    #     mask = chars.ne(0)
    #     chars = self.embed(chars)

    #     chars = self.in_fc(chars)
    #     chars = self.transformer(chars, mask)
    #     chars = self.fc_dropout(chars)
    #     chars = self.out_fc(chars)
    #     logits = F.log_softmax(chars, dim=-1)

    #     if target is None:
    #         paths = self.crf.decode(logits, mask)
    #         return torch.LongTensor(paths).to(chars.device)
    #     else:
    #         loss = self.crf(logits, target, mask)
    #         return loss

    # def forward(self, chars, target, bigrams=None):
    #     return self._forward(chars, target, bigrams)

    # def predict(self, chars, bigrams=None):
    #     return self._forward(chars, target=None, bigrams=bigrams)

    # def _score_sentence(self, feats, tags):
    #     score = torch.zeros(1).to(feats.device)
    #     # tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]), tags])
    #     prefix = torch.LongTensor([self.tag_to_ix[self.START_TAG]]).to(tags.device)
    #     tags = torch.cat([prefix, tags])
    #     for i, feat in enumerate(feats):
    #         score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
    #     score = score + self.transitions[self.tag_to_ix[self.STOP_TAG], tags[-1]]
    #     return score

    def _score_sentence(self, feats, tags):
        score = torch.zeros(tags.shape[0]).to(feats.device)
        tags = torch.cat([torch.full([tags.shape[0],1], self.tag_to_ix[self.START_TAG]).long().to(feats.device), tags], dim=1)
        for i in range(feats.shape[1]):
            feat = feats[:, i, :]
            score = score + self.transitions[tags[:, i + 1], tags[:, i]] + feat[range(feat.shape[0]),tags[:, i + 1]]
        score = score + self.transitions[self.tag_to_ix[self.STOP_TAG], tags[:, -1]]
        return score
    
    # def _forward_alg(self, feats):
    #     init_alphas = torch.full((1, self.tagset_size), -10000.).to(feats.device)
    #     init_alphas[0][self.tag_to_ix[self.START_TAG]] = 0
        
    #     forward_var = init_alphas
        
    #     for feat in feats:
    #         alphas_t = []
    #         for next_tag in range(self.tagset_size):
    #             emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
    #             trans_score = self.transitions[next_tag].view(1, -1)
    #             next_tag_var = forward_var + trans_score + emit_score
    #             alphas_t.append(log_sum_exp(next_tag_var).view(1))
    #         forward_var = torch.cat(alphas_t).view(1, -1)
    #     terminal_var = forward_var + self.transitions[self.tag_to_ix[self.STOP_TAG]]
    #     alpha = log_sum_exp(terminal_var)
    #     return alpha
    
    def _forward_alg(self, feats):
        init_alphas = torch.full([feats.shape[0], self.tagset_size], -10000.).to(feats.device)
        init_alphas[:, self.tag_to_ix[self.START_TAG]] = 0.

        forward_var_list = []
        forward_var_list.append(init_alphas)
        for feat_index in range(feats.shape[1]):
            gamar_r_l = torch.stack([forward_var_list[feat_index]] * feats.shape[2]).transpose(0, 1)
            t_r1_k = torch.unsqueeze(feats[:, feat_index, :], 1).transpose(1, 2)  # +1
            aa = gamar_r_l + t_r1_k + torch.unsqueeze(self.transitions, 0)
            forward_var_list.append(torch.logsumexp(aa, dim=2))
        terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_ix[self.STOP_TAG]].repeat([feats.shape[0], 1])
        alpha = torch.logsumexp(terminal_var, dim=1)
        return alpha
    
    def _get_features(self, sentence):
        mask = sentence.ne(0)
        chars = self.embed(sentence)

        chars = self.in_fc(chars)
        chars = self.transformer(chars, mask)
        chars = self.fc_dropout(chars)
        chars = self.out_fc(chars)
        logits = F.log_softmax(chars, dim=-1)

        return logits

    # def _viterbi_decode(self, feats):
    #     backpointers = []
        
    #     init_vvars = torch.full((1, self.tagset_size), -10000.).to(feats.device)
    #     init_vvars[0][self.tag_to_ix[self.START_TAG]] = 0
        
    #     forward_var = init_vvars
    #     for feat in feats:
    #         bptrs_t = []
    #         viterbivars_t = []
    #         for next_tag in range(self.tagset_size):
    #             next_tag_var = forward_var + self.transitions[next_tag]
    #             best_tag_id = argmax(next_tag_var)
    #             bptrs_t.append(best_tag_id)
    #             viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
    #         forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
    #         backpointers.append(bptrs_t)
            
    #     terminal_var = forward_var + self.transitions[self.tag_to_ix[self.STOP_TAG]]
    #     best_tag_id = argmax(terminal_var)
    #     path_score = terminal_var[0][best_tag_id]
        
    #     best_path = [best_tag_id]
    #     for bptrs_t in reversed(backpointers):
    #         best_tag_id = bptrs_t[best_tag_id]
    #         best_path.append(best_tag_id)
        
    #     start = best_path.pop()
    #     assert start == self.tag_to_ix[self.START_TAG]
    #     best_path.reverse()
    #     return path_score, best_path
    
    def _viterbi_decode(self, feats):
        backpointers = []

        init_vvars = torch.full((1, self.tagset_size), -10000.).to(feats.device)
        init_vvars[0][self.tag_to_ix[self.START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var_list = []
        forward_var_list.append(init_vvars)

        for feat_index in range(feats.shape[0]):
            gamar_r_l = torch.stack([forward_var_list[feat_index]] * feats.shape[1])
            gamar_r_l = torch.squeeze(gamar_r_l)
            next_tag_var = gamar_r_l + self.transitions
            viterbivars_t, bptrs_t = torch.max(next_tag_var, dim=1)

            t_r1_k = torch.unsqueeze(feats[feat_index], 0)
            forward_var_new = torch.unsqueeze(viterbivars_t, 0) + t_r1_k

            forward_var_list.append(forward_var_new)
            backpointers.append(bptrs_t.tolist())

        terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_ix[self.STOP_TAG]]
        best_tag_id = torch.argmax(terminal_var).tolist()
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag_to_ix[self.START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, torch.LongTensor(best_path).to(feats.device)

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return torch.sum(forward_score - gold_score)
    
    def forward(self, sentence):
        batch_size = sentence.shape[0]
        feats = self._get_features(sentence)
        score = torch.zeros(batch_size, 1).to(sentence.device)
        tag_seq = torch.LongTensor(batch_size, sentence.shape[1]).to(sentence.device)
        for i in range(batch_size):
            f_score, f_tag_seq = self._viterbi_decode(feats[i])
            score[i] = f_score
            tag_seq[i] = f_tag_seq
        # score, tag_seq = self._viterbi_decode(feats)
        # tag_seq = torch.LongTensor(tag_seq).to(sentence.device)
        return score, tag_seq


if __name__ == '__main__':
    a = 0