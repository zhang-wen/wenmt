import math
import torch as tc
import torch.nn as nn
import torch.nn.functional as F

from tools.utils import PAD, MAX_SEQ_SIZE, wlog

'''
Implements the sinusoidal positional encoding for non-recurrent neural networks
Args:
   dropout (float): dropout parameter
   n_embed (int): embedding size
'''
class PositionalEncoding(nn.Module):

    def __init__(self, dropout_prob, n_embed, max_len=MAX_SEQ_SIZE):

        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = tc.zeros(max_len, n_embed)
        position = tc.arange(0., max_len).unsqueeze(1)
        div_term = tc.exp(tc.arange(0., n_embed, 2) * -(math.log(10000.0) / n_embed))
        # keep dim 0 for padding token position encoding zero vector
        #inter_term = position.float() * div_term
        #pe[1:, 0::2] = tc.sin(inter_term)[1:]
        #pe[1:, 1::2] = tc.cos(inter_term)[1:]
        # [5000, 1] * [256] = [5000, 256] 
        pe[:, 0::2] = tc.sin(position * div_term)
        pe[:, 1::2] = tc.cos(position * div_term)
        pe = pe.unsqueeze(0)    # [5000, 512] -> [1, 5000, 512]
        self.register_buffer('pe', pe)
        self.n_embed = n_embed

        wlog('\t pe: {}'.format(pe.size()))

        self.dropout = None
        if dropout_prob is not None and 0. < dropout_prob <= 1.0:
            wlog('\t with emb dropout prob = {} ...'.format(dropout_prob))
            self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, emb):

        emb = emb * math.sqrt(self.n_embed)
        emb = emb + self.pe[:, :emb.size(1)]
        if self.dropout is not None: emb = self.dropout(emb)

        return emb

class WordEmbedding(nn.Module):

    def __init__(self,
                 n_vocab,
                 n_embed=512,
                 emb_dropout=0.,
                 position_encoding=False,
                 prefix='WordEmbedding'):

        super(WordEmbedding, self).__init__()
        wlog('WordEmbedding_{}'.format(prefix))
        self.position_encoding = position_encoding
        self.we = nn.Embedding(n_vocab, n_embed, padding_idx=PAD)
        self.n_vocab = n_vocab
        nn.init.normal_(self.we.weight, mean=0, std=n_embed ** -0.5)
        wlog('*Normal init word embedding weight {}'.format(self.we.weight.size()))
        nn.init.constant_(self.we.weight[PAD], 0)
        self.n_embed = n_embed
        if position_encoding is True:
            wlog('with position emb ...')
            self.spe = PositionalEncoding(emb_dropout, n_embed)
            self.emb_dropout = emb_dropout

    def forward(self, x):

        x_w_emb = self.we(x)
        if self.position_encoding is True:
            x_wp_emb = self.spe(x_w_emb)
        else:
            x_wp_emb = x_w_emb

        #x_wp_emb = F.dropout(x_wp_emb, p=self.emb_dropout, training=self.training)

        return x_w_emb, x_wp_emb






















