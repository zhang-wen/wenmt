from __future__ import division
import torch as tc
import torch.nn as nn
import wargs
#from gru import LGRU, TGRU
from tools.utils import *

'''
    Bi-directional Transition Gated Recurrent Unit network encoder
    input args:
        src_emb:        class WordEmbedding
        enc_hid_size:   the size of TGRU hidden state
        n_layers:       layer nubmer of encoder
'''
class StackedGRUEncoder(nn.Module):

    def __init__(self,
                 src_emb,
                 enc_hid_size=512,
                 dropout_prob=0.3,
                 n_layers=2,
                 bidirectional=True,
                 prefix='GRU_Encoder', **kwargs):

        super(StackedGRUEncoder, self).__init__()

        self.src_word_emb = src_emb
        n_embed = src_emb.n_embed
        self.enc_hid_size = enc_hid_size
        f = lambda name: str_cat(prefix, name)  # return 'Encoder_' + parameters name

        '''
        self.bigru = nn.GRU(input_size=n_embed, hidden_size=self.enc_hid_size,
                            num_layers=n_layers, bias=True, batch_first=True,
                            dropout=dropout_prob, bidirectional=bidirectional)
        self.bidirectional = bidirectional
        '''
        self.n_layers = n_layers
        self.layer_stack = nn.ModuleList([nn.GRUCell(n_embed, enc_hid_size, bias=True)])
        self.layer_stack.extend([
            nn.GRUCell(enc_hid_size, enc_hid_size, bias=True)
            for _ in range(n_layers - 1)])

    def forward(self, xs, xs_mask=None):

        if xs.dim() == 3: xs_e = xs
        else: x_w_e, xs_e = self.src_word_emb(xs)
        batch_size, src_L = xs_e.size(0), xs_e.size(1)

        inputs = xs_e.transpose(0, 1)
        for i, enc_layer in enumerate(self.layer_stack):
            # xs_e: (batch_size, L_src, n_embed)
            h = tc.zeros(batch_size, self.enc_hid_size, requires_grad=False)
            if wargs.gpu_id is not None and not h.is_cuda: h = h.cuda()
            outputs = []
            for Lidx in range(src_L):
                h = enc_layer(inputs[Lidx], h)
                outputs.append(h)
            inputs = outputs[::-1]

        if self.n_layers % 2 == 0:
            outputs = tc.stack(outputs[::-1]).transpose(0, 1)
        else:
            outputs = tc.stack(outputs).transpose(0, 1)
        #self.bigru.flatten_parameters()
        #if self.bidirectional is False:
        #    h0 = tc.zeros(batch_size, self.enc_hid_size, requires_grad=False)
        #else:
        #    h0 = tc.zeros(2, batch_size, self.enc_hid_size, requires_grad=False)
        #print xs_e.size(), h0.size()
        #output, hn = self.bigru(xs_e, h0)
        #output, hn = self.bigru(xs_e)

        return outputs * xs_mask[:, :, None]

