from __future__ import division, print_function

import copy
import numpy as np
import torch.nn as nn
from tools.utils import *
from tools.utils import PAD
import torch.nn.functional as F
from models.nn_utils import LayerNorm

'''
We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending
to subsequent positions. This masking, combined with fact that the output embeddings are offset by
one position, ensures that the predictions for position i can depend only on the known outputs at
positions less than i.

print(subsequent_mask(6))
tensor([[[ True, False, False, False, False, False],
         [ True,  True, False, False, False, False],
         [ True,  True,  True, False, False, False],
         [ True,  True,  True,  True, False, False],
         [ True,  True,  True,  True,  True, False],
         [ True,  True,  True,  True,  True,  True]]])

'''
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return tc.from_numpy(subsequent_mask) == 0

def make_std_mask(tgt, pad):
    "Create a mask to hide padding and future words."
    tgt_mask = (tgt != pad).unsqueeze(-2)
    #tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
    #    tgt_mask.data).to(tc.device('cuda', tgt_mask.get_device()))
    tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).cuda(tgt_mask.get_device())
    return tgt_mask


'''
In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully
connected feed-forward network, which is applied to each position separately and identically.
This consists of two linear transformations with a ReLU activation in between.
    FFN(x) = max(0,x * W_1+b_1) * W_2 + b_2

While the linear transformations are the same across different positions, they use different
parameters from layer to layer. Another way of describing this is as two convolutions with kernel
size 1. The dimensionality of input and output is d_model=512, and the inner-layer has dimensionality
d_ff=2048.
'''
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff, bias=True)
        self.w_2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

'''
    Args:
        d_model(int): the dimension of keys/values/queries in
                      MultiHeadAttention, also the input size of
                      the first-layer of the PositionwiseFeedForward.
        n_head(int): the number of head for MultiHeadAttention.
        hidden_size(int): the second-layer of the PositionwiseFeedForward.
        droput(float): dropout probability(0-1.0).
'''

'''
the output of each sub-layer is LayerNorm(x+Sublayer(x)), where Sublayer(x) is the function
implemented by the sub-layer itself. We apply dropout to the output of each sub-layer, before
it is added to the sub-layer input and normalized.

To facilitate these residual connections, all sub-layers in the model, as well as the embedding
layers, produce outputs of dimension d_model=512.
'''
class SelfAttSublayer(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SelfAttSublayer, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
        #sublayer_out, attn = sublayer(self.norm(x))
        #return x + self.dropout(sublayer_out), attn

def clones(module, N):
    "Produce N identical layers."
    wlog('clones -> {}'.format(N))
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class SelfAttEncoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, src_emb, layer, N):
        super(SelfAttEncoder, self).__init__()
        self.src_emb = src_emb
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask=None):
        _, x = self.src_emb(x)
        if mask is not None: mask = mask.unsqueeze(-2)
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x, att = layer(x, mask=mask)
        return self.norm(x), att

'''
Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is
a simple, position-wise fully connected feed-forward network.
'''
class SelfAttEncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(SelfAttEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SelfAttSublayer(size, dropout), 2)
        self.size = size

    def forward(self, x, mask=None):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask=mask))
        return self.sublayer[1](x, self.feed_forward), self.self_attn.attn


'''
The decoder is also composed of a stack of N=6 identical layers.
'''
class SelfAttDecoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, trg_emb, layer, N):
        super(SelfAttDecoder, self).__init__()
        self.trg_emb = trg_emb
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask=None):
        if src_mask is not None: src_mask = src_mask.unsqueeze(-2)
        tgt_mask = make_std_mask(x, PAD)
        _, x = self.trg_emb(x)
        for layer in self.layers:
            x, trg_slf_attn, trg_src_attn = layer(x, memory, src_mask=src_mask, tgt_mask=tgt_mask)
        return self.norm(x), trg_slf_attn, trg_src_attn


'''
In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer,
which performs multi-head attention over the output of the encoder stack. Similar to the encoder, we
employ residual connections around each of the sub-layers, followed by layer normalization.
'''
class SelfAttDecoderLayer(nn.Module):

    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(SelfAttDecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SelfAttSublayer(size, dropout), 3)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward), self.self_attn.attn, self.src_attn.attn


from models.nn_utils import MyLogSoftmax
epsilon = 1e-20
class Generator(nn.Module):

    "Define standard linear + softmax generation step."
    def __init__(self, d_model, trg_word_emb):
        super(Generator, self).__init__()
        self.vcb_proj = nn.Linear(d_model, trg_word_emb.n_vocab, bias=True)
        if wargs.proj_share_weight is False:
            nn.init.normal_(self.vcb_proj.weight, mean=0, std=d_model ** -0.5)
            wlog('*Normal init vcb_proj weight {}'.format(self.vcb_proj.weight.size()))
        if wargs.proj_share_weight is True:
            assert d_model == trg_word_emb.n_embed, '{}, {}'.format(d_model, trg_word_emb.n_embed)
            wlog('Copying weights of target word embedding into Generator')
            self.vcb_proj.weight = trg_word_emb.we.weight
        nn.init.zeros_(self.vcb_proj.bias)
        wlog('*Zero init vcb_proj bias {}'.format(self.vcb_proj.bias.size()))
        self.log_softmax = MyLogSoftmax()

    def pred_map(self, logit, noise=None):

        logit = self.vcb_proj(logit)

        if noise is not None:
            with tc.no_grad():
                logit.data.add_( -tc.log(-tc.log(tc.tensor(
                    logit.size()).cuda().uniform_(0, 1) + epsilon) + epsilon) ) / noise

        return logit

    def forward(self, model_out, training=True):
        pred_BLV = self.vcb_proj(model_out)
        if training is True: return self.log_softmax(pred_BLV, dim=-1)
        else: return pred_BLV

class BowMapper(nn.Module):

    "Define standard linear + softmax generation step."
    def __init__(self, d_model, trg_word_emb):
        super(BowMapper, self).__init__()
        self.bow_vcb_proj = nn.Linear(d_model, trg_word_emb.n_vocab, bias=True)
        nn.init.normal_(self.bow_vcb_proj.weight, mean=0, std=d_model ** -0.5)
        nn.init.zeros_(self.bow_vcb_proj.bias)
        wlog('*Normal init map_vocab weight {}'.format(self.bow_vcb_proj.weight.size()))
        #nn.init.uniform_(self.bow_vcb_proj.weight, a=-0.08, b=0.08)
        #nn.init.uniform_(self.bow_vcb_proj.bias, a=-0.08, b=0.08)
        #wlog('*Uniform init bow_vcb_proj weight {}'.format(self.bow_vcb_proj.weight.size()))

    def forward(self, context):
        return self.bow_vcb_proj(context)





