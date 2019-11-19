from __future__ import division, print_function

import sys
import math
import numpy as np
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from .nn_utils import MaskSoftmax, Linear
np.set_printoptions(threshold=sys.maxsize)

class Additive_Attention(nn.Module):

    def __init__(self, dec_hid_size, align_size):

        super(Additive_Attention, self).__init__()
        self.sa = nn.Linear(dec_hid_size, align_size)
        self.tanh = nn.Tanh()
        self.maskSoftmax = MaskSoftmax()
        self.a1 = nn.Linear(align_size, 1)

    def forward(self, s_tm1, xs_h, uh, xs_mask=None):

        e_ij = self.a1( self.tanh(self.sa(s_tm1)[:, None, :] + uh) ).squeeze(-1)

        e_ij = self.maskSoftmax(e_ij, mask=xs_mask, dim=1)  # (batch_size, key_len)
        # weighted sum of the h_j: (batch_size, enc_hid_size)
        attend = (e_ij[:, :, None] * xs_h).sum(1)

        return e_ij, attend

class Multihead_Additive_Attention(nn.Module):

    #dec_hid_size:   the dimension of n_head keys/values/queries: dec_hid_size % n_head == 0
    #n_head:    number of parallel heads.
    def __init__(self, enc_hid_size, dec_hid_size, n_head=8):

        super(Multihead_Additive_Attention, self).__init__()

        assert dec_hid_size % n_head == 0, 'dec_hid_size {} divided by n_head {}.'.format(dec_hid_size, n_head)
        self.n_head = n_head
        self.linear_query = Linear(dec_hid_size, dec_hid_size, bias=False)
        #self.mSoftMax = MaskSoftmax()
        dim_per_head = dec_hid_size // n_head
        self.a1 = Linear(dim_per_head, 1, bias=False)
        self.final_proj = Linear(2 * enc_hid_size, 2 * enc_hid_size, bias=True)

    '''
        Compute the context vector and the attention vectors.
        Args:
           q (FloatTensor): query [batch_size, dec_hid_size]             ->  hidden state
           v (FloatTensor): value [batch_size, key_len, 2*dec_hid_size]  ->  annotations
           k (FloatTensor): key [batch_size, key_len, dec_hid_size]      ->  uh
           attn_mask: binary mask indicating
                    which keys have non-zero attention [batch_size, key_len]
        Returns:
           (FloatTensor, FloatTensor) :
           * context vectors [batch_size, 2 * dec_hid_size]
           * probability            [batch_size, n_head, key_len]
    '''
    def forward(self, q, v, k, attn_mask=None):

        def split_heads(x, nhead):
            return x.view(x.size(0), x.size(1), nhead, x.size(-1) // nhead).permute(0, 2, 1, 3)

        def combine_heads(x, nhead):
            return x.permute(0, 2, 1, 3).contiguous().view(x.size(0), x.size(2), nhead * x.size(-1))

        q = self.linear_query(q)
        # 1. project key, value, and query
        q = split_heads(q[:, None, :], self.n_head) # [batch_size, n_head, 1, dim_per_head]
        k = split_heads(k, self.n_head)             # [batch_size, n_head, key_len, dim_per_head]

        hidden = tc.tanh(q + k)
        attn = self.a1(hidden).squeeze(-1)          # [batch_size, n_head, key_len]
        #print(attn_mask)
        #print(attn_mask.size())
        #print(attn_mask.dtype)
        if attn_mask is not None:   # [batch_size, key_len]
            attn_mask = attn_mask.unsqueeze(1).expand_as(attn).bool()    # expand along n_head dim
            #print(attn_mask)
            #print(attn_mask.dtype)
            assert attn_mask.size() == attn.size(), 'Attention mask shape {} mismatch ' \
                    'with Attention logit tensor shape {}.'.format(attn_mask.size(), attn.size())
            attn = attn.masked_fill_(attn_mask.bitwise_not(), float('-inf'))
        #print(attn)
        #print(attn.size())
        #print(attn.dtype)

        # 3. apply attention dropout and compute context vectors
        #alpha = self.mSoftMax(attn)            # [batch_size, n_head, key_len]
        alpha = F.softmax(attn, dim=-1)         # [batch_size, n_head, key_len]

        v = split_heads(v, self.n_head)             # [batch_size, n_head, key_len, 2*dim_per_head]
        attn = alpha[:, :, :, None] * v             # [batch_size, n_head, key_len, 2*dim_per_head]
        attn = combine_heads(attn, self.n_head)     # [batch_size, key_len, 2*d_model]

        attn = self.final_proj(attn.sum(1))       # [batch_size, 2 * d_model]

        alpha = alpha.sum(1) / self.n_head # get the attention of the first head, [batch_size, key_len]
        #alpha = alpha[:, 0, :].transpose(0, 1)  # get the attention of the first head, [key_len, batch_size]

        return alpha, attn


'''
---------------------------------------------------------------------------------------------------
An attention function can be described as mapping a query and a set of key-value pairs to an output,
where the query, keys, values, and output are all vectors. The output is computed as a weighted sum
of the values, where the weight assigned to each value is computed by a compatibility function of
the query with the corresponding key.

We call our particular attention “Scaled Dot-Product Attention”. The input consists of queries and
of dimension d_k, and values of dimension d_v. We compute the dot products of the query with all
keys, divide each by sqrt(d_k), and apply a softmax function to obtain the weights on the values.
'''

'''
In practice, we compute the attention function on a set of queries simultaneously, packed together
into a matrix Q. The keys and values are also packed together into matrices K and V.
We compute the matrix of outputs as:

Attention(Q, K, V) = softmax(Q*K^T / sqrt(d_k)) * V
'''
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    # query, key, value: torch.Size([B, n_heads, L, 512 / 8])
    d_k = query.size(-1)
    scores = tc.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # (B, n_heads, query_L, key_L), mask: (B, n_heads, -1, key_L)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return tc.matmul(p_attn, value), p_attn

'''
The two most commonly used attention functions are:
    * additive attention: computes the compatibility function using a feed-forward network with a
    single hidden layer.
    * dot-product (multiplicative) attention: is much faster and more space-efficient in practice,
    since it can be implemented using highly optimized matrix multiplication code
        the two are similar in theoretical complexity

Dot-product attention is identical to our algorithm, except for the scaling factor of 1 / sqrt(d_k)

For small values of d_k, the two mechanisms perform similarly.
For larger values of d_k, additive attention outperforms dot product attention without scaling.
We suspect that for large values of d_k, the dot products grow large in magnitude, pushing the
softmax function into regions where it has extremely small gradients (To illustrate why the dot
products get large, assume that the components of q and k are independent random variables with mean
0 and variance 1. Then their dot product, q·k=Sum_{i=1}^{d_k}q_i * k_i, has mean 0 and variance d_k)

To counteract this effect, we scale the dot products by 1 / sqrt(d_k).
'''


'''
Multi-head attention allows the model to jointly attend to information from different representation
subspaces at different positions. With a single attention head, averaging inhibits this.
MultiHead(Q,K,V)=Concat(head1,...,headh)W^O   where head_i=Attention(Q*W_i^Q,K*W_i^K,V*W_i^V)
Where the projections are parameter matrices:
    * W_i^Q (d_model, d_k)
    * W_i^K (d_model, d_k)
    * W_i^V (d_model, d_v)
    * W^O (h*d_v, d_model)
In this work we employ h=8 parallel attention layers, or heads. For each of these weights, we use
d_k=d_v=d_model/h=64.
Due to the reduced dimension of each head, the total computational cost is similar to that of
single-head attention with full dimensionality.
'''
from .self_att_model import clones
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)















