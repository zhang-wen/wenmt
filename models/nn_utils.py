import math
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from tools.utils import wlog

epsilon = 1e-20

class MaskSoftmax(nn.Module):

    def __init__(self):

        super(MaskSoftmax, self).__init__()

    def forward(self, x, mask=None, dim=-1):

        # input torch tensor or variable, take max for numerical stability
        x_max = tc.max(x, dim=dim, keepdim=True)[0]
        x_minus = x - x_max
        x_exp = tc.exp(x_minus)
        if mask is not None: x_exp = x_exp * mask
        x = x_exp / ( tc.sum( x_exp, dim=dim, keepdim=True ) + epsilon )

        return x

class MyLogSoftmax(nn.Module):

    def __init__(self):

        super(MyLogSoftmax, self).__init__()

    def forward(self, x, dim=-1):

        # input torch tensor
        x_max = tc.max(x, dim=dim, keepdim=True)[0]  # take max for numerical stability
        x_exp = tc.exp( x - x_max )
        x_exp_sum = tc.sum( x_exp, dim=dim, keepdim=True ) + epsilon
        log_norm = tc.log( x_exp_sum ) + x_max
        x = x - log_norm    # get log softmax
        prob = x_exp / x_exp_sum

        return log_norm, prob, x

'''Layer normalize the tensor x, averaging over the last dimension.'''
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(tc.ones(features))
        wlog('*Ones init a in layernorm {}'.format(self.a_2.size()))
        self.b_2 = nn.Parameter(tc.zeros(features))
        wlog('*Zeros init b in layernorm {}'.format(self.b_2.size()))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    wlog('*Xavier init linear weight {}'.format(m.weight.size()))
    if bias is True:
        nn.init.constant_(m.bias, 0.)
        wlog('*Zeros init linear bias {}'.format(m.bias.size()))
    return m

















