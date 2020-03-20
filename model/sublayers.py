#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 13:42:38 2020

@author: fatimamh
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import copy
'''---------------------------------------------------------------
'''
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
'''-------------------------------------------------------
'''
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

'''-------------------------------------------------------
'''
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

'''-------------------------------------------------------
'''
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #print('\nPositionwise feed forward\n-----------------')
        x = self.w_2(self.dropout(F.relu(self.w_1(x))))
        #print('x-w_2-drop-relu-w_1: {}'.format(x.shape))
        #print('nPositionwise feed forward\n-----------------')
        return x

'''-------------------------------------------------------
'''
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        #learnable parameters to calibrate normalization
        self.a_2 = nn.Parameter(torch.ones(features)) # alpha
        self.b_2 = nn.Parameter(torch.zeros(features)) # bias 
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

'''-------------------------------------------------------
'''
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

'''-------------------------------------------------------
'''
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    #print('\n-----attention--------------')
    #print('query: {}'.format(query.shape))
    #print('key: {}'.format(key.shape))
    #print('value: {}'.format(value.shape))
    #print('mask: {}'.format(mask.shape))
    d_k = query.size(-1)
    #print('d_k: {}'.format(d_k))

    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    #print('scores: {}'.format(scores.shape))

    if mask is not None:
        #print('here')
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    #print('-----attention--------------\n')
    return torch.matmul(p_attn, value), p_attn

'''-------------------------------------------------------
'''
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
        #print('\nMulti head Attention\n-----------------')
        #print('mask: {}'.format(mask.shape))
        
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        #print('query: {}'.format(query.shape))
        #print('key: {}'.format(key.shape))
        #print('value: {}'.format(value.shape))
        #print('mask: {}'.format(mask.shape))
        #print()
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        
        #print('query: {}'.format(query.shape))
        #print('key: {}'.format(key.shape))
        #print('value: {}'.format(value.shape))
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        #print('x: {}'.format(x.shape))
        #print('self.attn: {}'.format(self.attn.shape))

        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        #print('x: {}'.format(x.shape))

        x = self.linears[-1](x)
        #print('x: {}'.format(x.shape))

        #print('Multi head Attention\n-----------------')
        return x


