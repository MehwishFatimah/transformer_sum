#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 13:42:38 2020

@author: fatimamh
"""

import model_config as config

import torch
import torch.nn as nn 
from torch.autograd import Variable
from model.layers import *
from model.sublayers import *
import copy

'''---------------------------------------------------------------
'''
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

'''---------------------------------------------------------------
'''
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        #print('\nCore Encoder\n-----------------')
        for layer in self.layers:
            x = layer(x, mask)
            #print('x-layer {}: {}'.format(layer, x.shape))
        
        x = self.norm(x)
        #print('x-norm: {}'.format(x.shape))
        #print('Core Encoder\n-----------------')        
        return x
        
'''---------------------------------------------------------------
'''

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        #print('\nCore Decoder\n-----------------')
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
            #print('x-layer {}: {}'.format(layer, x.shape))
        x = self.norm(x)
        #print('x-norm: {}'.format(x.shape))
        #print('Core Decoder\n-----------------')        
        return x

'''---------------------------------------------------------------
'''

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


'''---------------------------------------------------------------
'''
class EncoderDecoder(nn.Module):
    """     A standard Encoder-Decoder architecture.     """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


'''---------------------------------------------------------------
'''
# from config to here, propogate to over all model modules
def make_model(device,                        \
               src_vocab = config.text_vocab, \
               tgt_vocab = config.sum_vocab,  \
               N         = config.n_layers,   \
               d_model   = config.d_model,    \
               d_ff      = config.ff_hidden,  \
               h         = config.n_heads,    \
               dropout   = config.dropout):
    
    "Helper: Construct a model from hyperparameters."
    c        = copy.deepcopy
    attn     = MultiHeadedAttention(h, d_model)
    ff       = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    # 5 main layers: encoder, deocder, 2seq and generator
    model    = EncoderDecoder(Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
                              Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
                              nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
                              nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
                              Generator(d_model, tgt_vocab))
    
    model = model.to(device)
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

'''
if __name__ == '__main__':
    
    device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(device))
    test = get_model(device)
    print(test.__dict__)
'''