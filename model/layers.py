#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 13:42:38 2020

@author: fatimamh
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from model.sublayers import *

'''-------------------------------------------------------
'''
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        #print('\nEncoder Layer\n-----------------')
        
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        #print('x: {}'.format(x.shape))
        
        x = self.sublayer[1](x, self.feed_forward)
        #print('x: {}'.format(x.shape))
        
        #print('Encoder Layer\n-----------------')
        return x

'''-------------------------------------------------------
''' 
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        #print('\nDecoder Layer\n-----------------')
        
        m = memory
        #print('m: {}'.format(m.shape))
        #print('x: {}'.format(x.shape))
        #print()
        #print('1 sublayer')            
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        #print('1 DL x: {}'.format(x.shape))
        
        #print()
        #print('2 sublayer')
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        #print('2 DL x: {}'.format(x.shape))
        
        #print()
        #print('3 sublayer')
        x = self.sublayer[2](x, self.feed_forward)
        #print('3 DL x: {}'.format(x.shape))
        
        #print('Decoder Layer\n-----------------')
        return x
         

