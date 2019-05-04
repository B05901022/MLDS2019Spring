# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 20:33:16 2019
@author: Austin Hsu
"""

"""
Reference:
    https://zhuanlan.zhihu.com/p/48731949
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
#import seaborn

#seaborn.set_context(context="talk")

"""
Encoder-Decoder Structure
Generator of results
"""

class EncoderDecoder(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 src_embed,
                 tgt_embed,
                 generator
                 ):
        super(EncoderDecoder, self).__init__()
        self.encoder   = encoder
        self.decoder   = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
    def forward(self,
                src,
                tgt,
                src_mask,
                tgt_mask,
                ):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
    def __init__(self,
                 d_model,
                 vocab
                 ):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

"""
Encoder
"""
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
    
class Encoder(nn.Module):
    def __init__(self,
                 layer,
                 N=6
                 ):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm   = LayerNorm(layer.size)
    def forward(self, x, mask):
        for single_layer in self.layers:
            x = single_layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    def __init__(self,
                 features,
                 eps=1e-6
                 ):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    def forward(self, x):
        mean   = x.mean(-1, keepdim=True)
        std    = x.std(-1, keepdim=True)
        output = self.a_2 * ( x - mean ) / ( std + self.eps) + self.b_2
        return output
    
class SublayerConnection(nn.Module):
    def __init__(self,
                 size,
                 dropout
                 ):
        super(SublayerConnection, self).__init__()
        self.norm    = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, sublayer):
        sublayer_output = self.dropout(self.norm(x))
        output = x + sublayer_output
        return output

class EncoderLayer(nn.Module):
    def __init__(self,
                 size,
                 self_attn,
                 feed_forward,
                 dropout
                 ):
        super(EncoderLayer, self).__init__()
        self.self_attn    = self_attn
        self.feed_forward = feed_forward
        self.sublayer     = clones(SublayerConnection(size, dropout), 2)
        self.size         = size
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        output = self.sublayer[1](x, self.feed_forward)
        return output

"""
Decoder
"""

class Decoder(nn.Module):
    def __init__(self,
                 layer,
                 N=6
                 ):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm   = LayerNorm(layer.size)
    def forward(self, x, memory, src_mask, tgt_mask):
        for single_layer in self.layers:
            x = single_layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self,
                 size,
                 self_attn,
                 src_attn,
                 feed_forward,
                 dropout
                 ):
        super(DecoderLayer, self).__init__()
        self.size         = size
        self.self_attn    = self_attn
        self.src_attn     = src_attn
        self.feed_forward = feed_forward
        self.sublayer     = clones(SublayerConnection(size, dropout), 3)
    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        output = self.sublayer[2](x, self.feed_forward)
        return output

"""
Mask
"""

def subsequent_mask(size):
    """
    Make sure later informations won't affect prediction of present timestep
    """
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    
    """
    np.triu
        means the upper triangle part of a matrix,
        k=1 means only parts that are above the diagonal part are leaved
        i.e. 
        a = array([[ 1, 2, 3],
                   [ 4, 5, 6],
                   [ 7, 8, 9],
                   [10,11,12]])
        np.triu(a, k=1)
         == array([[ 0, 2, 3],
                   [ 0, 0, 6],
                   [ 0, 0, 0],
                   [ 0, 0, 0]])
    """
    return torch.from_numpy(mask) == 0

"""
Attention
"""

def attention(query, key, value, mask=None, dropout=None):
    """
    
       Q       K       V
       |       |       |
      [ Matmul  ]      |
       |       |       |
      [ Scale   ]      |
       |       |       |
      [  Mask   ]      |
       |       |       |
      [ Softmax ]      |
       |       |       |
      [       Matmul     ]
                 |
              Output
      
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.mask_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

"""
Multi-Headed Attention
"""

class MultiHeadedAttention(nn.Module):
    def __init__(self,
                 h,
                 d_model,
                 dropout=0.1
                 ):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h   = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
        
"""
Positionwise Feed Forward
"""

class PositionwiseFeedForward(nn.Module):
    def __init__(self,
                 d_model,
                 d_ff,
                 dropout=0.1
                 ):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1     = nn.Linear(d_model, d_ff)
        self.w_2     = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        x      = self.w_1(x)
        x      = F.relu(x)
        x      = self.dropout(x)
        output = self.w_2(x)
        return output
    
"""
Embedding
"""

class Embeddings(nn.Module):
    def __init__(self,
                 d_model,
                 vocab,
                 pretrained_weight=None,
                 ):
        super(Embeddings, self).__init__()
        if pretrained_weight is None:
            self.lut = nn.Embedding(vocab, d_model)
        else:
            self.lut = nn.Embedding(vocab, d_model)
            self.lut.weight.data.copy_(torch.from_numpy(pretrained_weight))
            self.lut.weight.requires_grad = False
        self.d_model = d_model
    def forward(self, x):
        
        return self.lut(x) * math.sqrt(self.d_model)
    
"""
Positional Encoding
"""

class PositionalEncoding(nn.Module):
    def __init__(self,
                 d_model,
                 dropout,
                 max_len=5000
                 ):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe           = torch.zeros(max_len, d_model)
        position     = torch.arange(0. , max_len).unsqueeze(1)
        div_term     = torch.exp(torch.arange(0. , d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2]  = torch.sin(position * div_term)
        pe[:, 1::2]  = torch.cos(position * div_term)
        pe           = pe.unsqueeze(0)
        #pe           = pe[:, :20]
        #pe           = pe.squeeze(0)
        #pe           = pe.transpose(1,0)
        #pe           = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        #print('x : ',x.shape)
        #print('pe : ',Variable(self.pe[:, :x.size(1)], requires_grad=False).shape)
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

"""
Full Model
"""

def make_model(src_vocab,
               tgt_vocab,
               N=6,
               d_model=250, #512,#32,#256
               d_ff=1024, #2048,#256,#1024
               h=5,
               dropout=0.1,
               pretrained_weight=None,
               ):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff   = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
                Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
                Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
                nn.Sequential(Embeddings(d_model, src_vocab, pretrained_weight), c(position)),
                nn.Sequential(Embeddings(d_model, tgt_vocab, pretrained_weight), c(position)),
                Generator(d_model, tgt_vocab)
                )
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    
    return model
'''
tmp_model = make_model(250,250,6)
print('model complete')
None
'''
'''
class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data[0] * norm
'''       