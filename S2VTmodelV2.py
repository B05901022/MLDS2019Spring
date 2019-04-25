# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 13:45:50 2019

@author: u8815
"""
"""
Encoder and Decoder with Attention
"""
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self,hidden_size,input_size,h):
        super(Encoder,self).__init__().view(1, self.batch_size, -1)
        self.hidden_size=hidden_size
        self.batch_size=batch_size
        self.embedding=nn.Embedding(input_size,hidden_size)
        self.GRU=nn.GRU(hidden_size,hidden_size,batch_first=True)
        self.h=h
    def forward(self,encoder_input):
        embedded_input=self.embedding(encoder_input)
        embedded_output,h_output=self.GRU(embedded_input,self.h)
        return embedded_output,h_output
class AttentionDecoder(nn.Module):
    def __init__(self,hidden_size,input_size,batch_size,h):
        super(AttentionDecoder,self).__init__()
        self.hidden_size=hidden_size
        self.batch_size=batch_size
        self.embedding=nn.Embedding(input_size,hidden_size)
        self.GRU=nn.GRU(input_size=hidden_size,hidden_size=hidden_size)
        self.h=h
        self.attn=nn.Linear(self.hidden_size*2,15) 
        self.attn_comb=nn.Linear(self.hidden_size*2,self.hidden_size)
        self.dropout=nn.Dropout(0.5)
        self.out=nn.Linear(self.hidden_size,250)
    def forward(self,decoder_input,hidden,encoder_output):
        embedded_input=self.embedding(decoder_input).view(1,self.batch_size,-1)
        #embedded_input=self.dropout(embedded_input)
        #Attention Start
        attention_weight=F.softmax(self.attn(torch.cat((embedded_input[0],hidden[0]),1)),dim=1)
        attention_applied=torch.bmm(attention_weight.unsqueeze(0),encoder_output.unsqueeze(0))
        attention_output=torch.cat((embedded_input[0],attention_applied[0]),1)
        attention_output=self.attn_comb(attention_output).unsqueeze(0)
        #End
        attention_output=F.relu(attention_output)
        embedded_output,h_output=self.GRU(attetion_output,self.h)
        embedded_output=F.softmax(embedded_output) 
        return embbeded_output,h_output
