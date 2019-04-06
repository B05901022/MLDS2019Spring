# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 14:21:41 2019

@author: u8815
"""

import torch.nn as nn
import torch
import numpy as np
import argparse
import torchvision
"""
for filename in os.listdir(sys.argv[1]):
    feature.append(np.load(sys.argv[1]+filename))
"""
class S2VT(nn.Module):
    def __init__(self,attention,batch_size,e_layers,e_hidden,
                 d_layers,d_hidden,one_hot_length):
        super(S2VT,self).__init__()
        self.attention=attention
        self.batch_size=batch_size
        self.encoder_layers=e_layers
        self.encoder_hidden=e_hidden
        self.decoder_layers=d_layers
        self.decoder_hidden=d_hidden
        self.ohl=one_hot_length
        self.encoder_h=torch.zeros((e_layers,batch_size,e_hidden),dtype=torch.float32)
        self.encoder_c=torch.zeros((e_layers,batch_size,e_hidden),dtype=torch.float32)
        self.decoder_h=torch.zeros((d_layers,batch_size,d_hidden),dtype=torch.float32)
        self.decoder_c=torch.zeros((d_layers,batch_size,d_hidden),dtype=torch.float32)
        self.encoder=nn.LSTM(input_size=4096,
                                hidden_size=e_hidden,
                                num_layers=e_layers)

        self.decoder=nn.LSTM(input_size=d_hidden+e_hidden+25,
                                hidden_size=d_hidden,
                                num_layers=d_layers)
    def add_pad(self,input_feature,i,max_len=500):
        if i==1:
            pad=torch.zeros((self.decoder_layers,self.batch_size,self.decoder_hidden),
                            dtype=torch.float32)
        
            processed=torch.cat((input_feature,pad),dim=2)
            return processed
        elif i==0:
            bos=torch.zeros((self.decoder_layers,self.batch_size,self.decoder_hidden),
                            dtype=torch.float32)
            bos[:,:,2]=1
            return bos
            #processed=torch.cat((input_feature,bos),dim=2)
    def embedding_layer(self,c,control):
        if control:
            el=nn.Linear(self.ohl,self.decoder_h)
        else:
            el=nn.Linear(self.decoder_h,self.ohl)
        return el(c)
    def forward(self,input_feature,max_len,input_fromavi):
        sentence=[]
        """Encoding"""
        eencoded_data,(he,ce)=self.encoder(input_feature,self.encoder_h,self.encoder_c)
        eeinput_data=self.add_pad(input_feature,1)
        decoded_data,(hd,cd)=self.decoder(eeinput_data,self.decoder_h,self.decoder_c)
        """Decoding""" 
        decoding_padding=torch.zeros((max_len,self.batch_size,self.decoder_hidden),
                            dtype=torch.float32)
        ddinput_data,(he,ce)=self.encoder(decoding_padding,he, ce)
        
        for s in range(max_len):        
            if s==0:
                dencoded_data,(hd,cd)=self.decoder(ddinput_data,(hd,cd))
                input_embb=self.embedding_layer(self.add_pad(inputdata=None,i=0),1) 
                input_fromlstm=(ddinput_data[s]).unsqueeze(0)
            else:
                scheduled=0.9
                if np.random.uniform()<scheduled:
                    input_embb=embedding_layer(input_fromavi[s],1)
                else:
                    input_embb=decoded_data
                input_fromlstm=(ddinput_data[s]).unsqueeze(0)
            eeinput_data=torch.cat((input_embb,input_fromlstm))
            decoded_data,(hd,cd)=nn.LSTM(input_size=eeinput_data.size(),
                             hidden_size=self.decoder_hidden,
                             num__layers=self.decoder_layers)(eeinput_data,(hd,cd))
            word=self.embedding_layer(decoded_data,0).squeeze(0)
            sentense.append(word)
        return sentenses        
    def test(self,input_feature,max_len):
        sentence=[]
        """Encoding"""
        eencoded_data,(he,ce)=self.encoder(input_feature,self.encoder_h,self.encoder_c)
        eeinput_data=self.add_pad(input_feature,1)
        decoded_data,(hd,cd)=self.decoder(eeinput_data,(self.decoder_h,self.decoder_c))
        """Decoding""" 
        decoding_padding=torch.zeros((max_len,self.batch_size,self.decoder_hidden),
                            dtype=torch.float32)
        ddinput_data,(he,ce)=self.encoder(decoding_padding,he, ce)
        
        for s in range(max_len):        
            if s==0:
                dencoded_data,(hd,cd)=self.decoder(ddinput_data,(hd,cd))
                input_embb=self.embedding_layer(self.add_pad(i=0),1) 
                input_fromlstm=(ddinput_data[s]).unsqueeze(0)
            else:
                input_embb=decoded_data
                input_fromlstm=(ddinput_data[s]).unsqueeze(0)
            eeinput_data=torch.cat((input_embb,input_fromlstm))
            decoded_data,(hd,cd)=self.decoder(eeinput_data,(hd,cd))
            word=self.embedding_layer(decoded_data,0).squeeze(0)
            word=torch.nn.Softmax(dim=1)(word)
            sentence.append(word)
        return sentence
    
"""
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s
"""