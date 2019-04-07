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
        self.encoder_h=torch.zeros((e_layers,batch_size,e_hidden),dtype=torch.float32).cuda()
        self.encoder_c=torch.zeros((e_layers,batch_size,e_hidden),dtype=torch.float32).cuda()
        self.decoder_h=torch.zeros((d_layers,batch_size,d_hidden),dtype=torch.float32).cuda()
        self.decoder_c=torch.zeros((d_layers,batch_size,d_hidden),dtype=torch.float32).cuda()
        self.encoder=nn.LSTM(input_size=4096,
                                hidden_size=e_hidden,
                                num_layers=e_layers)

        self.decoder=nn.LSTM(input_size=e_hidden+d_hidden,
                                hidden_size=d_hidden,
                                num_layers=d_layers)
        self.embedding_layer_i=nn.Linear(self.ohl,d_hidden)
        self.embedding_layer_o=nn.Linear(d_hidden,self.ohl)
    def add_pad(self,input_feature,i,max_len=500):
        if i==1:
            pad=torch.zeros((len(input_feature),self.batch_size,self.decoder_hidden),
                            dtype=torch.float32).cuda()       
            processed=torch.cat((input_feature,pad),dim=2)
            return processed
            #processed=torch.cat((input_feature,bos),dim=2)
    """
    def embedding_layer(self,c,control):
        if control:
            el=nn.Linear(self.ohl,self.decoder_h)
        else:
            el=nn.Linear(self.decoder_h,self.ohl)
        return el(c)
    """
    def forward(self,input_feature,max_len,correct_answer):
        sentence=[]
        """Encoding"""
        input_feature=input_feature.view(80,4,4096)
        encoded_sequence,(he,ce)=self.encoder(input_feature,(self.encoder_h,self.encoder_c))
        decoded_input=self.add_pad(encoded_sequence,1)
        decoded_output,(hd,cd)=self.decoder(decoded_input,(self.decoder_h,self.decoder_c))
        """Decoding""" 
        padding=torch.zeros((max_len,self.batch_size,4096),
                            dtype=torch.float32).cuda()
        print(padding.shape,"pad")
        encoded_padding,(he,ce)=self.encoder(padding,(he, ce))
        bos=torch.zeros((1,self.batch_size,self.ohl),
                        dtype=torch.float32).cuda()
        print(bos.shape,"bos")
        bos[:,:,-2]=1
        for s in range(max_len): 
            correct=None
            sample=None
            if (s==0):
                #dencoded_data,(hd,cd)=self.decoder(ddinput_data,(hd,cd))
                bos_embedding=self.embedding_layer_i(bos) 
                sample=bos_embedding
                print(sample.shape)
                correct=(encoded_padding[s]).unsqueeze(0)
                print(correct.shape)
                decoded_input=torch.cat((sample,correct),dim=2)
                decoded_output,(hd,cd)=self.decoder(decoded_input,(hd,cd))
                word=self.embedding_layer_o(decoded_output).squeeze(0)
                sentence.append(word)    
            else:
                a=correct_answer[s] #.
                sample=self.embedding_layer_i(a)
                correct=(encoded_padding[s]).unsqueeze(0)
                print("correct_answe:", end='')
                print(correct_answer.shape)
                print("a:", end='')
                print(a.shape)
                print("sample:", end='')
                print(sample.shape)
                print("correct:", end='')
                print(correct.shape)
                decoded_input=torch.cat((sample,correct),dim=2)
                decoded_output,(hd,cd)=self.decoder(decoded_input,(hd,cd))
                word=self.embedding_layer_o(decoded_output).squeeze(0)
                sentence.append(word)
        sentence=torch.stack(sentence)

        return sentence

    '''
    def test(self,input_feature,max_len):
        sentence=[]
        """Encoding"""
        input_feature=torch.unsqueeze(input_feature,0)
        input_feature=input_feature.view(80,1,4096)
        eencoded_data,(he,ce)=self.encoder(input_feature,(self.encoder_h,self.encoder_c))
        print(eencoded_data.shape)
        eeinput_data=self.add_pad(eencoded_data,1)
        decoded_data,(hd,cd)=self.decoder(eeinput_data,(self.decoder_h,self.decoder_c))
        """Decoding""" 
        decoding_padding=torch.zeros((max_len,self.batch_size,4096),
                            dtype=torch.float32).cuda()
        ddinput_data,(he,ce)=self.encoder(decoding_padding,(he, ce))
        
        for s in range(max_len):        
            if s==0:
                #dencoded_data,(hd,cd)=self.decoder(ddinput_data,(hd,cd))
                input_embb=self.embedding_layer_i(self.add_bos()) 
                input_fromlstm=(ddinput_data[s]).unsqueeze(0)
            else:
                input_embb=decoded_data
                input_fromlstm=(ddinput_data[s]).unsqueeze(0)
            eeinput_data=torch.cat((input_embb,input_fromlstm),dim=2)
            decoded_data,(hd,cd)=self.decoder(eeinput_data,(hd,cd))
            word=self.embedding_layer_o(decoded_data).squeeze(0)
            sentence.append(word)
        return sentence        
    '''
