# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 11:43:55 2019

@author: u8815
"""
import torch.nn as nn
import torch
import numpy as np
"""
    S2VT model
    input:train_x,train_y (dim=250,len=15)
    need to split data beforehand
    
"""

class AttentionDecoder(nn.Module):
    def __init__(self,hidden_size, output_size,max_length=15,dropout_rate=0.25):
        super (AttentionDecoder,self).__init__()
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.attn=nn.Linear(self.hidden_size*2,self.max_length)
        self.attn_comb=nn.Linear(self.hidden_size*2,self.output_size)
        self.dropout=nn.Dropout(dropout_rate)
    
class S2VT_model(nn.Module):
    def __init__(self,encoder_hidden,decoder_hidden,batch_size,dropout_rate=0.25,vector_dim=250,max_length=15):
        super (S2VT_model,self).__init__()
        #self.batchsize=batchsize #batchsize
        self.encoder=nn.LSTM(input_size=vector_dim,
                             batch_size=batch_size,
                             hidden_size=encoder_hidden)
        self.decoder=nn.LSTM(input_size=encoder_hidden,
                             batch_size=batch_size,
                             hidden_size=decoder_hidden)
        self.encoder_hidden=encoder_hidden
        self.decoder_hidden=decoder_hidden
        self.batchsize=batch_size
        self.initial_h=torch.zeros((1,batch_size,encoder_hidden),dtype=torch.float32).cuda()
        self.initial_c=torch.zeros((1,batch_size,decoder_hidden),dtype=torch.float32).cuda()
        self.input_embedding=nn.Linear(250,vector_dim) 
        self.output_embedding=nn.Linear(decoder_hidden,250) 
        self.softmax=torch.nn.Softmax(dim=1)
        self.attn=nn.Linear(self.encoded_hidden_size*2,max_length)
        self.attn_comb=nn.Linear(self.encodeder_hidden*2,self.hidden_size)
        self.dropout=nn.Dropout(dropout_rate)
    def inverse_sigmoid(self,x,k=1):
        return k/(k+np.exp(x/k))
    def forward(self,input_vector,input_label,epoch,check=0,sampling_start=200,):
        sentence=[]
        if check==1:
            input_vector=self.input_embedding(input_vector)
        hidden=torch.zeros((1,batch_size,encoder_hidden),dtype=torch.float32).cuda()
        encoded_vector,(encoder_h,encoder_c)=self.encoder(input_vector,(self.initial_h,self.initial_c))
        BOS=torch.zeros((len(encoded_vector),self.batchsize,self.decoder_hidden),dtype=torch.float32).cuda()
        #Attention
        attn_weights=F.softmax(
            self.attn(torch.cat((input_vector[0], hidden[0]), 1)), dim=1)
        attn_applied=torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output=torch.cat((embedded[0], attn_applied[0]), 1)
        output=self.attn_comb(output).unsqueeze(0)
        #End
        if epoch==0:
            decoded_vector,(decoder_h,decoder_c)=self.decoder(BOS,(encoder_h,encoder_c))
        else:
            # Schedule Sampling 
            if check==1:
                input_label=self.input_embedding(input_label)
            if epoch<sampling_start: #After sampling_start epochs, Schedule sampling will work.
                decoded_vector,(decoder_h,decoder_c)=self.decoder(input_label,(encoder_h,encoder_c))
            else:
                schedule=np.random.uniform()
                if schedule<self.inverse_sigmoid(epoch-500,100):            
                    decoded_vector,(decoder_h,decoder_c)=self.decoder(input_label,(encoder_h,encoder_c))
                else:
                    decoded_vector,(decoder_h,decoder_c)=self.decoder(encoded_vector,(encoder_h,encoder_c))
            # End
        output_vector=self.output_embedding(decoded_vector)
        output_vector=self.softmax(output_vector)
        index=torch.argmax(output_vector,dim=1)
        sentence.append(index)
        #output : vectors (index)
        return sentence
    
    
    
        
        