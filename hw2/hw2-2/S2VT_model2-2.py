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
    output:output_vector (dim=250,len=15)
    
"""
class S2VT_model(nn.Module):
    def __init__(self,encoder_hidden,decoder_hidden,batch_size,vector_dim=250):
        super (S2VT_model,self).__init__()
        #self.batchsize=batchsize #batchsize
        self.encoder=nn.LSTM(input_size=vector_dim,
                             batchsize=batch_size,
                             hiddensize=encoder_hidden)
        self.decoder=nn.LSTM(input_size=encoder_hidden,
                             batchsize=batch_size,
                             hiddensize=decoder_hidden)
        self.decoder_hidden=decoder_hidden
        self.batchsize=batch_size
        self.initial_h=torch.zeros((1,batch_size,encoder_hidden),dtype=torch.float32).cuda()
        self.initial_c=torch.zeros((1,batch_size,decoder_hidden),dtype=torch.float32).cuda()
        self.input_embedding=nn.Linear(250,vector_dim) 
        self.output_embedding=nn.Linear(decoder_hidden,250) 
        self.softmax=torch.nn.Softmax(dim=1)
    def inverse_sigmoid(self,x,k=1):
        return k/(k+np.exp(x/k))
    def forward(self,input_vector,input_label,epoch,check=0,sampling_start=200):
        sentence=[]
        BOS=torch.zeros((len(encoded_vector),self.batchsize,self.decoder_hidden),dtype=torch.float32).cuda()
        if check==1:
            input_vector=self.input_embedding(input_vector)
        encoded_vector,(encoder_h,encoder_c)=self.encoder(input_vector,(self.initial_h,self.initial_c))
        if epoch==0:
            decoded_vector,(decoder_h,decoder_c)=self.decoder(BOS,(encoder_h,encoder_c))
        else:
            # Schedule Sampling 
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
    
    
    
        
        