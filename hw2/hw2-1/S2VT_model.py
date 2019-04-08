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
import sklearn.decomposition

class S2VT(nn.Module):
    def __init__(self,attention,batch_size,e_layers,e_hidden,
                 d_layers,d_hidden,one_hot_length,minibatch):
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
        self.encoder=nn.LSTM(input_size=1024,
                                hidden_size=e_hidden,
                                num_layers=e_layers,
                                bidirectional=False
                                )

        self.decoder=nn.LSTM(input_size=e_hidden+d_hidden,
                                hidden_size=d_hidden,
                                num_layers=d_layers,
                                bidirectional=False)
        self.embedding_layer_i=nn.Linear(self.ohl,d_hidden)
        self.embedding_layer_o=nn.Linear(d_hidden,self.ohl)
        self.embedding_layer_down=nn.Sequential(nn.Linear(4096,1024),
                                                nn.ReLU())
        self.softmax=torch.nn.Softmax(dim=1)
        self.relu=nn.ReLU()
        self.minibatch=minibatch
            #processed=torch.cat((input_feature,bos),dim=2)
    """
    def embedding_layer(self,c,control):
        if control:
            el=nn.Linear(self.ohl,self.decoder_h)
        else:
            el=nn.Linear(self.decoder_h,self.ohl)
        return el(c)
    """
    def inverse_sigmoid(self,x,k):
        return k/(k+np.exp(x/k))
    def forward(self,input_feature,max_len,correct_answer):
        sentence=[]
        """Encoding"""
        input_feature=self.embedding_layer_down(input_feature)
        input_feature=input_feature.view(input_feature.shape[1],input_feature.shape[0],1024)
        encoded_sequence,(he,ce)=self.encoder(input_feature,(self.encoder_h,self.encoder_c))
        #decoded_input=self.add_pad(encoded_sequence,1)
        pad=torch.zeros((len(encoded_sequence),self.batch_size,self.decoder_hidden),dtype=torch.float32).cuda()       
        decoded_input=torch.cat((encoded_sequence,pad),dim=2)
        decoded_output,(hd,cd)=self.decoder(decoded_input,(self.decoder_h,self.decoder_c))
        """Decoding""" 
        padding=torch.zeros((max_len,self.batch_size,1024),
                            dtype=torch.float32).cuda()
        #print(padding.shape,"pad")
        encoded_padding,(he,ce)=self.encoder(padding,(he, ce))
        bos=torch.zeros((1,self.batch_size,self.ohl),
                        dtype=torch.float32).cuda()
        #print(bos.shape,"bos")
        bos[:,:,-2]=1
        for s in range(max_len): 
            correct=None
            sample=None
            if (s==0):
                #dencoded_data,(hd,cd)=self.decoder(ddinput_data,(hd,cd))
                bos_embedding=self.embedding_layer_i(bos) 
                sample=bos_embedding
                correct=(encoded_padding[s]).unsqueeze(0)
                decoded_input=torch.cat((sample,correct),dim=2)
                decoded_output,(hd,cd)=self.decoder(decoded_input,(hd,cd))
                word=self.embedding_layer_o(decoded_output).squeeze(0)
                sentence.append(word)    
            else:
                a=correct_answer[s].unsqueeze(0)
                
                schedule=self.inverse_sigmoid(self.minibatch, k=4)#schedule=self.inverse_sigmoid((s-22)/10,1)
                c=np.random.uniform()
                if c<schedule:
                    sample=self.embedding_layer_i(a)
                else: 
                    sample=decoded_output
                    
                
                #sample=self.embedding_layer_i(a)
                #print("encoded_padding[s]:", encoded_padding[s].shape)
                correct=(encoded_padding[s]).unsqueeze(0)
                """
                sample = torch.unsqueeze(sample, 0)#maybe not right
                sample = torch.unsqueeze(sample, 0)#maybe not 
            
                print("correct_answer:", end='')
                print(correct_answer.shape)
                print("a:", end='')
                print(a.shape)
                print("sample:", end='')
                print(sample.shape)
                print("correct:", end='')
                print(correct.shape)
                """
                #decoded_input=torch.cat((sample,correct),dim=2)
                decoded_input=torch.cat((sample,correct),dim=2)#maybe not right
                decoded_output,(hd,cd)=self.decoder(decoded_input,(hd,cd))
                word=self.embedding_layer_o(decoded_output).squeeze(0)
                sentence.append(word)
        return sentence

    
    def test(self,input_feature,max_len):
       sentence=[]
        """Encoding"""
        input_feature=self.embedding_layer_down(input_feature)
        input_feature=input_feature.view(input_feature.shape[1],input_feature.shape[0],1024)
        encoded_sequence,(he,ce)=self.encoder(input_feature,(self.encoder_h,self.encoder_c))
        #decoded_input=self.add_pad(encoded_sequence,1)
        pad=torch.zeros((len(encoded_sequence),self.batch_size,self.decoder_hidden),dtype=torch.float32).cuda()       
        decoded_input=torch.cat((encoded_sequence,pad),dim=2)
        decoded_output,(hd,cd)=self.decoder(decoded_input,(self.decoder_h,self.decoder_c))
        """Decoding""" 
        padding=torch.zeros((max_len,self.batch_size,1024),
                            dtype=torch.float32).cuda()
        #print(padding.shape,"pad")
        encoded_padding,(he,ce)=self.encoder(padding,(he, ce))
        bos=torch.zeros((1,self.batch_size,self.ohl),
                        dtype=torch.float32).cuda()
        #print(bos.shape,"bos")
        bos[:,:,-2]=1
        for s in range(max_len): 
            correct=None
            sample=None
            if (s==0):
                #dencoded_data,(hd,cd)=self.decoder(ddinput_data,(hd,cd))
                bos_embedding=self.embedding_layer_i(bos) 
                sample=bos_embedding
                correct=(encoded_padding[s]).unsqueeze(0)
                decoded_input=torch.cat((sample,correct),dim=2)
                decoded_output,(hd,cd)=self.decoder(decoded_input,(hd,cd))
                word=self.embedding_layer_o(decoded_output).squeeze(0)
                sentence.append(word)    
            else:
                """
                a=correct_answer[s].unsqueeze(0)
                
                schedule=self.inverse_sigmoid((s-22)/10,1)
                c=np.random.uniform()
                if c<schedule:
                    sample=self.embedding_layer_i(a)
                else: 
                    sample=decoded_output
                """
                
                sample=decoded_output
                #print("encoded_padding[s]:", encoded_padding[s].shape)
                correct=(encoded_padding[s]).unsqueeze(0)
                """
                sample = torch.unsqueeze(sample, 0)#maybe not right
                sample = torch.unsqueeze(sample, 0)#maybe not 
            
                print("correct_answer:", end='')
                print(correct_answer.shape)
                print("a:", end='')
                print(a.shape)
                print("sample:", end='')
                print(sample.shape)
                print("correct:", end='')
                print(correct.shape)
                """
                #decoded_input=torch.cat((sample,correct),dim=2)
                decoded_input=torch.cat((sample,correct),dim=2)#maybe not right
                decoded_output,(hd,cd)=self.decoder(decoded_input,(hd,cd))
                word=self.embedding_layer_o(decoded_output).squeeze(0)
                word=self.softmax(word)
                sentence.append(word)
        return sentence

    
        return sentence
    
