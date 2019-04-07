# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 16:15:49 2019

@author: Austin Hsu
"""

import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import torch.functional as F
import numpy as np
import argparse
import json
import os
import yaml
from tqdm import tqdm

import load_data
import S2VT_model

###DEVICE###
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###REPRODUCIBLE###
torch.manual_seed(1)

###HYPERPARAMETER###
EPOCH      = 200
BATCHSIZE  = 16
ADAMPARAM  = {'lr':0.001, 'betas':(0.9, 0.999), 'eps':1e-08, 'weight_decay':1e-05}
MODELPARAM = {'e_layers':256,'e_hidden':256,'d_layers':256,'d_hidden':256}

###DATA LOADING PARAMS###
LOADPARAM  = {'directory': '../../../hw2-1/MLDS_hw2_1_data', 'min_count':3, 'random_seed':None, 'batch_size':16}
       
def main(args):
    
    ###DATALOADER###
    train_dataloader, one_hot_len, max_len, word_dict, datasize = load_data.load_data(**LOADPARAM)
    
    ###LOAD MODEL###
    train_model = S2VT_model.S2VT(
            attention = 0,
            batch_size = BATCHSIZE,
            **MODELPARAM,
            one_hot_length=one_hot_len
            ).cuda()
    
    ###OPTIMIZER###
    optimizer = torch.optim.Adam(train_model.parameters(), **ADAMPARAM)
    
    ###LOSS FUNCTION###
    loss_func = nn.CrossEntropyLoss()
    
    print("Training starts...")
    
    for e in range(EPOCH):
        print("Epoch ", e+1)
        epoch_loss = 0
        
        for b_num, (b_x, b_y) in enumerate(tqdm(train_dataloader)):
            b_x = b_x.cuda()
            b_y = b_y.cuda()
            optimizer.zero_grad()
            pred = train_model(b_x, max_len, b_y)
            loss = loss_func(pred, b_y)
            loss.backward()
            optimizer.step()
            print("Batch: ", b_num, "loss: ", loss.item(), end = '\r')
            epoch_loss += loss.item()
            
        torch.save(train_model, './models/'+args.model_no+'_model.pkl')
        torch.save(optimizer.state_dict(), './models/'+args.model_no+'_model.optim')
        print("")   
        print("Epoch loss: ", epoch_loss / datasize)
        #loss_history.append(epoch_loss / len(train_data))
    
    print("Training finished.")
    
    return 
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_no', '-m', type=str, default='0')
    args = parser.parse_args()
    main(args)
    
