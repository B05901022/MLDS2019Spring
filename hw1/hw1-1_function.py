# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 16:34:40 2019

@author: Austin Hsu
"""

import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import torch.functional as F
import numpy as np

###DEVICE###
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###REPRODUCIBLE###
torch.manual_seed(1)

###HYPERPARAMETER###
EPOCH = 200
BATCHSIZE = 500
ADAMPARAM = {'lr':0.0001, 'betas':(0.9, 0.999), 'eps':1e-08}

###SHALLOW CNN MODEL###
class net_shallow(nn.Module):
    def __init__(self):
        super(shallow, self).__init__()
        self.dnn_layer = nn.Sequential(
                nn.Linear(1, 256),
                nn.Linear(256, 1))#769
        
    def forward(self, x):
        output = self.dnn_layer(x)
        return output

def parameter_count(input_model):
    model_params = list(input_model.parameters())
    total_params = 0
    for i in model_params:
        layer_parmas = 1
        for j in i.size():
            layer_parmas *= j
        total_params += layer_parmas
    return total_params

def exp_func(x):
    return np.sin(20 * np.cos(x))

def main():
    ###Load dataset###
    train_x = torch.from_numpy(np.random.rand(50000))
    train_y = torch.from_numpy(exp_func(train_x))
    train_data = Data.TensorDataset(train_x, train_y) 

    ###DATALOADER###
    train_dataloader = Data.DataLoader(
            dataset = train_data,
            batch_size = BATCHSIZE,
            shuffle = True,
            num_workers = 1)
    
    ###LOAD MODEL###
    train_model = net_shallow().to(device)
    
    ###OPTIMIZER###
    optimizer = torch.optim.Adam(train_model.parameters(), lr=ADAMPARAM['lr'], betas=ADAMPARAM['betas'], eps=ADAMPARAM['eps'], weight_decay=1e-5)
    
    ###LOSS FUNCTION###
    loss_func = nn.MSELoss()
    
    for e in range(EPOCH):
        
        print("Epoch ", e)
        epoch_loss = 0
        epoch_acc  = 0
        
        for b_num, (b_x, b_y) in enumerate(train_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            optimizer.zero_grad()
            pred = train_model(b_x)
            loss = loss_func(pred, b_y)
            loss.backward()
            optimizer.step()
            print("Batch: ", b_num, "loss: ", loss.item(), end = '\r')
            epoch_loss += loss.item()
            
        torch.save(train_model, 'function_shallow_model.pkl')
        torch.save(optimizer.state_dict(), 'function_shallow_model.optim')
        print("")
        print("Epoch loss: ", epoch_loss / (len(train_data)/BATCHSIZE))    
        
    return 
    

if __name__ == '__main__':
    main()