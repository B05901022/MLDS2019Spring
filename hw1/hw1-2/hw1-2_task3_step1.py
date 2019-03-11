#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 00:56:07 2019

@author: austinhsu
"""

import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import argparse

###DEVICE###
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###REPRODUCIBLE###
torch.manual_seed(1)

###HYPERPARAMETER###
EPOCH = 100
BATCHSIZE = 50
ADAMPARAM = {'lr':0.001, 'betas':(0.9, 0.999), 'eps':1e-08}

###MODEL###
class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.layer1 = nn.Linear(1,128)
        self.layer2 = nn.Linear(128,1)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        output = self.layer2(x)
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
    return np.sin(5*np.pi*x)/(5*np.pi*x)

#=============================================================================================
"""
reference:https://github.com/Ageliss/For_shared_codes/blob/master/Second_order_gradients.py
"""    

def get_second_order_grad(xs, ys):
    grads = torch.autograd.grad(ys, xs, create_graph=True)
    grads2 = []
    for j, (grad, x) in enumerate(zip(grads, xs)):
        grad = torch.reshape(grad, [-1])
        grads2_tmp = []
        for count, g in enumerate(grad):
            g2 = torch.autograd.grad(g, x, retain_graph=True)[0]
            g2 = torch.reshape(g2, [-1])
            grads2_tmp.append(g2[count].data.cpu().numpy())
        grads2.append(torch.from_numpy(np.reshape(grads2_tmp, x.size())).to(device))
    return grads2

"""
xs = optimizer.param_groups[0]['params']
ys = loss
grads2 = get_second_order_grad(grads, xs) # second order gradient
"""

#===============================================================================================

def main(args):
    ###Load dataset###
    train_x = np.random.rand(5000).reshape(5000,1)
    train_y = torch.from_numpy(exp_func(train_x))
    train_x = torch.from_numpy(train_x)
    train_data = Data.TensorDataset(train_x, train_y) 

    ###DATALOADER###
    train_dataloader = Data.DataLoader(
            dataset = train_data,
            batch_size = BATCHSIZE,
            shuffle = True,
            num_workers = 1)
    
    ###LOAD MODEL###
    train_model = net().to(device)
    
    ###OPTIMIZER###
    #optimizer = torch.optim.Adam(train_model.parameters(), lr=ADAMPARAM['lr'], betas=ADAMPARAM['betas'], eps=ADAMPARAM['eps'], weight_decay=1e-5)
    optimizer = torch.optim.Adagrad(train_model.parameters(), lr=0.001)
    
    ###LOSS FUNCTION###
    loss_func = nn.MSELoss()
    
    ###LOSS RECORD###
    loss_history = []
    
    ###Training###    
    for e in range(EPOCH):
        
        #print("Epoch ", e)
        epoch_loss = 0
        
        for b_num, (b_x, b_y) in enumerate(train_dataloader):
            b_x = b_x.to(device)
            b_x = b_x.float()
            b_y = b_y.to(device)
            b_y = b_y.float()
            optimizer.zero_grad()
            pred = train_model(b_x)
            loss = loss_func(pred, b_y)
            Hessian = np.array(get_second_order_grad(optimizer.param_groups[0]['params'], loss))
            w_t = np.array(optimizer.param_groups[0]['params'])
            print(w_t[0].shape)
            print(w_t[1].shape)
            print(w_t[2].shape)
            print(w_t[3].shape)
            print()
            print(Hessian[0].shape)
            print(Hessian[1].shape)
            print(Hessian[2].shape)
            print(Hessian[3].shape)
            sssss
            loss.backward()
            optimizer.step()
            print("Epoch", e, "Batch: ", b_num, "loss: ", loss.item(), end = '\r')
            epoch_loss += loss.item()
            
        torch.save(train_model, 'hw1-2_task3_step1_function_model.pkl')
        torch.save(optimizer.state_dict(), 'hw1-2_task3_step1_function_model.optim')
        #if e%100 == 0:
        print("")
        print("Epoch loss: ", epoch_loss / len(train_data))
        loss_history.append( epoch_loss / len(train_data))
        
        
    return 
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='On function y = sin(5*pi*x)/(5*pi*x)')
    args = parser.parse_args()
    main(args)