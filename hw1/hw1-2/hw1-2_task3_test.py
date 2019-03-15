#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 22:23:01 2019

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
EPOCH = 10
BATCHSIZE = 500
ADAMPARAM = {'lr':0.00001, 'betas':(0.9, 0.999), 'eps':1e-08}

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
Hessian

reference:https://github.com/Ageliss/For_shared_codes/blob/master/Second_order_gradients.py
"""

"""
xs = optimizer.param_groups[0]['params']
ys = loss
grads2 = get_second_order_grad(grads, xs) # second order gradient
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
            grads2_tmp.append(g2.data.cpu().numpy())
        s = max(x.size())
        grads2.append(torch.from_numpy(np.array(grads2_tmp).reshape(s, s)).to(device))
    return grads2

def get_gradient_norm(params):
    grad_all = torch.Tensor([0])
    
    #total_norm = 0.0
    for p in params:
        grad = torch.Tensor([0])
        
        if p.grad is not None:
            grad += torch.norm(p.grad)**2#p.grad.data.norm(2)
        grad.requires_grad = True
        grad_all += grad
    #grad_all.requires_grad = True
    #print(grad_all)
    return (grad_all ** 0.5).to(device)

def criterion(pred, true, params):
    org_loss = nn.MSELoss()
    loss = org_loss(pred, true)
    loss += get_gradient_norm(params)[0]
    return loss

#===============================================================================================

def main(args):
    ###Load dataset###
    train_x = np.random.rand(500000).reshape(500000,1)
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
    optimizer = torch.optim.Adagrad(train_model.parameters(), lr=0.0001)
    
    ###LOSS FUNCTION###
    loss_func = nn.MSELoss()
    
    ###LOSS RECORD###
    loss_history = []
    
    
    ###Training###    
    for e in range(EPOCH*0):
        
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
            #Hessian = np.array(get_second_order_grad(optimizer.param_groups[0]['params'], loss))
            #w_t = np.array(optimizer.param_groups[0]['params'])
            loss.backward()
            optimizer.step()
            print("Epoch", e, "Batch: ", b_num, "loss: ", loss.item(), end = '\r')
            epoch_loss += loss.item()
            
        #torch.save(train_model, 'hw1-2_task3_step1_function_model_'+str(e)+'.pkl')
        #torch.save(optimizer.state_dict(), 'hw1-2_task3_step1_function_model_'+str(e)+'.optim')
        #if e%100 == 0:
        print("")
        print("Epoch loss: ", epoch_loss / len(train_data) * BATCHSIZE)
        loss_history.append( epoch_loss / len(train_data) * BATCHSIZE)
    
    
    ###SECOND TRAINING###
    
    print('Second part...')
    
    ratio_list = []
    loss_list  = []
    ratio_list_2 = []
    loss_list_2 = []
    
    for e in range(EPOCH):
        
        #print("Epoch ", e)
        epoch_loss = 0
        
        for b_num, (b_x, b_y) in enumerate(train_dataloader):
            temp_ratio_list = []
            b_x = b_x.to(device)
            b_x = b_x.float()
            b_y = b_y.to(device)
            b_y = b_y.float()
            optimizer.zero_grad()
            pred = train_model(b_x)
            loss = loss_func(pred, b_y)
            loss.backward()
            loss_new = criterion(pred, b_y, optimizer.param_groups[0]['params'])
            """
            Hessian = np.array(get_second_order_grad(optimizer.param_groups[0]['params'], loss))
            #w_t = np.array(optimizer.param_groups[0]['params'])
            minimas = [np.linalg.eig(i.cpu())[0] for i in Hessian]
            tot_len = 0
            pos_eig = 0
            for i in minimas:
                tot_len += len(i)
                pos_eig += np.sum(np.array([1 if j > 0 else 0 for j in i]))
            temp_ratio_list.append(pos_eig/tot_len)
            """
            loss_new.backward()
            optimizer.step()
            print("Epoch", e, "Batch: ", b_num, "loss: ", loss.item(), end = '\r')
            epoch_loss += loss.item()
        """
        ratio_list.append(np.mean(np.array(temp_ratio_list)))
        loss_list.append(epoch_loss/len(train_data) * BATCHSIZE)
        ratio_list_2 += temp_ratio_list
        loss_list_2 += [epoch_loss/len(train_data) * BATCHSIZE for i in range(len(temp_ratio_list))]
        """ 
        torch.save(train_model, 'hw1-2_task3_step2_function_model.pkl')
        torch.save(optimizer.state_dict(), 'hw1-2_task3_step2_function_model.optim')
        
        print("")
        print("Epoch loss: ", epoch_loss / len(train_data) * BATCHSIZE)
        #loss_history.append( epoch_loss / len(train_data) * BATCHSIZE)
    """
    np.save('loss', np.array(loss_list))
    np.save('minimal_ratio', np.array(ratio_list))
    
    plt.figure(1)
    plt.scatter(ratio_list, loss_list)
    #plt.ylim(0,0.002)
    plt.savefig('task3.png', format='png')
    
    plt.figure(2)
    plt.scatter(ratio_list_2, loss_list_2)
    plt.savefig('task3_2.png', format='png')
    
    plt.show()
    """   
        
    return 
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='On function y = sin(5*pi*x)/(5*pi*x)')
    args = parser.parse_args()
    main(args)