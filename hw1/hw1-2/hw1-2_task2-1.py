# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 16:34:40 2019
@author: Austin Hsu
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
ADAMPARAM = {'lr':0.001, 'betas':(0.9, 0.999), 'eps':1e-08}

"""
Total Parameter:
    net_shallow:385
    net_medium:385
    net_deep:385
    
"""

###SHALLOW MODEL###
class net_shallow(nn.Module):
    def __init__(self):
        super(net_shallow, self).__init__()
        self.layer1 = nn.Linear(1,128)
        self.layer2 = nn.Linear(128,1)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        output = self.layer2(x)
        return output
    
###MDEIUM MODEL###
class net_medium(nn.Module):
    def __init__(self):
        super(net_medium, self).__init__()
        self.layer1 = nn.Linear(1, 16)
        self.layer2 = nn.Linear(16, 16)
        self.layer3 = nn.Linear(16, 4)
        self.layer4 = nn.Linear(4, 2)
        self.layer5 = nn.Linear(2, 1)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        output = self.layer5(x)
        return output
    
###Deep MODEL###
class net_deep(nn.Module):
    def __init__(self):
        super(net_deep, self).__init__()
        self.layer1 = nn.Linear(1, 8)
        self.layer2 = nn.Linear(8, 8)
        self.layer3 = nn.Linear(8, 8)
        self.layer4 = nn.Linear(8, 8)
        self.layer5 = nn.Linear(8, 8)
        self.layer6 = nn.Linear(8, 4)
        self.layer7 = nn.Linear(4, 4)
        self.layer8 = nn.Linear(4, 4)
        self.layer9 = nn.Linear(4, 1)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        x = F.relu(self.layer6(x))
        x = F.relu(self.layer7(x))
        x = F.relu(self.layer8(x))
        output = self.layer9(x)
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

def exp_func(x, function_no):
    #return np.sin(x*5)
    if function_no == '0':
        return np.sin(20 * np.cos(x))
    if function_no == '1':
        return np.cos(-17*x)+np.sin(-20*x)

def main(args):
    ###Load dataset###
    train_x = np.random.rand(500000).reshape(500000,1)
    train_y = torch.from_numpy(exp_func(train_x, args.function))
    train_x = torch.from_numpy(train_x)
    train_data = Data.TensorDataset(train_x, train_y) 

    ###DATALOADER###
    train_dataloader = Data.DataLoader(
            dataset = train_data,
            batch_size = BATCHSIZE,
            shuffle = True,
            num_workers = 1)
    
    ###LOAD MODEL###
    if args.model_type == 'shallow':
        train_model = net_shallow().to(device)
    elif args.model_type == 'medium':
        train_model = net_medium().to(device)
    elif args.model_type == 'deep':
        train_model = net_deep().to(device)
    
    ###OPTIMIZER###
    optimizer = torch.optim.Adam(train_model.parameters(), lr=ADAMPARAM['lr'], betas=ADAMPARAM['betas'], eps=ADAMPARAM['eps'], weight_decay=1e-5)
    #optimizer = torch.optim.Adagrad(train_model.parameters(), lr=0.001)
    
    ###LOSS FUNCTION###
    loss_func = nn.MSELoss()
    
    ###LOSS RECORD###
    loss_history = []
    
    grad_history = []
    
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
            loss.backward()
            optimizer.step()
            print("Epoch", e, "Batch: ", b_num, "loss: ", loss.item(), end = '\r')
            epoch_loss += loss.item()
            
            grad_all = 0.0
            grad_norm = 0.0
            #total_norm = 0.0
            for p in train_model.parameters():
                grad = 0.0
                #param_norm = 0.0
                if p.grad is not None:
                    #grad = (p.grad.cpu().data.numpy() ** 2).sum()
                    grad = p.grad.data.norm(2)
                    #total_norm += param_norm.item() ** 2
                grad_all += grad
                #total_norm = total_norm ** (1. / 2)
            grad_norm = grad_all ** 0.5
            grad_history.append(grad_norm)
            
        torch.save(train_model, 'function_'+ args.function + '_' +args.model_type+'_model.pkl')
        torch.save(optimizer.state_dict(), 'function_'+ args.function + '_'+args.model_type+'_model.optim')
        #if e%100 == 0:
        print("")
        print("Epoch loss: ", epoch_loss / len(train_data))
        loss_history.append( epoch_loss / len(train_data))
    
    ###LOSS HISTORY###
    plt.figure(1)
    plt.plot(np.arange(EPOCH)+1, loss_history)
    plt.savefig('pictures/'+ args.function + '_' + args.model_type + '_loss.png', format='png')
    np.save('function_' + args.function + '_' + args.model_type + '_loss', np.array(loss_history))
    
    ###Testing###
    plt.figure(2)
    test_x = train_x[:500].to(device)
    test_x = test_x.float()
    test_y = train_y[:500]
    pred_y = train_model(test_x)
    plt.scatter(test_x.cpu().detach().numpy(), test_y.cpu().detach().numpy(), color='blue', label='label')
    plt.scatter(test_x.cpu().detach().numpy(), pred_y.cpu().detach().numpy(), color='red', label='pred')
    plt.savefig('pictures/'+ args.function + '_'+args.model_type +'_function.png', format='png')
    
    ###GRADIENT NORM HISTORY#
    plt.figure(3)
    plt.plot(np.arange(EPOCH*1000)+1, grad_history)
    plt.savefig('pictures'+args.model_type+'_grad.png', format='png')
    np.save('CNN_'+args.model_type+'_grad', np.array(grad_history))
        
    return 
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='On function y = sin(20*cos(x))')
    parser.add_argument('--model_type', '-type', type=str, default='shallow')
    parser.add_argument('--function', '-func', type=str, default='0')
    args = parser.parse_args()
    main(args)
