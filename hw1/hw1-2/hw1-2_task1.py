# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 16:30:03 2019

@author: u8815
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 14:17:54 2019

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
import sklearn

###DEVICE###
device = torch.device("cpu")

###REPRODUCIBLE###
###HYPERPARAMETER###
EPOCH = 30
BATCHSIZE = 20000
ADAMPARAM = {'lr':0.001, 'betas':(0.9, 0.999), 'eps':1e-08}
DOWNLOAD_DATASET = True

"""
Total Parameter:
    net_shallow:72106
    net_deep:72030
"""

###SHALLOW CNN MODEL###
class net_shallow(nn.Module):
    def __init__(self):
        super(net_shallow, self).__init__()
        self.conv_layer = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=(3,3), stride=(1,1), padding=1), #32 * 32
                nn.BatchNorm2d(8),
                nn.ReLU(),
                nn.AvgPool2d((2,2)), #16 * 16
                nn.Conv2d(8, 16, kernel_size=(3,3), stride=(1,1), padding=1), #16 * 16
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.AvgPool2d((2,2)) #8 * 8
                )
        self.dnn_layer = nn.Sequential(
                nn.Linear(784, 40),
                nn.ReLU(),
                nn.Linear(40, 40),
                nn.ReLU(),
                nn.Linear(40, 10))
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.dnn_layer(x)
        return output

###DEEP CNN MODEL###
class net_deep(nn.Module):
    def __init__(self):
        super(net_deep, self).__init__()
        self.conv_layer = nn.Sequential(
                nn.Conv2d(3, 4, kernel_size=(3,3), stride=(1,1), padding=1), #32 * 32
                nn.BatchNorm2d(4),
                nn.ReLU(),
                nn.Conv2d(4, 4, kernel_size=(3,3), stride=(1,1), padding=1), #32 * 32
                nn.BatchNorm2d(4),
                nn.ReLU(),
                nn.Conv2d(4, 4, kernel_size=(3,3), stride=(1,1), padding=1), #32 * 32
                nn.BatchNorm2d(4),
                nn.ReLU(),
                nn.AvgPool2d((2,2)), #16 * 16
                nn.Conv2d(4, 8, kernel_size=(3,3), stride=(1,1), padding=1), #16 * 16
                nn.BatchNorm2d(8),
                nn.ReLU(),
                nn.Conv2d(8, 8, kernel_size=(3,3), stride=(1,1), padding=1), #16 * 16
                nn.BatchNorm2d(8),
                nn.ReLU(),
                nn.AvgPool2d((2,2)) #8 * 8
                )
        self.dnn_layer = nn.Sequential(
                nn.Linear(784, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Linear(16,10))
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
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

def main(args):
    ###Load CIFAR10 dataset###
    train_data = torchvision.datasets.MNIST(
            root = './CIFAR10_dataset',
            train = True,
            transform = torchvision.transforms.ToTensor(),
            download = DOWNLOAD_DATASET)
    """
    test_data = torchvision.datasets.CIFAR10(
            root = './CIFAR10_dataset',
            train = False)
    """
    ###DATALOADER###
    train_dataloader = Data.DataLoader(
            dataset = train_data,
            batch_size = BATCHSIZE,
            shuffle = True,
            num_workers = 1)
    
    ###LOAD MODEL###
    if args.model_type == 'shallow':
        train_model = net_shallow().to(device)
    elif args.model_type == 'deep':
        train_model = net_deep().to(device)
    
    ###OPTIMIZER###
    #optimizer = torch.optim.Adam(train_model.parameters(), lr=ADAMPARAM['lr'], betas=ADAMPARAM['betas'], eps=ADAMPARAM['eps'], weight_decay=1e-5)
    optimizer = torch.optim.SGD(train_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    
    ###LOSS FUNCTION###
    loss_func = nn.CrossEntropyLoss()
    
    ###RECORD###
    loss_history = []
    acc_history = []
    weight_history=[]
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
            epoch_acc  += torch.sum(torch.eq(torch.argmax(pred, dim=1), b_y), dim=0).item()
            
        torch.save(train_model, args.model_type+'_model.pkl')
        torch.save(optimizer.state_dict(), args.model_type+'_model.optim')
        print("")   
        print("Epoch loss: ", epoch_loss / len(train_data))
        print("Epoch acc:  ", epoch_acc  / len(train_data))
        print (e%3)
        if e%3==2:
            loss_history.append(epoch_loss / len(train_data))
            acc_history.append(epoch_acc  / len(train_data))
            torch.save(train_model,"Weight_epoch"+str(e)+"event4.pkl")
    ###LOSS HISTORY###
    np.save('CNN_'+args.model_type+'_loss_event4', np.array(loss_history))
    
    ###ACCURACY HISTORY###

    np.save('CNN_'+args.model_type+'_acc_event4', np.array(acc_history))
    
    return 
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', '-type', type=str, default='shallow')
    args = parser.parse_args()
    main(args)