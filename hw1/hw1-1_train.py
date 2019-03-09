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

###DEVICE###
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###REPRODUCIBLE###
torch.manual_seed(1)

###HYPERPARAMETER###
EPOCH = 200
BATCHSIZE = 500
ADAMPARAM = {'lr':0.0001, 'betas':(0.9, 0.999), 'eps':1e-08}
DOWNLOAD_DATASET = True

###SHALLOW CNN MODEL###
class shallow(nn.Module):
    def __init__(self):
        super(shallow, self).__init__()
        self.conv_layer = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=(3,3), stride=(1,1), padding=1), #32 * 32
                nn.BatchNorm2d(16),
                nn.LeakyReLU(),
                nn.Conv2d(16, 16, kernel_size=(3,3), stride=(1,1), padding=1), #32 * 32
                nn.BatchNorm2d(16),
                nn.LeakyReLU(),
                nn.AvgPool2d((2,2)), #16 * 16
                nn.Conv2d(16, 32, kernel_size=(3,3), stride=(1,1), padding=1), #16 * 16
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
                nn.Conv2d(32, 32, kernel_size=(3,3), stride=(1,1), padding=1), #16 * 16
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
                nn.AvgPool2d((2,2)) #8 * 8
                )
        self.dnn_layer = nn.Sequential(
                nn.Linear(32*8*8, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(),
                nn.Linear(128, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(),
                nn.Linear(128, 10))
        
    def forward(self, x):
        x = self.conv_layer(x)
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

def main():
    ###Load CIFAR10 dataset###
    train_data = torchvision.datasets.CIFAR10(
            root = './CIFAR10_dataset',
            train = True,
            transform = torchvision.transforms.ToTensor(),
            download = DOWNLOAD_DATASET)

    test_data = torchvision.datasets.CIFAR10(
            root = './CIFAR10_dataset',
            train = False)

    ###DATALOADER###
    train_dataloader = Data.DataLoader(
            dataset = train_data,
            batch_size = BATCHSIZE,
            shuffle = True,
            num_workers = 1)
    
    ###LOAD MODEL###
    train_model = shallow().to(device)
    
    ###OPTIMIZER###
    optimizer = torch.optim.SGD(train_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    
    ###LOSS FUNCTION###
    loss_func = nn.CrossEntropyLoss()
    
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
            epoch_acc  += torch.sum(torch.eq(pred, b_y), dim=0).item() / BATCHSIZE 
            
        torch.save(train_model, 'shallow_model.pkl')
        torch.save(optimizer.state_dict(), 'shallow_model.optim')
        print("")
        print("Epoch loss: ", epoch_loss / BATCHSIZE)
        print("Epoch acc:  ", epoch_acc  / BATCHSIZE)       
        
    return 
    

if __name__ == '__main__':
    main()