# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 15:21:23 2019

@author: Austin Hsu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

shallow_model = torch.load('function_0_shallow_model.pkl')
medium_model = torch.load('function_0_medium_model.pkl')
deep_model = torch.load('function_0_deep_model.pkl')

shallow_model_1 = torch.load('function_1_shallow_model.pkl')
medium_model_1 = torch.load('function_1_medium_model.pkl')
deep_model_1 = torch.load('function_1_deep_model.pkl')

train_x = np.random.rand(1000).reshape(1000,1)
train_y = np.sin(20*np.cos(train_x))
train_z = np.cos(-17*train_x)+np.sin(-20*train_x)

train_x = torch.from_numpy(train_x).to(device)
train_y = torch.from_numpy(train_y).to(device)
train_z = torch.from_numpy(train_z).to(device)

train_x = train_x.float()

pred_shallow = shallow_model(train_x)
pred_medium  = medium_model(train_x)
pred_deep    = deep_model(train_x)

pred_shallow_1 = shallow_model_1(train_x)
pred_medium_1  = medium_model_1(train_x)
pred_deep_1    = deep_model_1(train_x)

plt.figure(1)
plt.scatter(train_x.cpu().detach().numpy(), train_y.cpu().detach().numpy(), color='black', s = 5, label='label')
plt.scatter(train_x.cpu().detach().numpy(), pred_shallow.cpu().detach().numpy(), color='red', s = 5, label='shallow')
plt.scatter(train_x.cpu().detach().numpy(), pred_medium.cpu().detach().numpy(), color='green', s = 5, label='medium')
plt.scatter(train_x.cpu().detach().numpy(), pred_deep.cpu().detach().numpy(), color='blue', s = 5, label='deep')
plt.legend(loc='lower left')
plt.savefig('pictures/function_output_0_comparison.png', format='png')

plt.figure(2)
plt.scatter(train_x.cpu().detach().numpy(), train_z.cpu().detach().numpy(), color='black', s = 5, label='label')
plt.scatter(train_x.cpu().detach().numpy(), pred_shallow_1.cpu().detach().numpy(), color='red', s = 5, label='shallow')
plt.scatter(train_x.cpu().detach().numpy(), pred_medium_1.cpu().detach().numpy(), color='green', s = 5, label='medium')
plt.scatter(train_x.cpu().detach().numpy(), pred_deep_1.cpu().detach().numpy(), color='blue', s = 5, label='deep')
plt.legend(loc='upper right')
plt.savefig('pictures/function_output_1_comparison.png', format='png')

shallow_loss = np.load('function_0_shallow_loss.npy')
medium_loss = np.load('function_0_medium_loss.npy')
deep_loss = np.load('function_0_deep_loss.npy')

shallow_loss_1 = np.load('function_1_shallow_loss.npy')
medium_loss_1 = np.load('function_1_medium_loss.npy')
deep_loss_1 = np.load('function_1_deep_loss.npy')

plt.figure(3)
plt.plot(np.arange(10)+1, shallow_loss, color='red', label='shallow')
plt.plot(np.arange(10)+1, medium_loss, color='green', label='medium')
plt.plot(np.arange(10)+1, deep_loss, color='blue', label='deep')
plt.legend(loc='upper right')
plt.savefig('pictures/function_loss_0_comparison.png', format='png')

plt.figure(4)
plt.plot(np.arange(10)+1, shallow_loss_1, color='red', label='shallow')
plt.plot(np.arange(10)+1, medium_loss_1, color='green', label='medium')
plt.plot(np.arange(10)+1, deep_loss_1, color='blue', label='deep')
plt.legend(loc='upper right')
plt.savefig('pictures/function_loss_1_comparison.png', format='png')





