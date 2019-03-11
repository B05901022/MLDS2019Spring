# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 18:58:14 2019

@author: u8815
"""
import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import torch.functional as F
import numpy as np
import argparse
import sklearn.decomposition
a=[]
b=[]
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
                nn.Linear(784, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 10))
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.dnn_layer(x)
        return output

pca=sklearn.decomposition.PCA(2)
for k in range(1,9,1):
    for i in range(10):
        a.append(torch.load("Weight_epoch"+str(3*i+2)+"event"+str(k)+".pkl"))
        for j in a[-1].parameters():
            pass
        b.append(j.cpu().detach().numpy())
cv=pca.fit_transform(b)
cv1=[]
cv2=[]
for s in range(8):
    for ss in range(10):
        cv1.append(cv[10*s+ss][0])
for v in range(8):
    for vv in range(10):
        cv2.append(cv[10*v+vv][1])
plt.text(cv1,cv2)
plt.savefig("pca.png")
