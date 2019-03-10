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
pca=sklearn.decomposition.PCA(2)
for i in range(10):
    for k in range(1,9,1):
        a.append(torch.load("Weight_epoch"+str(3*i+2)+"event"+str(k)+".pkl"))
        for j in a[-1].parameters():
            pass
        b.append(j.cpu().detach().numpy())
cv=pca.fit_transform(b)
cv1=[]
cv2=[]
for s in range(10):
    for ss in range(8):
        cv1.append(cv[s][0])
for v in range(10):
    for vv in range(8):
        cv2.append(cv[v][0])
plt.text(cv1,cv2)
plt.savefig("pca.png")
