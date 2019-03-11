# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 20:44:13 2019

@author: u8815
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
a=[]
b=[]
x=[3,6,9,12,15,18,21,24,27,30]
for k in range(1,9,1):
    a.append(np.load("CNN_shallow_acc_event"+str(k)+".npy"))
    b.append(np.load("CNN_shallow_loss_event"+str(k)+".npy"))
plt.figure(1)
plt.plot(x,a[0],c="black")
plt.plot(x,a[1],c="blue")
plt.plot(x,a[2],c="orange")
plt.plot(x,a[3],c="yellow")
plt.plot(x,a[4],c="gray")
plt.plot(x,a[5],c="cyan")
plt.plot(x,a[6],c="green")
plt.plot(x,a[7],c="purple")
plt.savefig("acc.png")
plt.figure(2)
plt.plot(x,b[0],c="black")
plt.plot(x,b[1],c="blue")
plt.plot(x,b[2],c="orange")
plt.plot(x,b[3],c="yellow")
plt.plot(x,b[4],c="gray")
plt.plot(x,b[5],c="cyan")
plt.plot(x,b[6],c="green")
plt.plot(x,b[7],c="purple")
plt.savefig("loss.png")