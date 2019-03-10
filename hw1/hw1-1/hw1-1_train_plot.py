#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 05:11:55 2019

@author: austinhsu
"""

import numpy as np
import matplotlib.pyplot as plt

shallow_loss = np.load('CNN_shallow_loss.npy')
deep_loss = np.load('CNN_deep_loss.npy')

shallow_acc = np.load('CNN_shallow_acc.npy')
deep_acc = np.load('CNN_deep_acc.npy')

plt.figure(1)
plt.plot(np.arange(100)+1, shallow_loss, color='blue', label='shallow')
plt.plot(np.arange(100)+1, deep_loss, color='red', label='deep')
plt.legend(loc='upper right')
plt.savefig('pictures/CNN_loss_comparison.png', format='png')

plt.figure(2)
plt.plot(np.arange(100)+1, shallow_acc, color='blue', label='shallow')
plt.plot(np.arange(100)+1, deep_acc, color='red', label='deep')
plt.legend(loc='lower right')
plt.savefig('pictures/CNN_acc_comparison.png', format='png')