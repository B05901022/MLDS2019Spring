#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 22:32:55 2019

@author: austinhsu
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

loss = np.load('loss.npy')
m_r = np.load('minimal_ratio.npy')

plt.scatter(m_r, loss_new)
plt.show()