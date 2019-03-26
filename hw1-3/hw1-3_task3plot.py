# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 19:50:39 2019

@author: u8815
"""
import numpy as np
import matplotlib.pyplot as plt
acc_train=np.load("acc_train.npy")
acc_test=np.load("acc_test.npy")
cross_entropy_train=np.load("cross_entropy_train.npy")
cross_entropy_test=np.load("cross_entropy_test.npy")
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('alpha')
ax1.set_ylabel('cross_entropy', color=color)
ax1.plot(cross_entropy_train[:,1], cross_entropy_train[:,0], color=color,label="train")
ax1.plot(cross_entropy_test[:,1], cross_entropy_test[:,0],'--', color=color,label="test")
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('accuracy', color=color)  # we already handled the x-label with ax1
ax2.plot(cross_entropy_test[:,1], acc_train, color=color)
ax2.plot(cross_entropy_test[:,1], acc_test,'--', color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
ax1.legend()
plt.savefig("acc and cross_entropy.jpg")