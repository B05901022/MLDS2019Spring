# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 19:50:39 2019

@author: u8815
"""
import numpy as np
import matplotlib.pyplot as plt
loss=[]
val_loss=[]
acc=[]
val_acc=[]
batch=[]
sen=[]
for i in range(3,13,1):
    temp=np.load("model13_"+str(2**i)+".npy")
    loss.append(temp[0])
    acc.append(temp[1])
    val_loss.append(temp[2])
    val_acc.append(temp[3])
    temp2=np.load("norm_model_"+str(2**i)+".npy")
    batch.append(np.log10(2**i))
    sen.append(temp2)
loss=np.array(loss)
acc=np.array(acc)
val_loss=np.array(val_loss)
val_acc=np.array(val_acc)
sen=np.array(sen)
batch=np.array(batch)
fig, ax1, = plt.subplots()
plt.figure(1)
color = 'tab:red'
ax1.set_xlabel('batch')
ax1.set_ylabel('sensitivity', color=color)
ax1.plot(batch,sen, color=color,label="sensitivity")
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('cross_entropy', color=color)  # we already handled the x-label with ax1
ax2.plot(batch,loss, color=color)
ax2.plot(batch,val_loss,'--', color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly 
ax1.legend()
plt.savefig("sensitive and cross_entropy.jpg")
fig2, ax3, = plt.subplots()
plt.figure(2)
color = 'tab:red'
ax3.set_xlabel('batch')
ax3.set_ylabel('sensitivity', color=color)
ax3.plot(batch,sen, color=color,label="sensitivity")
ax3.tick_params(axis='y', labelcolor=color)

ax4 = ax3.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax4.set_ylabel('accuracy', color=color)  # we already handled the x-label with ax1
ax4.plot(batch,acc, color=color)
ax4.plot(batch,val_acc,'--', color=color)
ax4.tick_params(axis='y', labelcolor=color)

fig2.tight_layout()  # otherwise the right y-label is slightly clipped
ax3.legend()
plt.savefig("sensitive and acc.jpg")
