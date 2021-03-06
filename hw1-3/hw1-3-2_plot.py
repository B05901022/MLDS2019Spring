import numpy as np
import torch
import matplotlib.pyplot as plt

DNN_1_acc = np.load('DNN_1_acc.npy')
DNN_5_acc = np.load('DNN_5_acc.npy')
DNN_7_acc = np.load('DNN_7_acc.npy')
DNN_11_acc = np.load('DNN_11_acc.npy')
DNN_12_acc = np.load('DNN_12_acc.npy')
DNN_15_acc = np.load('DNN_15_acc.npy')
DNN_16_acc = np.load('DNN_16_acc.npy')
DNN_18_acc = np.load('DNN_18_acc.npy')
DNN_20_acc = np.load('DNN_20_acc.npy')
DNN_23_acc = np.load('DNN_23_acc.npy')
DNN_26_acc = np.load('DNN_26_acc.npy')
DNN_25_acc = np.load('DNN_25_acc.npy')
DNN_22_acc = np.load('DNN_22_acc.npy')
DNN_24_acc = np.load('DNN_24_acc.npy')
DNN_21_acc = np.load('DNN_21_acc.npy')

DNN_1_loss = np.load('DNN_1_loss.npy')
DNN_5_loss = np.load('DNN_5_loss.npy')
DNN_7_loss = np.load('DNN_7_loss.npy')
DNN_11_loss = np.load('DNN_11_loss.npy')
DNN_12_loss = np.load('DNN_12_loss.npy')
DNN_15_loss = np.load('DNN_15_loss.npy')
DNN_16_loss = np.load('DNN_16_loss.npy')
DNN_18_loss = np.load('DNN_18_loss.npy')
DNN_20_loss = np.load('DNN_20_loss.npy')
DNN_23_loss = np.load('DNN_23_loss.npy')
DNN_26_loss = np.load('DNN_26_loss.npy')
DNN_25_loss = np.load('DNN_25_loss.npy')
DNN_22_loss = np.load('DNN_22_loss.npy')
DNN_24_loss = np.load('DNN_24_loss.npy')
DNN_21_loss = np.load('DNN_21_loss.npy')

param = np.array([ 811, 1599, 1704, 6274, 9854, 12384, 19490, 25374, 25914, 29910, 36121, 38518, 39738, 44652, 51610 ])
acc_list_train = np.array([ DNN_1_acc[0], DNN_5_acc[0], DNN_7_acc[0], DNN_11_acc[0], DNN_12_acc[0], DNN_15_acc[0], DNN_16_acc[0], DNN_18_acc[0], DNN_20_acc[0], DNN_23_acc[0], DNN_26_acc[0], DNN_25_acc[0], DNN_22_acc[0], DNN_24_acc[0], DNN_21_acc[0] ])
acc_list_test = np.array([ DNN_1_acc[1], DNN_5_acc[1], DNN_7_acc[1], DNN_11_acc[1], DNN_12_acc[1], DNN_15_acc[1], DNN_16_acc[1], DNN_18_acc[1], DNN_20_acc[1], DNN_23_acc[1], DNN_26_acc[1], DNN_25_acc[1], DNN_22_acc[1], DNN_24_acc[1], DNN_21_acc[1] ])

loss_list_train = np.array([ DNN_1_loss[0], DNN_5_loss[0], DNN_7_loss[0], DNN_11_loss[0], DNN_12_loss[0], DNN_15_loss[0], DNN_16_loss[0], DNN_18_loss[0], DNN_20_loss[0], DNN_23_loss[0], DNN_26_loss[0], DNN_25_loss[0], DNN_22_loss[0], DNN_24_loss[0], DNN_21_loss[0] ])
loss_list_test = np.array([ DNN_1_loss[1], DNN_5_loss[1], DNN_7_loss[1], DNN_11_loss[1], DNN_12_loss[1], DNN_15_loss[1], DNN_16_loss[1], DNN_18_loss[1], DNN_20_loss[1], DNN_23_loss[1], DNN_26_loss[1], DNN_25_loss[1], DNN_22_loss[1], DNN_24_loss[1], DNN_21_loss[1] ])

plt.figure(1)
plt.scatter( param, acc_list_train, color = 'blue')
plt.scatter( param, acc_list_test, color = 'orange')
plt.show()

plt.figure(2)
plt.scatter( param, loss_list_train, color = 'blue')
plt.scatter( param, loss_list_test, color = 'orange')
plt.ylim(top = 0.004)
plt.ylim(bottom = -0.001)
plt.show()
