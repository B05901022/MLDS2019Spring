# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 19:15:26 2019

@author: Austin Hsu
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

results = pd.read_csv(open('sharpness.csv', 'r'))

results_1 = results[:8]
results_2 = results[8:]

results_plot = results_1

plot_1 = sns.pointplot("batch_size", "train_acc", data = results_plot, color='blue', label = 'train acc', logx=True)

plot_2 = sns.pointplot("batch_size", "test_acc", data = results_plot, color='blue', label = 'test acc', logx=True, linestyles='--')

plot_1.set(xlabel='batch_size(log scale)', ylabel='acc')

plot_2.set(xlabel='batch_size(log scale)', ylabel='acc')

plott = plot_1.twinx()

plot_3 = sns.pointplot("batch_size", "sharpness", data = results_plot, color='red', label = 'sharpness', logx=True)

plot_3.set(xlabel='batch_size(log scale)', ylabel='sharpness')

#sns.pointplot("batch_size", "sharpness", data = results_2, logx=True)
