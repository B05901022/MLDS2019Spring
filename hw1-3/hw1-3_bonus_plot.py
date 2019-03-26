#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 11:28:07 2019

@author: austinhsu
"""

import seaborn as sns
import pandas as pd

results = pd.read_csv(open('sharpness.csv', 'r'))

results_1 = results[:8]
results_2 = results[8:]

results_plot = results_2

plot_1 = sns.pointplot("batch_size", "train_loss", data = results_plot, color='blue', logx=True)

plot_2 = sns.pointplot("batch_size", "test_loss", data = results_plot, color='blue', logx=True, linestyles='--')

plot_1.set(xlabel='batch_size(log scale)', ylabel='loss')

plot_2.set(xlabel='batch_size(log scale)', ylabel='loss')

plott = plot_1.twinx()

plot_3 = sns.pointplot("batch_size", "sharpness", data = results_plot, color='red', logx=True)

plot_3.set(xlabel='batch_size(log scale)', ylabel='sharpness')

#sns.pointplot("batch_size", "sharpness", data = results_2, logx=True)
