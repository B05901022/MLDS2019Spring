# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 21:52:11 2019

@author: Austin Hsu
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

rwd_history1 = np.load("./training_curve/hw4-1/PG_CV3_800.npy")
rwd_history2 = np.load("./training_curve/hw4-1/PG_CV3_1600.npy")

rwd_concat   = np.concatenate((rwd_history1, rwd_history2))
axis_concat  = np.arange(rwd_concat.shape[0])

acer_result  = pd.read_csv('./ACER_on_Pong.csv')
score = acer_result['mean_episode_reward'].to_numpy()
timestep = acer_result['total_timesteps'].to_numpy()
epi_len  = acer_result['mean_episode_length'].to_numpy()
episodes = (timestep/epi_len)
timesort = np.argsort(episodes)
epi_mean = np.mean(epi_len)

rwd_plot = pd.DataFrame(
    {"iteration": axis_concat,
     "reward": rwd_concat
    }
)

#sns.factorplot(data = rwd_plot, x="iteration", y="reward")
plt.figure()
plt.xlabel("episode")
plt.ylabel("mean reward")
plt.legend()
plt.plot(axis_concat, rwd_concat)
plt.plot(timestep/epi_mean, score)
#plt.plot(timestep, score)
plt.show()