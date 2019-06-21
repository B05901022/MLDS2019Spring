import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import yaml

rwd_history1 = yaml.load(open("./model/hw4-2/DQN_Nxx_19600.yaml",'r'))
rwd_history2 = yaml.load(open("./model/hw4-2/DQN_xDx_19600.yaml",'r'))
rwd_history3 = yaml.load(open("./model/hw4-2/DQN_xxD_19600.yaml",'r'))
rwd_history4 = yaml.load(open("./model/hw4-2/DQN_NDx_19600.yaml",'r'))

plt.figure(1)
plt.xlabel("episode")
plt.ylabel("mean reward")
plt.plot(rwd_history1['traincurve'], color='red',   label='Noisy')
plt.plot(rwd_history2['traincurve'], color='green', label='Dueling')
plt.plot(rwd_history3['traincurve'], color='blue',  label='Double')
#plt.plot(rwd_history4['traincurve'], color='orange',label='Noisy+Dueling')
plt.legend()
plt.show()