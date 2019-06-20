#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 07:56:29 2019

@author: leo
"""
import os
from agent_dir.agent import Agent
from agent_dir.Model import ActorCritic
#import scipy
import numpy as np
import skimage
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from environment import Environment
from tqdm import tqdm
import collections
device="cuda:0"
def prepro(o,
           image_size=[105,80],#[80,80],
           ):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
    
    Input: 
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array 
        Grayscale image, shape: (80, 80, 1)
    
    """
    
    #y = o.astype(np.uint8)
    #resized = scipy.misc.imresize(y, image_size)

    y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]   #gray scale
    resized = skimage.transform.resize(y, image_size)[17:-8,:]            #delete score board
    
    return np.expand_dims(resized.astype(np.float32),axis=2)              #shape (height, wodth) -> (1, height, wodth)
class Agent_AC(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_AC,self).__init__(env)
        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            model = torch.load(args.model_name+".ckpt")
            self.model = ActorCritic()
            self.model.load_state_dict(model['model'].state_dict())
            self.hyper_param = args.__dict__
            if self.model.noisy:
                self.model.linear_1[0].remove_noise()
                self.model.advantage.remove_noise()
                if self.model.dueling:
                    self.model.value.remove_noise()
            self.model = self.model.to(device)
            
        elif args.train_dqn:
            if args.load_model:
                model = torch.load(args.model_name+".ckpt")  # dictionary for checkpoint
                self.model = model['model']
                self.target_net = model['target_net']
                #self.update_target_net()
                self.step_count = 0
                if model['epsilon'] == True:
                    self.epsilon = model['epsilon_value']
                
                self.replay_buffer_len = 10000
                self.replay_buffer = collections.deque([], maxlen = self.replay_buffer_len) 
                self.optimizer = ['Adam', 'RMSprop', 'SGD']
                
                self.training_curve = model['curve']
                self.hyper_param = args.__dict__
                
                if self.hyper_param['optim'] in self.optimizer:
                    if self.hyper_param['optim'] == 'Adam':
                        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                                          lr = self.hyper_param['learning_rate'],
                                                          betas = (0.9, 0.999))
                    elif self.hyper_param['optim'] == 'RMSprop':
                        self.optimizer = torch.optim.RMSprop(self.model.parameters(),
                                                          lr = self.hyper_param['learning_rate'],
                                                          alpha = 0.9)
                    elif self.hyper_param['optim'] == 'SGD':
                        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                                          lr = self.hyper_param['learning_rate'])
                    #state_dict = torch.load(args.model_name+".optim")
                    #self.optimizer.load_state_dict(state_dict)
                else:
                    print("Unknown Optimizer or missing state_dict!")
                    exit()
                self.env.clip_reward = False
            elif args.new_model:
                if os.path.isfile(args.model_name+".pkl"):
                    # model name conflict
                    confirm = input('Model \'{}\' already exists. Overwrite it? [y/N] ' \
                                    .format(args.model_name.strip('.pkl')))

                    if confirm not in ['y', 'Y', 'yes', 'Yes', 'yeah']:
                        print('Process aborted.')
                        exit()
 
                self.model = ActorCritic
                #self.target_net = DQN(84, 84, args.Dueling, args.Noisy)
                self.update_target_net()
                self.step_count = 0
                if args.epsilon:
                    self.epsilon = 1
                
                self.replay_buffer_len = 10000
                self.replay_buffer = collections.deque([], maxlen = self.replay_buffer_len)
                self.optimizer = ['Adam', 'RMSprop', 'SGD']
                
                self.training_curve = []
                self.hyper_param = args.__dict__
                
                if self.hyper_param['optim'] in self.optimizer:
                    if self.hyper_param['optim'] == 'Adam':
                        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                                          lr = self.hyper_param['learning_rate'],
                                                          betas = (0.9, 0.999))
                    elif self.hyper_param['optim'] == 'RMSprop':
                        self.optimizer = torch.optim.RMSprop(self.model.parameters(),
                                                          lr = self.hyper_param['learning_rate'],
                                                          alpha = 0.9)
                    elif self.hyper_param['optim'] == 'SGD':
                        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                                          lr = self.hyper_param['learning_rate'])
                else:
                    print("Unknown Optimizer!")
                    exit()
                self.env.clip_reward = False
        print(self.model)
        ##################
        # YOUR CODE HERE #
        ##################


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        
        pass



    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################

        batch_size = self.hyper_param['batch_size']
        batch_data = None
        lastest_r = collections.deque([], maxlen = 100)
        for e in range(self.hyper_param['episode']):
            o = self.env.reset()
            o = prepro(o)
            unclipped_episode_r = 0
            while True:
                a = self.make_action(o)
                o_next, r, done, _ = self.env.step(a+1) # map 0,1,2 to 1,2,3
                o_next = prepro(o_next)
                
                unclipped_episode_r += r
                r = np.sign(r)
                #print(r)
                state_reward_tuple = (o, a, r, o_next, done)
                o = o_next
                self.step_count += 1
                if self.hyper_param['epsilon']:
                    self.update_epsilon()
                
                # push to replay buffer
                self.replay_buffer.append(state_reward_tuple)
                
                # get batch data and update current net
                if len(self.replay_buffer) > batch_size and self.step_count%4 == 0:
                    if not self.hyper_param['base']:
                        batch_data = random.sample(self.replay_buffer, batch_size)
                        loss = self.update_param_DDQN(batch_data)  
                        batch_data = None
                        print("Loss: %4f" % (loss), end = '\r')
                    elif self.hyper_param['base']:
                        batch_data = random.sample(self.replay_buffer, batch_size)
                        loss = self.update_param_base_DQN(batch_data)  
                        batch_data = None
                        print("Loss: %4f" % (loss), end = '\r')
                # update target net every 1000 step
                if self.step_count % 1000 == 0:
                    print("Update target net")
                    self.update_target_net()
                
                # if game is over, print mean reward and 
                if done:
                    lastest_r.append(unclipped_episode_r)
                    print("Episode : %d Mean : %4f Lastest : %4f" % \
                          (e+1, np.mean(lastest_r), unclipped_episode_r), end = '\n')
                    self.training_curve.append(np.mean(lastest_r))
                    break
                
            unclipped_episode_r = 0
            batch_data = None
            # save model every 500 episode
            if (e+1)%500 == 0:
                self.save_checkpoint(episode = e+1)
def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        
        if test:
            if type(self.last_frame) == type(None):
                observation = prepro(observation)
                self.last_frame = observation
            else:
                o = prepro(observation)
                observation = o - self.last_frame
                self.last_frame = o
                
            """
            observation = torch.Tensor(observation).view(1,-1).cuda()
            prob = self.model[0](observation)
            """
            
             
            observation = torch.Tensor(observation).cuda() # (80,80,1)
            action,prob = self.model(observation)
            action=torch.argmax(action).item()
            return action, prob #self.env.get_random_action()
                
        else:

            """
            observation = torch.Tensor(observation).view(1,-1).cuda()
            prob = self.model[0](observation)
            """
            
            observation = torch.Tensor(observation).cuda() # (80,80,1)
            action,prob = self.model(observation)
            action=torch.argmax(action).item()
            return action, prob #self.env.get_random_action()
                

            