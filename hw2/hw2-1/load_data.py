# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 14:12:18 2019

@author: Austin Hsu
"""

import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import torch.functional as F
import numpy as np
import argparse
import json
import os
import yaml

"""
###DATA LOADING PARAMS###
loading_params = {'directory': './MLDS_hw2_1_data', 'min_count':3, 'random_seed':1, 'batch_size':100}
"""

def sent2words(sent):
    res_para = [j.split(' ') for j in sent.lower()[:-1].split(',')]
    res_tok = res_para.pop(0)
    while len(res_para) != 0:
        res_tok.extend([','] + res_para.pop(0)[1:])
    for j in range(len(res_tok)):
        if res_tok[j][-2:] == "'s" and res_tok[j] != "'s":
            res_tok[j] = res_tok[j][:-2]
            res_tok.insert(j+1,"'s")
    return res_tok

def padding(caption_list, one_hot_len, max_len):
    padded_list = []
    padded_term = np.zeros((1, one_hot_len))
    for caption in caption_list:
        padding_len = max_len - len(caption)
        padded_list = caption + padded_term * padding_len
    return padded_list

def load_data(directory, min_count, random_seed, batch_size):
    print('Loading starts...')
    train_feat = dict()
    with open(os.path.join(directory, 'training_label.json'), 'r') as f:    
        train_labels = json.loads(f.read())
    for i in train_labels:
        train_feat[i['id']] = np.load(os.path.join(directory, 'training_data/feat', i['id'] + '.npy'))
    print('Word count starts...')
    word_dict = dict()
    for i in range(len(train_labels)):
        for sentence in range(len(train_labels[i]['caption'])):
            tokens = sent2words(train_labels[i]['caption'][sentence])
            for word in tokens:
                if word not in word_dict.keys():
                    word_dict[word] = 1
                else:
                    word_dict[word] += 1    
            tokens += ['<EOS>']
            train_labels[i]['caption'][sentence] = tokens
    pop_words = [i for i in word_dict.keys() if word_dict[i] <= min_count]
    for i in pop_words:
        word_dict.pop(i)
    one_hot_len = len(word_dict) + 3 #<BOS>, <EOS>, <UNK>
    print('Word count finished.')
    print('Total word count : %d'%one_hot_len)
    
    #Creating one hot vector
    for num, key in enumerate(word_dict):
        one_hot_vect = np.zeros((1, one_hot_len))
        one_hot_vect[0, num] = 1
        word_dict[key] = one_hot_vect#.tolist()
    
    for num, key in enumerate(['<EOS>', '<BOS>', '<UNK>']):
        one_hot_vect = np.zeros((1, one_hot_len))
        one_hot_vect[0, -num-1] = 1
        word_dict[key] = one_hot_vect#.tolist()
    
    #Convert to one hot vector
    max_len = 0
    train_x = []
    train_y = []
    for data_label in range(len(train_labels)):
        print('Converting data %d / %d' % (data_label+1, len(train_labels)), end = '\r')
        video_caption = train_labels[data_label]['caption']
        video_id      = train_labels[data_label]['id']
        caption_list  = []
        for single_sent in video_caption:
            sent_convert_list = [word_dict[word] if word in word_dict.keys() else word_dict['<UNK>'] for word in single_sent]
            max_len = max(max_len, len(sent_convert_list))
            caption_list.append(sent_convert_list)
        train_y.append(caption_list)
        train_x.append(train_feat[video_id])
    print("")
    
    #Padding
    datasize = len(train_y)
    padded_train_y = []
    for caption_list in range(len(train_y)):
        print('Padding data %d / %d' % (caption_list+1, len(train_y)), end = '\r')
        padded_train_y.append(padding(train_y[caption_list], one_hot_len, max_len))
    train_y = padded_train_y
    del padded_train_y
    print("")
    
    """
    ###DUMPING IS A RIDICULOUS IDEA###
    print('Dumping preprocessed data...')
    np.save('proc_train_x', np.array(train_x))
    np.save('proc_train_y', np.array(train_y))
    yaml.dump(word_dict, open('word_dict.yaml', 'w'))
    """
    
    #Random select caption
    if random_seed != None:
        np.random.seed(random_seed)
    train_y_chosen = []
    for captions in range(len(train_y)):
        print('Randomly picking data %d / %d' % (captions+1, len(train_y)), end = '\r')
        chosen = np.random.randint(len(train_y[captions]))
        train_y_chosen.append(train_y[captions][chosen])
    train_y = train_y_chosen
    del train_y_chosen
    print("")
    
    print("Max length : %d" % max_len)
    
    train_x = torch.Tensor(train_x)
    train_y = torch.Tensor(train_y)
    
    Dataset = Data.TensorDataset(train_x, train_y)
    DataLoader = Data.DataLoader(dataset = Dataset, batch_size=batch_size, shuffle=True, num_workers=1) 
    
    print('Loading finished.')            
    print("")
    for x,y in DataLoader:
        print(x.shape)
        break
    return DataLoader, one_hot_len, max_len, word_dict, datasize
        