# -*- coding: utf-8 -*-
"""
Created on Thu May  2 20:32:04 2019

@author: Austin Hsu
"""

import numpy as np
from gensim.models import word2vec
from tqdm import tqdm
from sklearn.preprocessing import normalize
from transformer_tutorial import make_model, subsequent_mask, Generator

import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
import torch.functional as F
import argparse
import os

def load_test_dataset(model_name='../word2vec_only_train.model',
                      directory='../../../../MLDS_dataset/hw2-2/evaluation/',
                      pad_len=20,
                      min_len=2,
                      ):
    
    dataset = open(os.path.join(directory,'input.txt'), 'r', encoding='UTF-8').read().split('\n')
    
    w2v_model = word2vec.Word2Vec.load(model_name)
    
    embedding_matrix = np.zeros((len(w2v_model.wv.vocab.items()) + 4, w2v_model.vector_size))
    word2idx = {}

    vocab_list = [(word, w2v_model.wv[word]) for word, _ in w2v_model.wv.vocab.items()]
    for i, vocab in enumerate(vocab_list):
        word, vec = vocab
        embedding_matrix[i + 4] = vec
        word2idx[word] = i + 4
    word2idx['<PAD>'] = 0
    word2idx['<BOS>'] = 1
    word2idx['<EOS>'] = 2
    word2idx['<UNK>'] = 3
    
    print("Loading dataset...")
    test_x = []
    
    for sent in dataset:
        sent_len = len(sent)
        padded_sent = [1]
        padded_sent += sent2idx(sent, word2idx)
        padded_sent += [2]
        padded_sent += [0] * (pad_len - sent_len - 2)
        test_x.append(padded_sent)
    
    print("Loading finished.")  
    
    test_x = np.array(test_x)
    
    return test_x, word2idx

def sent2idx(sentence,
             word2idx,
             ):
    idxsent = []
    for word in sentence:
        try:
            idxsent.append(word2idx[word])
        except:
            idxsent.append(3)
    return idxsent

def inference():
    
    test_x, w2i = load_test_dataset()
    i2w = {w2i[i]:i for i in w2i.keys()}
    tensor_x = torch.stack([torch.from_numpy(np.array(i)) for i in test_x])
    Transformer_model = torch.load('../../../../MLDS_models/hw2-2/epoch_0_Transformer_model.pkl')
    Transformer_model.eval()
    
    results = []
    
    for one_sent in tensor_x:
        mask = subsequent_mask(20).long().cuda()
        test_y = torch.Tensor([[1] + [0]*19]).long().cuda()
        test_sent = one_sent.unsqueeze(0).long().cuda()
        
        for i in range(20):
            
            test_y = Transformer_model(test_sent, test_y, mask, mask)
            test_y = torch.argmax(Transformer_model.generator(test_y), dim=2)
        result = test_y.cpu().numpy()[0]
        result = [i2w[word] for word in result]
        results.append(result)
  
    return results
        
