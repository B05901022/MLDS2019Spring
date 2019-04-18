# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:22:07 2019

@author: Austin Hsu
"""

import numpy as np
import tensorflow as tf
from gensim.models import word2vec
from tqdm import tqdm
from tensorflow.keras.layers import Embedding
from sklearn.preprocessing import normalize

def word2vec_model(directory='../../../MLDS_dataset/hw2-2/clr_conversation.txt',
                   model_name='word2vec_only_train.model',
                   pre=False
                   ):
    
    """
    Parse the TRAINING data and generate an embedding relationship.
    input: directory of dataset, 
           model name of word to vector model,
           prebuild model or not(False: generate a new model)
    output: parsed dataset, word to vector model
    
    TODO: Make a similar function for test data parsing
    """
    
    dataset = open(directory, 'r', encoding='UTF-8').read().split('+++$+++')
    dataset = [[j.split(' ') for j in i.split('\n') if j != ''] for i in dataset]
    
    if not pre:
        sentences = []
        for sent in dataset:
            sentences += sent
        
        print('Generating word2vec model')
        word2vec_model = word2vec.Word2Vec(sentences=sentences, size=250, window=5, min_count=5, workers=4, iter=100, sg=1)
        word2vec_model.save(model_name)
        print('Finished word2vec model')
    else:
        word2vec_model = word2vec.Word2Vec.load(model_name)
        
    dataset = np.array(dataset)
    
    return dataset, word2vec_model

def text_to_index(corpus,
                  word2idx,
                  min_sent_len=2,
                  max_sent_len=15,
                  max_unk=2,
                  pad_len=15
                  ):
    
    """
    Converts text to index.
    input: parsed corpus, word to index matrix
    output: (available_dialogue, sentences, word index)
    """
    
    new_corpus = []
    for doc in corpus:
        new_doc = []
        for sent in doc:
            new_sent = []
            sent_len = 0
            unk = 0
            for word in sent:
                sent_len += 1
                try:
                    new_sent.append(word2idx[word])
                except:
                    new_sent.append(0)
                    unk += 1
            if unk <= max_unk and sent_len >= min_sent_len and sent_len <= max_sent_len:
                new_doc.append(np.array(new_sent + [0] * (pad_len - len(new_sent))))
            else:
                new_doc.append(np.zeros(15,))
        new_corpus.append(np.array(new_doc))
    return np.array(new_corpus)

def valid_dialogue(idx_corpus,
                   pad_len=15
                   ):
    
    """
    Generate valid training data.
    input: indexed corpus
    output: train_x, train_y
    """
    
    train_x = []
    train_y = []
    cut_tag = np.zeros(15,)
    for available_dialogue in idx_corpus:
        for sent in range(len(available_dialogue)-1):
            if (available_dialogue[sent] == cut_tag).all() or (available_dialogue[sent+1] == cut_tag).all():
                pass
            else:
                train_x.append(available_dialogue[sent])
                train_y.append(available_dialogue[sent+1])
    return np.array(train_x), np.array(train_y)

def embedding_idx(corpus,
                  embedding_matrix
                  ):
    
    """
    Converts indexes to vector of size 250 for model input.
    input: corpus(train_x, train_y or so), embedding matrix
    output: vectors of size 250
    """
    
    new_corpus = []
    for sent in corpus:
        new_sent = []
        for word in sent:
            try:
                new_sent.append(embedding_matrix[int(word)])
            except:
                new_sent.append(np.zeros(15,))
        new_corpus.append(new_sent)
    return np.array(new_corpus)

def recover(corpus,
            embedding_matrix_normalized,
            idx2word
            ):
    
    """
    Recovers vectors of dim=250 to words by cosine similarity.
    input: model output, NORMALIZED embedding matrix, index to word dictionary
    output: sentences(list of words)
    """
    
    new_corpus = []
    for sent in corpus:
        new_sent = []      
        for word in sent:
            if np.linalg.norm(word) != 0:
                new_word = np.argmax(np.dot(embedding_matrix_normalized,
                                            word / np.linalg.norm(word)), axis=0)
                try:
                    new_sent.append(idx2word[new_word])
                except:
                    pass
        new_corpus.append(new_sent)      
    return new_corpus

def main():
    
    dataset, w2v_model = word2vec_model(pre=True)#pre=True
    
    ###EMBEDDING###
    embedding_matrix = np.zeros((len(w2v_model.wv.vocab.items()) + 1, w2v_model.vector_size))
    word2idx = {}

    vocab_list = [(word, w2v_model.wv[word]) for word, _ in w2v_model.wv.vocab.items()]
    for i, vocab in enumerate(vocab_list):
        word, vec = vocab
        embedding_matrix[i + 1] = vec
        word2idx[word] = i + 1
        
    idx2word = {word2idx[i]:i for i in word2idx.keys()}
    embedding_matrix_normalized = normalize(embedding_matrix, axis=1)
    
    dataset = text_to_index(dataset, word2idx)#dataset:(available dialogue, sentences, word_index)
    
    train_x, train_y = valid_dialogue(dataset)
    
    """
    #will cause OOM 
    train_x = embedding_idx(train_x, embedding_matrix=embedding_matrix)
    train_y = embedding_idx(train_y, embedding_matrix=embedding_matrix)
    """
    
    return dataset, embedding_matrix, embedding_matrix_normalized, train_x, train_y, vocab_list, word2idx, idx2word

#dtst, emb, emb_n, tx, ty, vblist, w2i, i2w = main()