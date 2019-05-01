# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:22:07 2019
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

###DEVICE###
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###REPRODUCIBLE###
torch.manual_seed(1)

###HYPERPARAMETER###
EPOCH      = 2
BATCHSIZE  = 1
ADAMPARAM  = {'lr':0.001, 'betas':(0.9, 0.999), 'eps':1e-08}#, 'weight_decay':1e-05}

###DATA LOADING PARAMS###
LOADPARAM  = {'directory': './MLDS_hw2_1_data', 'min_count':3, 'random_seed':None, 'batch_size':4}

def load_dataset(word2idx,
                 directory='./../mlds_hw2_2_data/clr_conversation.txt',
                 pad_len=20,
                 min_len=2,
                 ):
    
    dataset = open(directory, 'r', encoding='UTF-8').read().split('+++$+++')
    dataset = [[j.split(' ') for j in i.split('\n') if j != ''] for i in dataset]
    
    print("Generating dataset...")
    train_x = []
    train_y = []
    register= []
    for valid in dataset:
        for sent in valid:
            sent_len = len(sent)
            if sent_len <= (pad_len - 2) and sent_len >= min_len:
                padded_sent = [1]
                padded_sent += sent2idx(sent, word2idx)
                padded_sent += [2]
                padded_sent += [0] * (pad_len - sent_len - 2)
                if register != []:
                    train_x.append(register)
                    train_y.append(padded_sent)
                    register = padded_sent
                else:
                    register = padded_sent
            else:
                register = []
        register = []
    print("Generating finished.")  
    
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    
    return train_x, train_y

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

def word2vec_model(directory='./../mlds_hw2_2_data/clr_conversation.txt',
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
        word2vec_model = word2vec.Word2Vec(sentences=sentences, size=250, window=5, min_count=5, workers=16, iter=100, sg=1)
        word2vec_model.save(model_name)
        print('Finished word2vec model')
    else:
        word2vec_model = word2vec.Word2Vec.load(model_name)
        
    dataset = np.array(dataset)
    
    return dataset, word2vec_model

def text_to_index(corpus,
                  word2idx,
                  min_sent_len=2,
                  max_sent_len=18,
                  max_unk=2,
                  pad_len=20
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
                    new_sent.append(3)# idx2word[3]='UNK'
                    unk += 1
            if unk <= max_unk and sent_len >= min_sent_len and sent_len <= max_sent_len:
                new_doc.append(np.array([1]+new_sent +[2]+[0] * (pad_len - len(new_sent)-2)))#Add EOS and BOS signal
            else:
                new_doc.append(np.zeros(20,))
        new_corpus.append(np.array(new_doc))
    return np.array(new_corpus)

def valid_dialogue(idx_corpus,
                   ):
    
    """
    Generate valid training data.
    input: indexed corpus
    output: train_x, train_y
    """
    
    train_x = []
    train_y = []
    cut_tag = np.zeros(20,)
    for available_dialogue in idx_corpus:
        for sent in range(len(available_dialogue)-1):
            #print(available_dialogue[sent] == cut_tag)
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
                new_sent.append(np.zeros(20,))
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
    
    _ , w2v_model = word2vec_model(pre=True)#pre=True
    
    ###EMBEDDING###
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
    idx2word = {word2idx[i]:i for i in word2idx.keys()}
    
    embedding_matrix_normalized = normalize(embedding_matrix, axis=1)
    
    """
    dataset = text_to_index(dataset, word2idx)#dataset:(available dialogue, sentences, word_index)
    
    train_x, train_y = valid_dialogue(dataset)
    """
    """
    #will cause OOM 
    train_x = embedding_idx(train_x, embedding_matrix=embedding_matrix)
    train_y = embedding_idx(train_y, embedding_matrix=embedding_matrix)
    """
    
    Transformer_model = make_model(src_vocab = 250,
                                   tgt_vocab = 250,
                                   ).cuda()
    
    train_x, train_y = load_dataset(word2idx=word2idx)
    print(train_x.shape, ',', train_y.shape)
    #train_x = embedding_idx(train_x, embedding_matrix=embedding_matrix)
    #train_y = embedding_idx(train_y, embedding_matrix=embedding_matrix)
    tensor_x = torch.stack([torch.from_numpy(np.array(i)) for i in train_x])
    tensor_y = torch.stack([torch.from_numpy(np.array(i)) for i in train_y])
    
    print(tensor_x.shape, ',', tensor_y.shape)
    
    train_dataset = Data.TensorDataset(tensor_x,tensor_y) # create your datset
    train_dataloader = Data.DataLoader(train_dataset)#batch_size=BATCHSIZE) # create your dataloader
    
    print('dataloader complete')
    
    ###OPTIMIZER###
    optimizer = torch.optim.Adam(Transformer_model.parameters(), **ADAMPARAM)
    #optimizer = torch.optim.Adam(Transformer_model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    
    ###LOSS FUNCTION###
    loss_func = nn.CrossEntropyLoss()
    
    print("Training starts...")
    
    history_best_epoch_loss = 1.0
    loss_list = []
    
    for e in range(EPOCH):
        print("Epoch ", e+1)
        epoch_loss = 0
        
        for b_num, (b_x, b_y) in enumerate(tqdm(train_dataloader)):
            #b_x = embedding_idx(b_x, embedding_matrix=embedding_matrix)
            #b_y = embedding_idx(b_y, embedding_matrix=embedding_matrix)
            #print(b_x.shape,',',b_y.shape)
            
            #b_xt = torch.from_numpy(b_x).long()
            #b_yt = torch.from_numpy(b_y).long()
            mask = subsequent_mask(20).cuda()
            #print(b_xt.shape,',',b_yt.shape)
            
            #b_xt = b_xt.cuda()
            #b_yt = b_yt.cuda()
            '''
            b_xt = b_xt.squeeze(0)
            b_yt = b_yt.squeeze(0)
            b_xt = b_xt.transpose(1,0)
            b_yt = b_yt.transpose(1,0)
            b_xt = b_xt.unsqueeze(0)
            b_yt = b_yt.unsqueeze(0)
            '''
            #b_xt = b_xt.view(b_xt.size(0), -1) 
            #b_yt = b_yt.view(b_yt.size(0), -1) 
            #print(b_xt.shape,',',b_yt.shape)
            b_x = b_x.cuda()
            b_y = b_y.cuda()
            optimizer.zero_grad()
            pred = Transformer_model(b_x, b_y, mask, mask)
            #pred=torch.stack(pred)
            #print('pred : ', pred.shape)
            #pred = pred[0,:,0]
            #pred = pred.unsqueeze(0)
            #print('_pred_ : ', pred.shape)
            #print('b_y : ', b_y.shape)
            #pred=torch.stack(pred)
            
            b_y = embedding_idx(b_y, embedding_matrix=embedding_matrix)
            print('b_y : ', b_y.shape)
            b_yt = torch.from_numpy(b_y).long()
            b_yt = b_yt.cuda()
            print('b_yt.shape : ', b_yt.shape)
            
            #b_y = b_y.view(b_y.size(0), -1)
            #pred = pred.view(pred.size(0), -1) 
            #loss = loss_func(pred , torch.max(b_y, 1)[1])
            pred = Transformer_model.generator(pred)
            print('_pred_ : ', pred.shape)
            for i in range(0,19):
                print(pred[0][i].unsqueeze(0).shape, ',', b_yt[0][i].unsqueeze(0).shape)
                loss = loss_func(pred[0][i], b_yt)
                #loss = loss_func(pred, b_yt)
                #loss = loss_func(pred,b_y)
                #loss = loss_func(pred, b_yt)
                loss.backward(retain_graph=True)
                optimizer.step()
                epoch_loss += loss.item()
            
        current_epoch_loss = epoch_loss / datasize
        loss_list.append(current_epoch_loss)
        
        history_best_epoch_loss = min(current_epoch_loss,history_best_epoch_loss)
        print("Epoch loss: ", current_epoch_loss)
        
        print("Best loss : %.8f" % history_best_epoch_loss)
    
    print("Training finished.")
    
    
    return embedding_matrix, embedding_matrix_normalized, train_x, train_y, word2idx, idx2word

main()
print('done main')