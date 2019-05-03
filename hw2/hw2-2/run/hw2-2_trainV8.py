# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:22:07 2019
@author: Austin Hsu
"""
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
from gensim.models import word2vec
from tqdm import tqdm
from sklearn.preprocessing import normalize
from transformer_tutorial import make_model, subsequent_mask, Generator
from torch.autograd import Variable

import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
import torch.functional as F
import argparse
import os


###DEVICE###
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###REPRODUCIBLE###
torch.manual_seed(1)

###HYPERPARAMETER###
EPOCH      = 5
BATCHSIZE  = 64
ADAMPARAM  = {'lr':0.001, 'betas':(0.9, 0.98), 'eps':1e-09}#, 'weight_decay':1e-05}

def load_dataset(word2idx,
                 directory='../../../../MLDS_dataset/hw2-2/',
                 pad_len=20,
                 min_len=2,
                 ):
    
    dataset = open(os.path.join(directory,'clr_conversation.txt'), 'r', encoding='UTF-8').read().split('+++$+++')
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

def word2vec_model(directory='../../../../MLDS_dataset/hw2-2/clr_conversation.txt',
                   model_name='../word2vec_only_train.model',
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
    
    """
    dataset = open(directory, 'r', encoding='UTF-8').read().split('+++$+++')
    dataset = [[j.split(' ') for j in i.split('\n') if j != ''] for i in dataset]
    """
    
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
        
    #dataset = np.array(dataset)
    
    return word2vec_model #, dataset

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

def lrate_refresh(lr,
                  step_num,
                  warmup_steps=4000,
                  d_model=512,
                  ):
    
    return d_model ** (-0.5) * min(step_num ** (-0.5), step_num * warmup_steps ** (-1.5))

def criterion(pred, target, smooth=0.1, vocab=71475, pad_len=20, batch_size=64):
    
    smooth_target = torch.zeros((pad_len*batch_size, vocab))
    for i, j in enumerate(target):
        smooth_target[i, int(j)] = 1.0 - smooth
    smooth_target += torch.ones((pad_len*batch_size, vocab)) / vocab
    loss = nn.KLDivLoss(reduce = True, size_average=False)
    
    return loss(pred, smooth_target.cuda())

def mask_unpred(target, ite):
    alt_target = target.contiguous()
    alt_target[:, ite+1:] = 0.0
    return alt_target

def mask_back(pred, ite):
    alt_pred = pred.contiguous()
    pad = torch.zeros((71574,))
    pad[0] = 1
    pad.cuda().float()
    alt_pred[:, :, ite+1:] = pad[0]
    return alt_pred

def main(args):
    
    w2v_model = word2vec_model(directory=args.data_directory, pre=True)#pre=True
    
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
    """
    idx2word = {word2idx[i]:i for i in word2idx.keys()}
    
    embedding_matrix_normalized = normalize(embedding_matrix, axis=1)
    """
    Transformer_model = make_model(src_vocab = 71475,
                                   tgt_vocab = 71475,
                                   ).cuda()
    
    train_x, train_y = load_dataset(directory=args.data_directory, word2idx=word2idx)
    """
    quicker train
   
    train_x, train_y = train_x[:train_x.shape[0]], train_y[:train_x.shape[0]]
    """
    tensor_x = torch.stack([torch.from_numpy(np.array(i)) for i in train_x])
    tensor_y = torch.stack([torch.from_numpy(np.array(i)) for i in train_y])
    
    train_dataset = Data.TensorDataset(tensor_x,tensor_y) # create your datset
    train_dataloader = Data.DataLoader(train_dataset,batch_size=BATCHSIZE, shuffle=True) # create your dataloader
    
    print('dataloader complete')
    
    ###OPTIMIZER###
    optimizer = torch.optim.Adam(Transformer_model.parameters(), **ADAMPARAM)
    #optimizer = torch.optim.Adam(Transformer_model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    lambda1 = lambda steps: 512 ** (-0.5) * min(steps ** (-0.5), steps * 4000 ** (-1.5)) if steps != 0 else 0.001
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda1)
    
    ###LOSS FUNCTION###
    #loss_func = nn.CrossEntropyLoss()
    loss_func = criterion
    
    print("Training starts...")
    
    #history_best_epoch_loss = 1000000.0
    #loss_list = []
    steps = 0
    
    for e in range(EPOCH):
        print("Epoch ", e+1)
        epoch_loss = 0
        
        for b_num, (b_x, b_y) in enumerate(tqdm(train_dataloader)):
            steps += 1
            mask = subsequent_mask(20).cuda()
            b_x = b_x.cuda()
            b_y = b_y.cuda()
            
            for ite in range(20):
                optimizer.zero_grad()
                pred = Transformer_model(b_x, mask_unpred(b_y, ite), mask, mask)
                pred = Transformer_model.generator(pred)
                loss = loss_func(pred.contiguous().view(-1, pred.size(-1)),  b_y.contiguous().view(-1))
                loss.backward(retain_graph=True)
                optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            #print('Step ', b_num, ', loss :', loss.item())
            if(b_num % 100 == 99 ):
                bn_loss = epoch_loss / 100
                epoch_loss = 0
            if steps % 10 == 0:
                torch.save(Transformer_model, args.model_directory+'/checkpoint/' +'epoch_'+str(e+1) + '_checkpoint_' + str(steps) + '_' + args.model_name + '.pkl')
                torch.save(optimizer.state_dict(), args.model_directory+'/checkpoint/' +'epoch_'+str(e+1) + '_checkpoint_' + str(steps) + '_' + args.model_name +'_model.optim')
        print(bn_loss)
    #np.save('./loss_record/'+'model_loss', np.array(loss_list))
    print("Training finished.")
    
    
    #return embedding_matrix, embedding_matrix_normalized, train_x, train_y, word2idx, idx2word

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_directory', '-dd', type=str, default='../../../../MLDS_dataset/hw2-2/')
    parser.add_argument('--model_name', '-mn', type=str, default='Transformer')
    parser.add_argument('--model_directory', '-md', type=str, default='../../../../MLDS_models/hw2-2/')
    args = parser.parse_args()
    main(args)