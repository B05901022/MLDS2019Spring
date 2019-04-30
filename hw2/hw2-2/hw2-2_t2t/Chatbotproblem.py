# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 19:57:24 2019

@author: Austin Hsu
"""

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry
from Dataparse import word2vec_model, text_to_index, embedding_idx
import numpy as np

@registry.register_problem
class Chatbot(text_problems.Text2TextProblem):
    """Respond a given sentence. From film conversations"""
    
    @property
    def approx_vocab_size(self):
        return 71472
    
    @property
    def is_generate_per_split(self):
        return False
    
    @property
    def dataset_splits(self):
        """evaluate with 10% of data"""
        return [{
                "split": problem.DatasetSplit.TRAIN,
                "shards":9},
                {
                "split": problem.DatasetSplit.EVAL,
                "shards":1
                }]
    
    @property
    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir
        del dataset_split
        
        dataset, w2v_model = word2vec_model(pre=True)#pre=True
    
        ###EMBEDDING###
        embedding_matrix = np.zeros((len(w2v_model.wv.vocab.items()) + 1, w2v_model.vector_size))
        word2idx = {}
    
        vocab_list = [(word, w2v_model.wv[word]) for word, _ in w2v_model.wv.vocab.items()]
        for i, vocab in enumerate(vocab_list):
            word, vec = vocab
            embedding_matrix[i + 1] = vec
            word2idx[word] = i + 1
        
        dataset = text_to_index(dataset, word2idx)#dataset:(available dialogue, sentences, word_index)
        
        cut_tag = np.zeros(15,)
        
        for available_dialogue in dataset:
            for sent in range(len(available_dialogue)-1):
                if (available_dialogue[sent] == cut_tag).all() or (available_dialogue[sent+1] == cut_tag).all():
                    pass
                else:
                    inputs = available_dialogue[sent]
                    targets= available_dialogue[sent+1]
                    [inputs, targets] = embedding_idx([inputs, targets])
                    yield{
                        "inputs": inputs,
                        "targets":targets
                    }
                    
        
        