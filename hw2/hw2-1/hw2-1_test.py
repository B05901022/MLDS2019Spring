# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 16:59:56 2019

@author: Austin Hsu
"""

import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import argparse
import os
import yaml
from tqdm import tqdm


###MODEL###
class S2VT(nn.Module):
    def __init__(self,attention,batch_size,e_layers,e_hidden,
                 d_layers,d_hidden,one_hot_length,minibatch):
        super(S2VT,self).__init__()
        self.attention=attention
        self.batch_size=batch_size
        self.encoder_layers=e_layers
        self.encoder_hidden=e_hidden
        self.decoder_layers=d_layers
        self.decoder_hidden=d_hidden
        self.ohl=one_hot_length
        self.encoder_h=torch.zeros((e_layers,batch_size,e_hidden),dtype=torch.float32).cuda()
        self.encoder_c=torch.zeros((e_layers,batch_size,e_hidden),dtype=torch.float32).cuda()
        self.decoder_h=torch.zeros((d_layers,batch_size,d_hidden),dtype=torch.float32).cuda()
        self.decoder_c=torch.zeros((d_layers,batch_size,d_hidden),dtype=torch.float32).cuda()
        self.encoder=nn.LSTM(input_size=512,
                                hidden_size=e_hidden,
                                num_layers=e_layers,
                                bidirectional=False
                                )

        self.decoder=nn.LSTM(input_size=e_hidden+d_hidden,
                                hidden_size=d_hidden,
                                num_layers=d_layers,
                                bidirectional=False)
        self.embedding_layer_i=nn.Linear(self.ohl,d_hidden)
        self.embedding_layer_o=nn.Linear(d_hidden,self.ohl)
        self.embedding_layer_down=nn.Sequential(nn.Linear(4096,512),
                                                nn.ReLU())
        self.softmax=torch.nn.Softmax(dim=1)
        self.relu=nn.ReLU()
        self.minibatch=minibatch

    def inverse_sigmoid(self,x,k):
        return k/(k+np.exp(x/k))
    
    def forward(self,input_feature,max_len,correct_answer):
        sentence=[]
        """Encoding"""
        input_feature=self.embedding_layer_down(input_feature)
        input_feature=input_feature.view(input_feature.shape[1],input_feature.shape[0],512)
        encoded_sequence,(he,ce)=self.encoder(input_feature,(self.encoder_h,self.encoder_c))
        pad=torch.zeros((len(encoded_sequence),self.batch_size,self.decoder_hidden),dtype=torch.float32).cuda()       
        decoded_input=torch.cat((encoded_sequence,pad),dim=2)
        decoded_output,(hd,cd)=self.decoder(decoded_input,(self.decoder_h,self.decoder_c))
        """Decoding""" 
        padding=torch.zeros((max_len,self.batch_size,512),
                            dtype=torch.float32).cuda()
        encoded_padding,(he,ce)=self.encoder(padding,(he, ce))
        bos=torch.zeros((1,self.batch_size,self.ohl),
                        dtype=torch.float32).cuda()
        bos[:,:,-2]=1
        for s in range(max_len): 
            correct=None
            sample=None
            if (s==0):
                bos_embedding=self.embedding_layer_i(bos) 
                sample=bos_embedding
                correct=(encoded_padding[s]).unsqueeze(0)
                decoded_input=torch.cat((sample,correct),dim=2)
                decoded_output,(hd,cd)=self.decoder(decoded_input,(hd,cd))
                word=self.embedding_layer_o(decoded_output).squeeze(0)
                sentence.append(word)    
            else:
                a=correct_answer[s].unsqueeze(0)
                
                schedule=self.inverse_sigmoid(s, k=4)
                c=np.random.uniform()
                if c<schedule:
                    sample=self.embedding_layer_i(a)
                else: 
                    sample=decoded_output
                correct=(encoded_padding[s]).unsqueeze(0)
                decoded_input=torch.cat((sample,correct),dim=2)
                decoded_output,(hd,cd)=self.decoder(decoded_input,(hd,cd))
                word=self.embedding_layer_o(decoded_output).squeeze(0)
                sentence.append(word)
        return sentence

    
    def test(self,input_feature,max_len):
        sentence=[]
        """Encoding"""
        print (input_feature.shape,"In")
        input_feature=self.embedding_layer_down(input_feature)
        input_feature=input_feature.view(input_feature.shape[1],input_feature.shape[0],1024)
        encoded_sequence,(he,ce)=self.encoder(input_feature,(self.encoder_h,self.encoder_c))
        pad=torch.zeros((len(encoded_sequence),self.batch_size,self.decoder_hidden),dtype=torch.float32).cuda()       
        decoded_input=torch.cat((encoded_sequence,pad),dim=2)
        decoded_output,(hd,cd)=self.decoder(decoded_input,(self.decoder_h,self.decoder_c))
        """Decoding""" 
        padding=torch.zeros((max_len,self.batch_size,1024), dtype=torch.float32).cuda()
        encoded_padding,(he,ce)=self.encoder(padding,(he, ce))
        bos=torch.zeros((1,self.batch_size,self.ohl), dtype=torch.float32).cuda()
        bos[:,:,-2]=1
        for s in range(max_len): 
            correct=None
            sample=None
            if (s==0):
                bos_embedding=self.embedding_layer_i(bos) 
                sample=bos_embedding
                correct=(encoded_padding[s]).unsqueeze(0)
                decoded_input=torch.cat((sample,correct),dim=2)
                decoded_output,(hd,cd)=self.decoder(decoded_input,(hd,cd))
                word=self.embedding_layer_o(decoded_output).squeeze(0)
                sentence.append(word)    
            else:                
                sample=decoded_output
                correct=(encoded_padding[s]).unsqueeze(0)
                decoded_input=torch.cat((sample,correct),dim=2)
                decoded_output,(hd,cd)=self.decoder(decoded_input,(hd,cd))
                word=self.embedding_layer_o(decoded_output).squeeze(0)
                word=self.softmax(word)
                sentence.append(word)
        return sentence

###DEVICE###
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###REPRODUCIBLE###
torch.manual_seed(1)

###HYPERPARAMETER###
MODELPARAM = {'e_layers':1,'e_hidden':256,'d_layers':1,'d_hidden':256}

###DATA LOADING PARAMS###
LOADPARAM  = {'directory': '../../../hw2-1/MLDS_hw2_1_data', 'batch_size':2}
max_len    = 44
 
def load_test(directory, batch_size):
    """
    directory
    """
    print('Loading starts...')
    test_feat = list()
    video_id_list = open(os.path.join(directory,"testing_data/id.txt"), 'r').read().split('\n')[:-1]
    for video_id in video_id_list:
        test_feat.append(np.load(os.path.join(directory, 'testing_data/feat', video_id + '.npy')))
    print('Loading word dict...')
    word_dict = yaml.load(open('word_dict.yaml', 'r'))
    word_dict = {np.argmax(word_dict[key]):key for key in word_dict.keys()}
    print('Loading successed.')
    test_x = np.array(test_feat)
    del test_feat
    
    test_x = torch.Tensor(test_x)
    
    Dataset = Data.TensorDataset(test_x)
    DataLoader = Data.DataLoader(dataset = Dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    print('Loading finished.')
    
    return DataLoader, video_id_list, word_dict
   
def main(args):
    
    ###DATALOADER###
    test_dataloader, video_id_list, word_dict = load_test(**LOADPARAM)
    
    ###LOAD MODEL###
    test_model = torch.load('./models/'+args.model_no+'_model.pkl').cuda()
    
    print("Testing starts...")
    
    pred_list = []
    
    for b_num, (b_x) in enumerate(tqdm(test_dataloader)):
        b_x = b_x[0].cuda()
        pred = test_model.test(b_x, max_len)
        pred = torch.stack(pred)
        resu = pred.clone().cpu().detach().numpy()
        resu = np.transpose(resu, (1,0,2))
        res0 = np.argmax(resu[0], axis=1).tolist()
        res1 = np.argmax(resu[1], axis=1).tolist()
        pres = [[video_id_list[2*b_num], res0], [video_id_list[2*b_num+1], res1]]
        pred_list.append(pres[0])
        pred_list.append(pres[1])
    
    with open('pred_result.csv', "w") as f:
        print('id,value', file=f)
        for i in range(len(pred_list)):
            video_id = pred_list[i][0]
            sent = str()
            for j in pred_list[i][1]:
                if word_dict[j] not in ['<EOS>', '<BOS>', '<UNK>']:
                    sent += word_dict[j]
                    sent += ' '
                if word_dict[j] == '<EOS>':
                    sent = sent[:-1]
                    break
            sent = sent[0].upper() + sent[1:]
            print(video_id)
            print(sent)
            print('%s,%s' % (video_id, sent), file=f)
            
    print("Testing finished.")
    
    return 
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_no', '-m', type=str, default='0')
    args = parser.parse_args()
    main(args)
  