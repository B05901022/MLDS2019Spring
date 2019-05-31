# -*- coding: utf-8 -*-
"""
Created on Thu May 30 19:55:48 2019

@author: u8815
"""

import os
import cv2
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
from tqdm import tqdm
import random
import sys
import argparse
ADAMPARAM = {'lr':0.0002, 'betas':(0.5, 0.999), 'eps':1e-5}
ADAMPARAM2= {'lr':0.0002, 'betas':(0.5, 0.999), 'eps':1e-5}
SGDPARAM  = {'lr':0.0002, 'momentum':0.9}
BATCHSIZE = 1024
WGANCLIP  = 0.01
#WGANCLIP  = 0.01

"""
Generator 

input:Noise(batchsize,100)
      Label(batchsize,130)
output:x_fake(batchsize,3,64,64)
"""
class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()   
        self.convt3=nn.Sequential(nn.BatchNorm2d(64,momentum=0.9),
                                  nn.ConvTranspose2d(64,3,4,2,1),
                                  nn.LeakyReLU()
                                  )
        self.convt64=nn.Sequential(nn.BatchNorm2d(128,momentum=0.9),
                                  nn.ConvTranspose2d(128,64,4,2,1),
                                  nn.LeakyReLU()
                                  )
        self.convt128=nn.Sequential(nn.BatchNorm2d(256,momentum=0.9),
                                  nn.ConvTranspose2d(256,128,4,2,1),
                                  nn.LeakyReLU()
                                  )
        self.convt256=nn.Sequential(nn.BatchNorm2d(512,momentum=0.9),
                                  nn.ConvTranspose2d(512,256,4,2,1),
                                  nn.LeakyReLU()
                                  )
        self.txt_emb=nn.Linear(130,256)
        self.noise_emb=nn.Linear(356,512*4*4)
        self.dropout=nn.Dropout(0.5)
        self.activation=nn.Tanh()
    def forward(self,noise,label):
        label=self.txt_emb(label)            #label:[-1,256]
        noise=torch.cat([label,noise],dim=1) #noise:[-1,356]
        noise=self.noise_emb(noise)          #noise:[-1,8192]
        noise=noise.view(-1,512,4,4)         #noise:[-1,512,4,4]
        noise=self.convt256(noise)           #noise:[-1,256,8,8]
        noise=self.convt128(noise)           #noise:[-1,128,16,16]
        noise=self.convt64(noise)            #noise:[-1,64,32,32]
        noise=self.convt3(noise)             #noise:[-1,3,64,64]
        x_fake=self.activation(noise)        #x_fake:[-1,3,64,64]
        return x_fake

"""
Discriminator

input:x(batchsize,100)
      Label(batchsize,130)
output:x_fake(batchsize,3,64,64)
"""
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        
        self.conv3=nn.Sequential(
                                  nn.Conv2d(3,32,5,2,0),
                                  nn.ReLU()
                                  )
        self.conv32=nn.Sequential(
                                  nn.Conv2d(32,64,5,2,1),
                                  nn.BatchNorm2d(64,momentum=0.9),
                                  nn.ReLU()
                                  )
        self.conv64=nn.Sequential(
                                  nn.Conv2d(64,128,5,2,1),
                                  nn.BatchNorm2d(128,momentum=0.9),
                                  nn.ReLU()
                                  )
        self.conv128=nn.Sequential(
                                  nn.Conv2d(128,256,5,1,1),
                                  nn.BatchNorm2d(256,momentum=0.9),
                                  nn.ReLU()
                                  )
        self.conv512=nn.Sequential(
                                  nn.Conv2d(512,512,1,(1,1)),
                                  nn.BatchNorm2d(512,momentum=0.9),
                                  nn.ReLU()
                                  )
        self.txt_emb=nn.Linear(130,256)
        self.score_out=nn.Sequential(nn.Linear(512*4*4,1),
                                     nn.Sigmoid())
        self.category_out=nn.Sequential(nn.Linear(512*4*4,130),
                                     nn.Softmax())
        self.dropout=nn.Dropout(0.8)
        self.activation=nn.Tanh()
    def forward(self,x,label):       
        label=self.txt_emb(label)            #label:[-1,256]
        label=label.view([-1,256,1,1])       #label:[-1,256,1,1]
        label=label.repeat([1,1,4,4])        #label:[-1,256,4,4]
        x=self.conv3(x)                     #x:[-1,32,64,64]
        x=self.dropout(x)
        x=self.conv32(x)                     #x:[-1,64,32,32]
        x=self.dropout(x)
        x=self.conv64(x)                    #x:[-1,128,16,16]
        x=self.dropout(x)
        x=self.conv128(x)                   #x:[-1,256,8,8]
        x=self.dropout(x)
        x=torch.cat([x,label],dim=1) 
        x=self.conv512(x)                   #x:[-1,512,4,4]
        x=self.dropout(x)        #x:[-1,768,4,4])
        x=x.view(-1,8192)                  #x:[-1,8192]
        score=self.score_out(x)              #score:[-1,1]
        category=self.category_out(x)        #category:[-1,130]
        return score,category
"""
Discriminator

input:x(batchsize,100)
      Label(batchsize,130)
output:x_fake(batchsize,3,64,64)
"""


"""
criterion_LS=nn.BCELoss().cuda()
"""
def criterion_LS(predicted_real,predicted_fake):
    return -(torch.mean(torch.log(predicted_real)) + torch.mean(torch.log(1-predicted_fake)))
def criterion_LC(cato_real,label_real,cato_fake,label_fake):
    sr=0
    sf=0
    for i in range(cato_real.shape[0]):
         sr=sr+cato_real[i][label_real[i]]
         sf=sf+cato_fake[i][label_fake[i]]
    return -(torch.mean(torch.log(sr/cato_real.shape[0])) + torch.mean(torch.log(sf/cato_real.shape[0])))
#criterion_LC=nn.CrossEntropyLoss().cuda()
def main(args):
    transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((64,64)),
             transforms.ToTensor(),
             ])
    data=np.load(args.data)
    data=np.moveaxis(data,3,1)
    data=torch.Tensor(data)
    label=torch.Tensor(np.load(args.label))
    dataset = Data.TensorDataset(data,label)
    train_dataloader = Data.DataLoader(dataset, batch_size=BATCHSIZE)
    total_batch=data.shape[0]//BATCHSIZE
    
    """
    //------
    Training
    ------//
    """
    
    train_generator = Generator().cuda()
    train_discriminator = Discriminator().cuda()
    
    #loss_func_g = 
    #loss_func_d = 
    
    noise_distribution = torch.distributions.Normal(torch.Tensor([0.0]), torch.Tensor([1.0]))
    dloss_record = []
    gloss_record = []
    
    print('Training starts...')
    
    for e in range(args.epoch):
        print('Epoch ', e+1)
        epoch_dloss = 0
        epoch_gloss = 0
        old_dloss = 0
        old_gloss = 0
        optimizer_g = torch.optim.Adam(train_generator.parameters(), **ADAMPARAM)
        optimizer_d = torch.optim.Adam(train_discriminator.parameters(), **ADAMPARAM2)
        #real_truth=torch.FloatTensor(np.ones((BATCHSIZE,1))).cuda().squeeze(1)
        #fake_truth=torch.FloatTensor(np.zeros((BATCHSIZE,1))).cuda().squeeze(1)
        for b_num, (b_x, b_y) in enumerate(train_dataloader):    

            """
            Train D
            """
            
            """
            sample_tag = torch.from_numpy(np.random.choice(BATCHSIZE, BATCHSIZE//10, replace=False))
            data_d     = torch.index_select(b_x, 1, sample_tag).cuda()
            """
            torch.set_printoptions(threshold=sys.maxsize)
            b_x_new = []
            for i in range(len(b_x)):
                b_x_new.append(transform(b_x[i]) *2 - 1)              
            b_x = torch.stack(b_x_new)
            #if b_x.shape[0]!=BATCHSIZE:
                #real_truth=torch.Tensor((np.ones((b_x.shape[0],1)))).squeeze(1).cuda()
                #fake_truth=torch.Tensor((np.ones((b_x.shape[0],1)))).squeeze(1).cuda()
                #shape,fake_truth.shape)
            for generating_train in range(args.k):
                # Data prepare
                #random_picker = torch.randperm(BATCHSIZE)
                data_d  = b_x.cuda()
                #data_wrong = b_x[random_picker].cuda()
                data_label = b_y.cuda()
                sample_noise = noise_distribution.sample((b_x.shape[0], 100)).squeeze(2).cuda()
                gen_label=torch.FloatTensor((np.zeros((b_x.shape[0],130)))).cuda()
                for i in range(gen_label.shape[0]):
                    gen_label[i][np.random.randint(0,129)]=1
                # Loss calculation

                optimizer_d.zero_grad()
                x_fake = train_generator(sample_noise,gen_label)
                data_fake,label_fake = train_discriminator(x_fake,gen_label)
                data_real,label_real = train_discriminator(data_d,data_label)
                label_real=label_real.to(torch.float).squeeze(1)
                label_fake=label_fake.to(torch.float).squeeze(1)
                data_label=torch.max(data_label,1)[1]
                gen_label=torch.max(gen_label,1)[1]
                #print (gen_label.shape,data_label.shape,label_real.shape,label_fake.shape)
                #data_wrong= train_discriminator(data_wrong, data_label)
                #dloss = loss_func_d(generated=generated, data=data_d, wrong_data=data_wrong)
                loss_Ls=criterion_LS(data_real,data_fake)
                loss_Lc=criterion_LC(label_real,data_label,label_fake,gen_label)
                dloss=loss_Lc+loss_Ls
                dloss.backward()
                epoch_dloss += dloss.item()
                optimizer_d.step()
            """
            for param in train_discriminator.parameters():
                param = torch.clamp(param, -1* WGANCLIP, WGANCLIP)
            """
            ##################################################################################################################
            
            """
            Train G
            """
            data_label = b_y.cuda()
            sample_noise = noise_distribution.sample((b_x.shape[0], 100)).squeeze(2).cuda()
            optimizer_g.zero_grad()
            gen_label=torch.FloatTensor((np.zeros((b_x.shape[0],130)))).cuda()
            for i in range(gen_label.shape[0]):
                gen_label[i][np.random.randint(0,129)]=1
            # Loss calculation
            optimizer_d.zero_grad()
            x_fake = train_generator(sample_noise,gen_label)
            data_fake,label_fake = train_discriminator(x_fake,gen_label)
            data_real,label_real = train_discriminator(data_d,data_label)
            label_real=label_real.to(torch.float)
            label_fake=label_fake.to(torch.float)
            data_label=torch.max(data_label,1)[1]
            gen_label=torch.max(gen_label,1)[1]
            loss_Ls=criterion_LS(data_real,data_fake)
            loss_Lc=criterion_LC(label_real,data_label,label_fake,gen_label)
            gloss=loss_Lc-loss_Ls
            gloss.backward()
            epoch_gloss += gloss.item()
            optimizer_g.step()
            #print(6)
            ##################################################################################################################
            
            """
            Save Model
            """
            
        
            print('batch: ', b_num, '/', total_batch, ' Discriminator Loss: ', (epoch_dloss-old_dloss)/args.k, ' Generator Loss: ', epoch_gloss-old_gloss, end='\r')
            old_dloss, old_gloss = epoch_dloss, epoch_gloss
        torch.save(train_generator, args.model_directory + args.model_name + '_epoch_' + str(e+1) + '_generator.pkl')
        torch.save(optimizer_g, args.model_directory + args.model_name + '_epoch_' + str(e+1) + '_generator.optim')
        torch.save(train_discriminator, args.model_directory + args.model_name + '_epoch_' + str(e+1) + '_discriminator.pkl')
        torch.save(optimizer_d, args.model_directory + args.model_name + '_epoch_' + str(e+1) + '_discriminator.optim')
        
        dloss_record.append(epoch_dloss)
        gloss_record.append(epoch_gloss)
        print()
        print("Discriminator Loss: ", epoch_dloss)
        print("Generator Loss: ", epoch_gloss)
    
    dloss_record = np.array(dloss_record)
    gloss_record = np.array(gloss_record)
    np.save("./loss_record/" + args.model_name + "dloss", dloss_record)
    np.save("./loss_record/" + args.model_name + "gloss", gloss_record)
    print('Training finished.')
    
    return
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='E:/large_image.npy')#AnimeDataset/
    parser.add_argument('--label', '-l', type=str, default='E:/tag.npy')
    parser.add_argument('--model_name', '-mn', type=str, default='GAN_3-2')
    parser.add_argument('--model_directory', '-md', type=str, default='D:/hw3-2/')
    parser.add_argument('--epoch', '-e', type=int, default=50)
    parser.add_argument('--k', '-k', type=int, default=1)
    args = parser.parse_args()
    main(args)         