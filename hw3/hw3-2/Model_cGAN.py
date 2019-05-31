# -*- coding: utf-8 -*-
"""
Created on Thu May 30 19:43:01 2019

@author: Austin Hsu
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

"""
ConvTranspose2d:
    output_height:
        (input_height - 1) * stride - 2*padding + kernel_size + output_padding
"""

ADAMPARAM_G = {'lr':0.0002, 'betas':(0.5, 0.999), 'eps':1e-5}
ADAMPARAM_D = {'lr':0.0001, 'betas':(0.5, 0.999), 'eps':1e-5}
SGDPARAM  = {'lr':0.0002, 'momentum':0.9}
BATCHSIZE = 512
WGANCLIP  = 0.01

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.txt_emb = nn.Sequential(nn.Linear(130,256),
                                     nn.BatchNorm1d(256, momentum=0.9),
                                     nn.LeakyReLU(),
                                     )
        self.fc = nn.Sequential(nn.Linear(356,4*4*512),
                                nn.BatchNorm1d(4*4*512, momentum=0.9),
                                nn.LeakyReLU(),
                                )
        self.conv_layers = nn.Sequential(nn.BatchNorm2d(512, momentum=0.9),
                                         nn.LeakyReLU(),
                                         #nn.Dropout(0.2),
                                         nn.ConvTranspose2d(512, 
                                                            256, 
                                                            kernel_size=4, 
                                                            stride=2, 
                                                            padding=1,
                                                            ), #128 * 32 * 32
                                                
                                        nn.BatchNorm2d(256, momentum=0.9),
                                        nn.LeakyReLU(),
                                        #nn.Dropout(0.2),
                                        nn.ConvTranspose2d(256,
                                                           128,
                                                           kernel_size=4,
                                                           stride=2,
                                                           padding=1,
                                                           ), #64 * 64 * 64
                                        nn.BatchNorm2d(128, momentum=0.9),
                                        nn.LeakyReLU(),
                                        #nn.Dropout(0.1),
                                        nn.ConvTranspose2d(128,
                                                           64,
                                                           kernel_size=4,
                                                           stride=2,
                                                           padding=1,
                                                           ), #3 * 64 * 64               
                                        nn.BatchNorm2d(64, momentum=0.9),
                                        nn.LeakyReLU(),
                                        #nn.Dropout(0.1),
                                        nn.ConvTranspose2d(64,
                                                           3,
                                                           kernel_size=4,
                                                           stride=2,
                                                           padding=1,
                                                           ), #3 * 64 * 64               
                                        nn.BatchNorm2d(3, momentum=0.9),
                                        #nn.LeakyReLU(),
                                        nn.Tanh(),                                        
                                        )
        
    def forward(self, x, label):
        y = self.txt_emb(label)
        x = torch.cat([x,y],dim=1)
        x = self.fc(x)
        x = x.view(-1,512,4,4)
        x = self.conv_layers(x)
        return x
    
"""
Conv2d:
    output_height:
        (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
"""
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.txt_emb = nn.Sequential(nn.Linear(130,256),
                                     )  
        self.conv_layers1 = nn.Sequential(nn.Conv2d(3,
                                                   32,
                                                   kernel_size=5,
                                                   stride=2,
                                                   padding=0,
                                                   ), #32 * 30 * 30
                                        nn.BatchNorm2d(32, momentum=0.9),
                                        nn.LeakyReLU(),
                                        nn.Dropout(0.8),
                                        nn.Conv2d(32,
                                                  64,
                                                  stride=2,
                                                  kernel_size=5,
                                                  padding=1,
                                                  ), #64 * 14 * 14
                                        nn.BatchNorm2d(64, momentum=0.9),
                                        nn.LeakyReLU(),
                                        nn.Dropout(0.8),
                                        nn.Conv2d(64,
                                                  128,
                                                  stride=2,
                                                  kernel_size=5,
                                                  padding=1,
                                                  ), #128 * 6 * 6
                                        nn.BatchNorm2d(128, momentum=0.9),
                                        nn.LeakyReLU(),
                                        nn.Dropout(0.8),
                                        nn.Conv2d(128,
                                                  256,
                                                  stride=1,
                                                  kernel_size=5,
                                                  padding=1,
                                                  ), #256 * 4 * 4
                                        nn.BatchNorm2d(256, momentum=0.9),
                                        nn.LeakyReLU(),
                                        nn.Dropout(0.8),
                                        
                                        )
        self.to_out = nn.Sequential(
                                    nn.Linear(512*4*4, 1),
                                    nn.Sigmoid()
                                    )
        self.conv_layers2 = nn.Sequential(nn.Conv2d(512,
                                                  512,
                                                  kernel_size=1,
                                                  stride=(1,1),
                                                  ),
                                         #nn.BatchNorm2d(512, momentum=0.9),
                                         nn.LeakyReLU(),
                                         #nn.Dropout(0.7),
                                         )
    def forward(self, x, label):
        y = self.txt_emb(label)
        y = y.view(-1,256,1,1)
        y = y.repeat([1,1,4,4])
        x = self.conv_layers1(x)
        x = torch.cat([x,y],dim=1)
        x = self.conv_layers2(x)
        x = x.view(-1,512*4*4)
        x = self.to_out(x)
        return x
    
def criterion_d(generated, data, wrong_data):
    
    """
    (batch, channel, height, weight)
    """
    
    return (torch.mean(torch.log(data)) + torch.mean(torch.log(1-generated)) + torch.mean(torch.log(1-wrong_data)))*-1
    
def criterion_g(generated):
    
    """
    (batch, channel, height, weight)
    """
    
    return torch.mean(torch.log(generated)) * -1
    #return torch.mean((1 - generated) ** 2) * 0.5
    
def main(args):
    
    """
    //---------------------------------
    Data loading and data preprocessing
    ---------------------------------//
    """
    transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((64,64)),
             transforms.ToTensor(),
             ])
    print("Loading data...")
    data=np.load(args.data)
    data=np.moveaxis(data,3,1)
    data=torch.Tensor(data)
    label=torch.Tensor(np.load(args.label))
    dataset = Data.TensorDataset(data,label)
    train_dataloader = Data.DataLoader(dataset, batch_size=BATCHSIZE)
    total_batch=data.shape[0]//BATCHSIZE
    print("Loading complete.")
    
    """
    //------
    Training
    ------//
    """
    
    train_generator = Generator().cuda()
    train_discriminator = Discriminator().cuda()
    """
    from torchsummary import summary
    summary(train_discriminator,[(3,128,128),(130,)])
    _=input("debug")
    """
    optimizer_g = torch.optim.Adam(train_generator.parameters(), **ADAMPARAM_G)
    optimizer_d = torch.optim.Adam(train_discriminator.parameters(), **ADAMPARAM_D)
    
    loss_func_g = criterion_g
    loss_func_d = criterion_d
    
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
        
        
        for b_num, (b_x, b_y) in enumerate(train_dataloader):
            
            b_x_new = []
            for i in range(len(b_x)):
                b_x_new.append((transform(b_x[i]) - 127.5) / 127.5)                
            b_x = torch.stack(b_x_new)

            """
            Train D
            """
            
            """
            sample_tag = torch.from_numpy(np.random.choice(BATCHSIZE, BATCHSIZE//10, replace=False))
            data_d     = torch.index_select(b_x, 1, sample_tag).cuda()
            """
            for generating_train in range(args.k):
                # Data prepare
                random_picker = torch.randperm(b_x.shape[0])
                true_tag = (b_y == b_y[random_picker]).all(1)
                false_label = torch.Tensor([random_picker[i] for i in range(random_picker.shape[0]) if true_tag[i] == 0]).long()
                data_d  = b_x.cuda()
                data_wrong = b_x[false_label].cuda()
                data_label = b_y.cuda()
                data_wrong_label = b_y[false_label].cuda()
                sample_noise = noise_distribution.sample((b_x.shape[0], 100)).squeeze(2).cuda()
                
                # Loss calculation
                optimizer_d.zero_grad()
                generated = train_generator(sample_noise,data_label)
                generated = train_discriminator(generated,data_label)
                data_d    = train_discriminator(data_d,data_label)
                data_wrong= train_discriminator(data_wrong, data_wrong_label)
                dloss = loss_func_d(generated=generated, data=data_d, wrong_data=data_wrong)
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
            sample_noise = noise_distribution.sample((b_x.shape[0], 100)).squeeze(2).cuda()
            optimizer_g.zero_grad()
            generated = train_generator(sample_noise,data_label)
            generated = train_discriminator(generated,data_label)
            #print(5)
            gloss = loss_func_g(generated=generated)
            gloss.backward()
            epoch_gloss += gloss.item()
            optimizer_g.step()
            #print(6)
            ##################################################################################################################
            
            """
            Save Model
            """
            
            train_iteration = int(b_num/total_batch*20)
            train_toy = '[' + '='*train_iteration + '>' + '-'*(19-train_iteration) + '] '
            if train_iteration == 20:train_toy = '['+'='*20+']'
            print(train_toy + 'batch: ', b_num, '/', total_batch, ' Discriminator Loss: ', (epoch_dloss-old_dloss)/args.k, ' Generator Loss: ', epoch_gloss-old_gloss, end='\r')
            old_dloss, old_gloss = epoch_dloss, epoch_gloss
        torch.save(train_generator, args.model_directory + args.model_name + '_epoch_' + str(e+1) + '_generator.pkl')
        torch.save(optimizer_g, args.model_directory + args.model_name + '_epoch_' + str(e+1) + '_generator.optim')
        torch.save(train_discriminator, args.model_directory + args.model_name + '_epoch_' + str(e+1) + '_discriminator.pkl')
        torch.save(optimizer_d, args.model_directory + args.model_name + '_epoch_' + str(e+1) + '_discriminator.optim')
        
        dloss_record.append(epoch_dloss/args.k)
        gloss_record.append(epoch_gloss)
        print()
        print("Discriminator Loss: ", epoch_dloss/args.k)
        print("Generator Loss: ", epoch_gloss)
    
    dloss_record = np.array(dloss_record)
    gloss_record = np.array(gloss_record)
    np.save("./loss_record/" + args.model_name + "dloss", dloss_record)
    np.save("./loss_record/" + args.model_name + "gloss", gloss_record)
    print('Training finished.')
    
    return
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='../../../MLDS_dataset/hw3-2/large_image.npy')#AnimeDataset/
    parser.add_argument('--label', '-l', type=str, default='../../../MLDS_dataset/hw3-2/tag.npy')
    parser.add_argument('--model_name', '-mn', type=str, default='cGAN_3-2')
    parser.add_argument('--model_directory', '-md', type=str, default='../../../MLDS_models/hw3-2/')
    parser.add_argument('--epoch', '-e', type=int, default=50)
    parser.add_argument('--k', '-k', type=int, default=2)
    args = parser.parse_args()
    main(args)         
