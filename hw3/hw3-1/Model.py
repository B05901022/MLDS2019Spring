# -*- coding: utf-8 -*-
"""
Created on Sun May  5 22:00:31 2019

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

"""
ConvTranspose2d:
    output_height:
        (input_height - 1) * stride - 2*padding + kernel_size + output_padding
"""

ADAMPARAM = {'lr':0.0002, 'betas':(0.5, 0.999), 'eps':1e-5}
BATCHSIZE = 512

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.to_2d = nn.Linear(100, 128*16*16)
        self.conv_layers = nn.Sequential(nn.ConvTranspose2d(128, 
                                                            128, 
                                                            kernel_size=4, 
                                                            stride=2, 
                                                            padding=1,
                                                            ), #128 * 32 * 32
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(128,
                                                           64,
                                                           kernel_size=4,
                                                           stride=2,
                                                           padding=1,
                                                           ), #64 * 64 * 64
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(64,
                                                           3,
                                                           kernel_size=3,
                                                           stride=1,
                                                           padding=1,
                                                           ), #64 * 3 * 3
                                        nn.Tanh(),
                                        )
    def forward(self, x):
        x = self.to_2d(x)
        x = x.view(-1, 128, 16, 16)
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
        self.conv_layers = nn.Sequential(nn.Conv2d(3,
                                                   32,
                                                   kernel_size=4,
                                                   padding=0,
                                                   ), #32 * 61 * 61
                                        nn.ReLU(),
                                        nn.Conv2d(32,
                                                  64,
                                                  kernel_size=4,
                                                  padding=1,
                                                  ), #64 * 60 * 60
                                        nn.ReLU(),
                                        nn.Conv2d(64,
                                                  128,
                                                  kernel_size=4,
                                                  ), #128 * 57 * 57
                                        nn.ReLU(),
                                        nn.Conv2d(128,
                                                  256,
                                                  kernel_size=4,
                                                  ), #256 * 54 * 54
                                        nn.ReLU(),
                                        )
        self.to_out = nn.Sequential(nn.Linear(256*54*54, 1),
                                    nn.Sigmoid(),
                                    )
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1)
        x = self.to_out(x)
        return x

def criterion_d(generated, data, samplesize):
    
    """
    (batch, channel, height, weight)
    """
    
    return (torch.sum(generated) - torch.sum(data)) / samplesize
    
def criterion_g(generated, samplesize):
    
    """
    (batch, channel, height, weight)
    """
    
    return torch.sum(generated) / samplesize * -1
    
def main(args):
    
    """
    //---------------------------------
    Data loading and data preprocessing
    ---------------------------------//
    """
    transform = transforms.Compose(
            [transforms.Scale([64,64]),
             transforms.ToTensor(),
             transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    traindata = torchvision.datasets.ImageFolder(root=args.data_directory, transform=transform)
    train_dataloader = Data.dataloader(traindata, batchsize=BATCHSIZE)
    
    """
    //------
    Training
    ------//
    """
    
    train_generator = Generator().cuda()
    train_discriminator = Discriminator().cuda()
    
    optimizer_g = torch.optim.Adam(train_generator.parameters(), **ADAMPARAM)
    optimizer_d = torch.optim.Adam(train_discriminator.parameters(), **ADAMPARAM)
    
    loss_func_g = criterion_g
    loss_func_d = criterion_d
    
    noise_distribution = torch.distributions.Normal(torch.Tensor([0.0]), torch.Tensor([1.0]))
    
    print('Training starts...')
    
    for e in range(args.epoch):
        print('Epoch ', e+1)
        #epoch_loss = 0
        
        for b_num, (b_x, b_y) in enumerate(tqdm(train_dataloader)):
            
            """
            Train D
            """
            
            sample_tag = torch.from_numpy(np.random.choice(BATCHSIZE, BATCHSIZE//10, replace=False))
            data_d     = b_x.index_select(1, sample_tag).cuda()
            sample_noise = noise_distribution.sample(BATCHSIZE//10, 100).squeeze(2).cuda()
            optimizer_d.zero_grad()
            generated = train_discriminator(train_generator(sample_noise))
            data_d    = train_discriminator(data_d)
            dloss = loss_func_d(generated, data_d, BATCHSIZE//10)
            dloss.backward()
            optimizer_d.step()
            
            ##################################################################################################################
            
            """
            Train G
            """
            
            for generating_train in range(args.k):
                sample_noise = noise_distribution.sample(BATCHSIZE//10, 100).squeeze(2).cuda()
                generated = train_discriminator(train_generator(sample_noise))
                optimizer_g.zero_grad()
                gloss = loss_func_g(generated, BATCHSIZE//10)
                gloss.backward()
                optimizer_g.step()
            
            ##################################################################################################################
            
            """
            Save Model
            """
            
            torch.save(train_generator, args.model_directory + args.model_name + '_epoch_' + e+1 + '_generator.pkl')
            torch.save(optimizer_g, args.model_directory + args.model_name + '_epoch_' + e+1 + '_generator.optim')
            torch.save(train_discriminator, args.model_directory + args.model_name + '_epoch_' + e+1 + '_discriminator.pkl')
            torch.save(optimizer_d, args.model_directory + args.model_name + '_epoch_' + e+1 + '_discriminator.optim')
    
    print('Training finished.')
    return
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_directory', '-dd', type=str, default='../../../../MLDS_dataset/hw3-1/AnimeDataset/faces/')
    parser.add_argument('--model_name', '-mn', type=str, default='GAN_1')
    parser.add_argument('--model_directory', '-md', type=str, default='../../../../MLDS_models/hw3-1/')
    parser.add_argument('--epoch', '-e', type=int, default=50)
    parser.add_argument('--k', '-k', type=int, default=3)
    args = parser.parse_args()
    main(args)         
