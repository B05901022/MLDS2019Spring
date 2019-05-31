#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 02:34:14 2019

@author: austinhsu
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
import matplotlib.pyplot as plt

import const

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.txt_emb = nn.Sequential(nn.Linear(130,256),
                                     nn.BatchNorm1d(256, momentum=0.9),
                                     nn.LeakyReLU(),
                                     )
        self.fc = nn.Sequential(nn.Linear(356,4*4*512),
                                #nn.BatchNorm1d(4*4*512, momentum=0.9),
                                #nn.LeakyReLU(),
                                )
        self.conv_layers = nn.Sequential(nn.BatchNorm2d(512, momentum=0.9),
                                         nn.LeakyReLU(),
                                         nn.Dropout(0.1),
                                         nn.ConvTranspose2d(512, 
                                                            256, 
                                                            kernel_size=4, 
                                                            stride=2, 
                                                            padding=1,
                                                            ), #128 * 32 * 32
                                                
                                        nn.BatchNorm2d(256, momentum=0.9),
                                        nn.LeakyReLU(),
                                        nn.Dropout(0.1),
                                        nn.ConvTranspose2d(256,
                                                           128,
                                                           kernel_size=4,
                                                           stride=2,
                                                           padding=1,
                                                           ), #64 * 64 * 64
                                        nn.BatchNorm2d(128, momentum=0.9),
                                        nn.LeakyReLU(),
                                        nn.Dropout(0.1),
                                        nn.ConvTranspose2d(128,
                                                           64,
                                                           kernel_size=4,
                                                           stride=2,
                                                           padding=1,
                                                           ), #3 * 64 * 64               
                                        nn.BatchNorm2d(64, momentum=0.9),
                                        nn.LeakyReLU(),
                                        nn.Dropout(0.1),
                                        nn.ConvTranspose2d(64,
                                                           3,
                                                           kernel_size=4,
                                                           stride=2,
                                                           padding=1,
                                                           ), #3 * 64 * 64               
                                        #nn.BatchNorm2d(3, momentum=0.9),
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
                                        nn.Dropout(0.7),
                                        nn.Conv2d(32,
                                                  64,
                                                  stride=2,
                                                  kernel_size=5,
                                                  padding=1,
                                                  ), #64 * 14 * 14
                                        nn.BatchNorm2d(64, momentum=0.9),
                                        nn.LeakyReLU(),
                                        nn.Dropout(0.7),
                                        nn.Conv2d(64,
                                                  128,
                                                  stride=2,
                                                  kernel_size=5,
                                                  padding=1,
                                                  ), #128 * 6 * 6
                                        nn.BatchNorm2d(128, momentum=0.9),
                                        nn.LeakyReLU(),
                                        nn.Dropout(0.7),
                                        nn.Conv2d(128,
                                                  256,
                                                  stride=1,
                                                  kernel_size=5,
                                                  padding=1,
                                                  ), #256 * 4 * 4
                                        nn.BatchNorm2d(256, momentum=0.9),
                                        nn.LeakyReLU(),
                                        nn.Dropout(0.7),
                                        
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
                                         nn.BatchNorm2d(512, momentum=0.9),
                                         nn.LeakyReLU(),
                                         nn.Dropout(0.7),
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
    
def main(args):
    
    tag_dict = const.tag_to_idx
    tags_wanted = ['pink hair black eyes', 
                   'black hair purple eyes',
                   'red hair red eyes',
                   'aqua hair green eyes',
                   'blonde hair orange eyes',
                   ]
    tags_wanted = [tag_dict[i] for i in tags_wanted]
    tags = torch.zeros(25,130)
    for i in range(len(tags_wanted)):
        tags[5*i:5*(i+1),tags_wanted[i]] = 1
    tags = tags.cuda()
    
    noise_distribution = torch.distributions.Normal(torch.Tensor([0.0]), torch.Tensor([1.0]))
    sample_noise = noise_distribution.sample((5, 100)).squeeze(2)
    sample_noise = torch.stack((sample_noise,sample_noise,sample_noise,sample_noise,sample_noise), dim=0).view(-1,100).cuda()

    r, c = 5, 5
    
    for e in tqdm(range(args.epoch)):
        test_generator = torch.load(args.model_directory+args.model_name+'_epoch_'+str(e+1)+'_generator.pkl').eval().cuda()        
        generated_waifu = test_generator(sample_noise, tags)
        generated_waifu = (-1*generated_waifu + 1)/2*255
        generated_waifu = generated_waifu.detach().cpu().numpy().astype(np.int32)
        generated_waifu = np.transpose(generated_waifu, [0,2,3,1])
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(generated_waifu[cnt, :,:,:])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("./generated/" + args.model_name + '_' + str(e+1) + ".png")
        plt.close()

        
    return 
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', '-mn', type=str, default='CGAN_3_LS_LS_LABELSMOOTH_False')
    parser.add_argument('--model_directory', '-md', type=str, default='../../../MLDS_models/hw3-2/')
    parser.add_argument('--epoch', '-e', type=int, default=50)
    #parser.add_argument('--k', '-k', type=int, default=2)
    args = parser.parse_args()
    main(args)   