#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 00:43:02 2019

@author: austinhsu
"""

import os
import cv2
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

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
   
def main(args):
    
    """
    //---------------------------------
    Data loading and data preprocessing
    ---------------------------------//
    """
    
    """
    //------
    Training
    ------//
    """
    
    test_generator = torch.load(args.model_directory + args.model_name + '_epoch_' + '5' + '_generator.pkl').eval().cuda()
    #test_discriminator = torch.load(args.model_directory + args.model_name + '_epoch_' + args.epoch + '_discriminator.pkl').cuda()

    noise_distribution = torch.distributions.Normal(torch.Tensor([0.0]), torch.Tensor([1.0]))
    
    print('Testing starts...')

    for e in tqdm(range(1)):
        sample_noise = noise_distribution.sample((25, 100)).squeeze(2).cuda()
        generated_waifu = test_generator(sample_noise)
        
        """
        SAVE PICTURE
        """
        
        r, c = 5, 5
        generated_waifu = generated_waifu + torch.ones((25,3,64,64)).cuda()
        generated_waifu = generated_waifu * 255/2
        generated_waifu = generated_waifu.detach().cpu().numpy().astype(np.int32)
        generated_waifu = np.transpose(generated_waifu, [0,2,3,1])
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(generated_waifu[cnt, :,:,:])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("./generated/" + args.model_name +".png")
        plt.close()
        
        
    print('Testing finished.')
    return
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--data_directory', '-dd', type=str, default='../../../../MLDS_dataset/hw3-1/AnimeDataset/faces/')
    parser.add_argument('--model_name', '-mn', type=str, default='WGAN2')
    parser.add_argument('--model_directory', '-md', type=str, default='../../../MLDS_models/hw3-1/')
    parser.add_argument('--epoch', '-e', type=int, default=50)
    #parser.add_argument('--k', '-k', type=int, default=3)
    args = parser.parse_args()
    main(args)         
