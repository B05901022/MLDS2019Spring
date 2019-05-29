# -*- coding: utf-8 -*-
"""
Created on Mon May 13 14:07:03 2019

@author: Austin Hsu
"""
"""
Cannot run. The Section of Data Processing is needed to change 
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
ADAMPARAM2= {'lr':0.0001, 'betas':(0.5, 0.999), 'eps':1e-5}
SGDPARAM  = {'lr':0.0002, 'momentum':0.9}
BATCHSIZE = 96
WGANCLIP  = 0.01

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.to_2d = nn.Linear(100,256*16*16)
        self.txt_emb = nn.Sequential(nn.Linear(130,256),
                                     nn.ReLU())
        self.fc = nn.Linear(256*257,4*4*256)
        self.conv_layers = nn.Sequential(nn.BatchNorm2d(256),
                                         nn.LeakyReLU(),
                                         nn.ConvTranspose2d(256, 
                                                            128, 
                                                            kernel_size=4, 
                                                            stride=2, 
                                                            padding=1,
                                                            ), #128 * 32 * 32
                                                
                                        nn.BatchNorm2d(128),
                                        nn.LeakyReLU(),
                                        nn.ConvTranspose2d(128,
                                                           64,
                                                           kernel_size=4,
                                                           stride=2,
                                                           padding=1,
                                                           ), #64 * 64 * 64
                                        nn.BatchNorm2d(64),
                                        nn.LeakyReLU(),
                                        nn.ConvTranspose2d(64,
                                                           32,
                                                           kernel_size=4,
                                                           stride=2,
                                                           padding=1,
                                                           ), #3 * 64 * 64               
                                        nn.BatchNorm2d(32),
                                        nn.LeakyReLU(),
                                        nn.ConvTranspose2d(32,16,
                                                    
                                                           kernel_size=4,
                                                           stride=2,
                                                           padding=1,
                                                           ), #3 * 64 * 64               
                                        nn.BatchNorm2d(16),
                                        nn.LeakyReLU(),
                                        nn.ConvTranspose2d(16,
                                                           3,
                                                           kernel_size=4,
                                                           stride=2,
                                                           padding=1,
                                                           ), #3 * 64 * 64               
                                        nn.Tanh(),#interval[0,1.0]
                                        
                                        )
        
    def forward(self, x, label):
        x = self.to_2d(x)
        y = self.txt_emb(label)
        #print(x.shape,y.shape)
        x = torch.cat([x,y],dim=1)
        x = self.fc(x)
        x = x.view(-1,256,4,4)
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
                                     nn.ReLU(),
                                    )    
        self.conv_layers1 = nn.Sequential(nn.Conv2d(3,
                                                   32,
                                                   kernel_size=5,
                                                   stride=2,
                                                   padding=0,
                                                   ), #32 * 64 * 64
                                        #nn.BatchNorm2d(32),
                                        nn.LeakyReLU(),
                                        nn.Dropout(0.7),
                                        nn.Conv2d(32,
                                                  64,
                                                  stride=2,
                                                  kernel_size=5,
                                                  padding=0,
                                                  ), #64 * 32 * 32
                                        nn.LeakyReLU(),
                                        nn.Dropout(0.7),
                                        #nn.BatchNorm2d(64),
                                        nn.Conv2d(64,
                                                  128,
                                                  stride=2,
                                                  kernel_size=5,
                                                  padding=0,
                                                  ), #128 * 16 * 16
                                        #nn.BatchNorm2d(64),
                                        nn.LeakyReLU(),
                                        nn.Dropout(0.7),
                                        nn.Conv2d(128,
                                                  256,
                                                  stride=2,
                                                  kernel_size=5,
                                                  padding=0,
                                                  ), #256 * 8 * 8
                                        #nn.BatchNorm2d(128),
                                        nn.LeakyReLU(),
                                        nn.Dropout(0.7),
                                        
                                        )
        self.to_out = nn.Sequential(
                                    nn.Linear(256*5*10, 1),
                                    nn.Sigmoid()
                                    )
        self.conv_layers2 = nn.Sequential(nn.Conv2d(256,
                                                  256,
                                                  kernel_size=1,
                                                  stride=(1,1),
                                                  )
                                         )
    def forward(self, x, label):
        y = self.txt_emb(label)
        y = y.view(-1,256,1,1)
        y = y.repeat([1,1,5,5])
        x = self.conv_layers1(x)
        print(x.shape,y.shape,99)
        x = torch.cat([x,y],dim=2)
        print(x.shape,101)
        x = self.conv_layers2(x)
        x = x.view(-1,256*5*10)
        #print(x.shape,100)
        x = self.to_out(x)
        #print(x.shape,2)
        return x

def criterion_d(generated, data, samplesize):
    
    """
    (batch, channel, height, weight)
    """
    
    return (torch.mean((1-data) ** 2) + torch.mean((generated) ** 2)) * 0.5
    #return (torch.sum(generated) - torch.sum(data)) / samplesize
    
def criterion_g(generated, samplesize):
    
    """
    (batch, channel, height, weight)
    """
    
    return torch.mean((1 - generated) ** 2) * 0.5
    
def main(args):
    
    """
    //---------------------------------
    Data loading and data preprocessing
    ---------------------------------//
    """
    '''

    transform = transforms.Compose(
            [transforms.Scale([64,64]),
             transforms.ToTensor(),
             #transforms.Normalize((0.0,0.0,0.0),(1.0,1.0,1.0))
             ])
    traindata = torchvision.datasets.ImageFolder(root=args.data_directory, transform=transform)
    train_dataloader = Data.DataLoader(traindata, batch_size=BATCHSIZE)
    
    total_batch = len(traindata) // BATCHSIZE    
    '''
    data=torch.Tensor(np.load(args.data))
    label=torch.Tensor(np.load(args.label))
    dataset = Data.TensorDataset(data,label)
    train_dataloader = Data.DataLoader(dataset, batch_size=BATCHSIZE)
    total_batch=data.shape[0]//BATCHSIZE+1
    
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
    optimizer_g = torch.optim.Adam(train_generator.parameters(), **ADAMPARAM)
    optimizer_d = torch.optim.Adam(train_discriminator.parameters(), **ADAMPARAM2)
    
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

            """
            Train D
            """
            
            """
            sample_tag = torch.from_numpy(np.random.choice(BATCHSIZE, BATCHSIZE//10, replace=False))
            data_d     = torch.index_select(b_x, 1, sample_tag).cuda()
            """
            for generating_train in range(args.k):
                data_d  = b_x.view(-1,3,128,128).cuda()
                data_label = b_y.cuda()
                sample_noise = noise_distribution.sample((BATCHSIZE, 100)).squeeze(2).cuda()
                optimizer_d.zero_grad()
                generated = train_generator(sample_noise,data_label)
                generated = train_discriminator(generated,data_label)
                data_d    = train_discriminator(data_d,data_label)
                dloss = loss_func_d(generated=generated, data=data_d, samplesize=BATCHSIZE)
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
            
            sample_noise = noise_distribution.sample((BATCHSIZE, 100)).squeeze(2).cuda()
            optimizer_g.zero_grad()
            generated = train_generator(sample_noise,data_label)
            generated = train_discriminator(generated,data_label)
            gloss = loss_func_g(generated=generated, samplesize=BATCHSIZE)
            gloss.backward()
            epoch_gloss += gloss.item()
            optimizer_g.step()
            
            ##################################################################################################################
            
            """
            Save Model
            """
            
            torch.save(train_generator, args.model_directory + args.model_name + '_epoch_' + str(e+1) + '_generator.pkl')
            torch.save(optimizer_g, args.model_directory + args.model_name + '_epoch_' + str(e+1) + '_generator.optim')
            torch.save(train_discriminator, args.model_directory + args.model_name + '_epoch_' + str(e+1) + '_discriminator.pkl')
            torch.save(optimizer_d, args.model_directory + args.model_name + '_epoch_' + str(e+1) + '_discriminator.optim')
            print('batch: ', b_num, '/', total_batch, ' Discriminator Loss: ', (epoch_dloss-old_dloss)/args.k, ' Generator Loss: ', epoch_gloss-old_gloss, end='\r')
            old_dloss, old_gloss = epoch_dloss, epoch_gloss
        
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
    parser.add_argument('--model_directory', '-md', type=str, default='H:/hw3-2/')
    parser.add_argument('--epoch', '-e', type=int, default=50)
    parser.add_argument('--k', '-k', type=int, default=2)
    args = parser.parse_args()
    main(args)         
