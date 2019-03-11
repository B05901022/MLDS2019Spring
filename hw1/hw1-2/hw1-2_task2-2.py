import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import torch.functional as F
import numpy as np
import argparse

###DEVICE###
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###REPRODUCIBLE###
torch.manual_seed(1)

###HYPERPARAMETER###
EPOCH = 250
BATCHSIZE = 500
ADAMPARAM = {'lr':0.001, 'betas':(0.9, 0.999), 'eps':1e-08}
DOWNLOAD_DATASET = True

"""
Total Parameter:
    net_shallow:72106
    net_deep:72030
"""

###SHALLOW CNN MODEL###
class net_shallow(nn.Module):
    def __init__(self):
        super(net_shallow, self).__init__()
        self.conv_layer = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=(3,3), stride=(1,1), padding=1), #32 * 32
                nn.BatchNorm2d(8),
                nn.LeakyReLU(),
                nn.AvgPool2d((2,2)), #16 * 16
                nn.Conv2d(8, 16, kernel_size=(3,3), stride=(1,1), padding=1), #16 * 16
                nn.BatchNorm2d(16),
                nn.LeakyReLU(),
                nn.AvgPool2d((2,2)) #8 * 8
                )
        self.dnn_layer = nn.Sequential(
                nn.Linear(16*8*8, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(),
                nn.Linear(64, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(),
                nn.Linear(64, 10))
        
    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        output = self.dnn_layer(x)
        return output

###DEEP CNN MODEL###
class net_deep(nn.Module):
    def __init__(self):
        super(net_deep, self).__init__()
        self.conv_layer = nn.Sequential(
                nn.Conv2d(3, 4, kernel_size=(3,3), stride=(1,1), padding=1), #32 * 32
                nn.BatchNorm2d(4),
                nn.LeakyReLU(),
                nn.Conv2d(4, 4, kernel_size=(3,3), stride=(1,1), padding=1), #32 * 32
                nn.BatchNorm2d(4),
                nn.LeakyReLU(),
                nn.Conv2d(4, 4, kernel_size=(3,3), stride=(1,1), padding=1), #32 * 32
                nn.BatchNorm2d(4),
                nn.LeakyReLU(),
                nn.AvgPool2d((2,2)), #16 * 16
                nn.Conv2d(4, 8, kernel_size=(3,3), stride=(1,1), padding=1), #16 * 16
                nn.BatchNorm2d(8),
                nn.LeakyReLU(),
                nn.Conv2d(8, 8, kernel_size=(3,3), stride=(1,1), padding=1), #16 * 16
                nn.BatchNorm2d(8),
                nn.LeakyReLU(),
                nn.AvgPool2d((2,2)) #8 * 8
                )
        self.dnn_layer = nn.Sequential(
                nn.Linear(8*8*8, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(),
                nn.Linear(128, 32),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(),
                nn.Linear(32, 16),
                nn.BatchNorm1d(16),
                nn.LeakyReLU(),
                nn.Linear(16,10))
        
    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        output = self.dnn_layer(x)
        return output

def parameter_count(input_model):
    model_params = list(input_model.parameters())
    total_params = 0
    for i in model_params:
        layer_parmas = 1
        for j in i.size():
            layer_parmas *= j
        total_params += layer_parmas
    return total_params

def main(args):
    ###Load CIFAR10 dataset###
    train_data = torchvision.datasets.CIFAR10(
            root = './CIFAR10_dataset',
            train = True,
            transform = torchvision.transforms.ToTensor(),
            download = DOWNLOAD_DATASET)
    """
    test_data = torchvision.datasets.CIFAR10(
            root = './CIFAR10_dataset',
            train = False)
    """
    ###DATALOADER###
    train_dataloader = Data.DataLoader(
            dataset = train_data,
            batch_size = BATCHSIZE,
            shuffle = True,
            num_workers = 1)
    
    ###LOAD MODEL###
    if args.model_type == 'shallow':
        train_model = net_shallow().to(device)
    elif args.model_type == 'deep':
        train_model = net_deep().to(device)
    
    ###OPTIMIZER###
    #optimizer = torch.optim.Adam(train_model.parameters(), lr=ADAMPARAM['lr'], betas=ADAMPARAM['betas'], eps=ADAMPARAM['eps'], weight_decay=1e-5)
    optimizer = torch.optim.SGD(train_model.parameters(), lr=0.01, momentum=0.9)
    
    ###LOSS FUNCTION###
    loss_func = nn.CrossEntropyLoss()
    
    ###RECORD###
    loss_history = []
    acc_history = []
    grad_history = []
    
    for e in range(EPOCH):
        
        print("Epoch ", e)
        epoch_loss = 0
        epoch_acc  = 0
        
        for b_num, (b_x, b_y) in enumerate(train_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            optimizer.zero_grad()
            pred = train_model(b_x)
            loss = loss_func(pred, b_y)
            loss.backward()
            optimizer.step()
            print("Batch: ", b_num, "loss: ", loss.item(), end = '\r')
            epoch_loss += loss.item()
            epoch_acc  += torch.sum(torch.eq(torch.argmax(pred, dim=1), b_y), dim=0).item()
            
            grad_all = 0.0
            grad_norm = 0.0
            #total_norm = 0.0
            for p in train_model.parameters():
                grad = 0.0
                #param_norm = 0.0
                if p.grad is not None:
                    #grad = (p.grad.cpu().data.numpy() ** 2).sum()
                    grad = p.grad.data.norm(2)
                    #total_norm += param_norm.item() ** 2
                grad_all += grad
                #total_norm = total_norm ** (1. / 2)
            grad_norm = grad_all ** 0.5
            grad_history.append(grad_norm)
            
        torch.save(train_model, args.model_type+'_model.pkl')
        torch.save(optimizer.state_dict(), args.model_type+'_model.optim')
        print("")   
        print("Epoch loss: ", epoch_loss / len(train_data))
        print("Epoch acc:  ", epoch_acc  / len(train_data))
        loss_history.append(epoch_loss / len(train_data))
        acc_history.append(epoch_acc  / len(train_data))
    
    ###LOSS HISTORY###
    plt.figure(1)
    plt.plot(np.arange(EPOCH)+1, loss_history)
    plt.savefig('picturesCNN_'+args.model_type+'_loss.png', format='png')
    np.save('CNN_'+args.model_type+'_loss', np.array(loss_history))
    
    ###ACCURACY HISTORY###
    plt.figure(2)
    plt.plot(np.arange(EPOCH)+1, acc_history)
    plt.savefig('picturesCNN_'+args.model_type+'_acc.png', format='png')
    np.save('CNN_'+args.model_type+'_acc', np.array(acc_history))
    
    ###GRADIENT NORM HISTORY#
    plt.figure(3)
    plt.plot(np.arange(EPOCH*100)+1, grad_history)
    plt.savefig('pictures'+args.model_type+'_grad.png', format='png')
    np.save('CNN_'+args.model_type+'_grad', np.array(grad_history))
    
    return 
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', '-type', type=str, default='shallow')
    args = parser.parse_args()
    main(args)

