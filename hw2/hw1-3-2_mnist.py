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
EPOCH = 20
BATCHSIZE = 500
ADAMPARAM = {'lr':0.01, 'betas':(0.9, 0.999), 'eps':1e-08}
DOWNLOAD_DATASET = True

class model_1(nn.Module): #811
    def __init__(self):
        super(model_1, self).__init__()
        self.dnn_layer = nn.Sequential(
                nn.Linear(28*28, 1),
                nn.BatchNorm1d(1),
                nn.LeakyReLU(),
                nn.Linear(1, 1),
                nn.BatchNorm1d(1),
                nn.LeakyReLU(),
                nn.Linear(1, 10))
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.dnn_layer(x)
        return output

class model_2(nn.Module):
    def __init__(self):
        super(model_2, self).__init__()
        self.dnn_layer = nn.Sequential(
                nn.Linear(28*28, 1),
                nn.BatchNorm1d(1),
                nn.LeakyReLU(),
                nn.Linear(1, 4),
                nn.BatchNorm1d(4),
                nn.LeakyReLU(),
                nn.Linear(4, 10))
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.dnn_layer(x)
        return output

class model_3(nn.Module):
    def __init__(self):
        super(model_3, self).__init__()
        self.dnn_layer = nn.Sequential(
                nn.Linear(28*28, 1),
                nn.BatchNorm1d(1),
                nn.LeakyReLU(),
                nn.Linear(1, 8),
                nn.BatchNorm1d(8),
                nn.LeakyReLU(),
                nn.Linear(8, 10))
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.dnn_layer(x)
        return output

class model_4(nn.Module):
    def __init__(self):
        super(model_4, self).__init__()
        self.dnn_layer = nn.Sequential(
                nn.Linear(28*28, 1),
                nn.BatchNorm1d(1),
                nn.LeakyReLU(),
                nn.Linear(1, 16),
                nn.BatchNorm1d(16),
                nn.LeakyReLU(),
                nn.Linear(16, 10))
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.dnn_layer(x)
        return output

class model_5(nn.Module): #1599
    def __init__(self):
        super(model_5, self).__init__()
        self.dnn_layer = nn.Sequential(
                nn.Linear(28*28, 2),
                nn.BatchNorm1d(2),
                nn.LeakyReLU(),
                nn.Linear(2, 1),
                nn.BatchNorm1d(1),
                nn.LeakyReLU(),
                nn.Linear(1, 10))
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.dnn_layer(x)
        return output

class model_6(nn.Module):
    def __init__(self):
        super(model_6, self).__init__()
        self.dnn_layer = nn.Sequential(
                nn.Linear(28*28, 2),
                nn.BatchNorm1d(2),
                nn.LeakyReLU(),
                nn.Linear(2, 4),
                nn.BatchNorm1d(4),
                nn.LeakyReLU(),
                nn.Linear(4, 10))
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.dnn_layer(x)
        return output

class model_7(nn.Module):
    def __init__(self):
        super(model_7, self).__init__()
        self.dnn_layer = nn.Sequential(
                nn.Linear(28*28, 2),
                nn.BatchNorm1d(2),
                nn.LeakyReLU(),
                nn.Linear(2, 8),
                nn.BatchNorm1d(8),
                nn.LeakyReLU(),
                nn.Linear(8, 10))
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.dnn_layer(x)
        return output
    
class model_8(nn.Module):
    def __init__(self):
        super(model_8, self).__init__()
        self.dnn_layer = nn.Sequential(
                nn.Linear(28*28, 2),
                nn.BatchNorm1d(2),
                nn.LeakyReLU(),
                nn.Linear(2, 16),
                nn.BatchNorm1d(16),
                nn.LeakyReLU(),
                nn.Linear(16, 10))
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.dnn_layer(x)
        return output

class model_9(nn.Module): #6327
    def __init__(self):
        super(model_9, self).__init__()
        self.dnn_layer = nn.Sequential(
                nn.Linear(28*28, 8),
                nn.BatchNorm1d(8),
                nn.LeakyReLU(),
                nn.Linear(8, 1),
                nn.BatchNorm1d(1),
                nn.LeakyReLU(),
                nn.Linear(1, 10))
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.dnn_layer(x)
        return output

class model_10(nn.Module):
    def __init__(self):
        super(model_10, self).__init__()
        self.dnn_layer = nn.Sequential(
                nn.Linear(28*28, 8),
                nn.BatchNorm1d(8),
                nn.LeakyReLU(),
                nn.Linear(8, 4),
                nn.BatchNorm1d(4),
                nn.LeakyReLU(),
                nn.Linear(4, 10))
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.dnn_layer(x)
        return output

class model_11(nn.Module): #6274
    def __init__(self):
        super(model_11, self).__init__()
        self.dnn_layer = nn.Sequential(
                nn.Linear(28*28, 8),
                nn.BatchNorm1d(8),
                nn.LeakyReLU(),
                nn.Linear(8, 8),
                nn.BatchNorm1d(8),
                nn.LeakyReLU(),
                nn.Linear(8, 10))
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.dnn_layer(x)
        return output
    
class model_12(nn.Module): #9854
    def __init__(self):
        super(model_12, self).__init__()
        self.dnn_layer = nn.Sequential(
                nn.Linear(28*28, 12),
                nn.BatchNorm1d(12),
                nn.LeakyReLU(),
                nn.Linear(12, 16),
                nn.BatchNorm1d(16),
                nn.LeakyReLU(),
                nn.Linear(16, 10))
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.dnn_layer(x)
        return output
    
class model_13(nn.Module):
    def __init__(self):
        super(model_13, self).__init__()
        self.dnn_layer = nn.Sequential(
                nn.Linear(28*28, 16),
                nn.BatchNorm1d(16),
                nn.LeakyReLU(),
                nn.Linear(16, 1),
                nn.BatchNorm1d(1),
                nn.LeakyReLU(),
                nn.Linear(1, 10))
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.dnn_layer(x)
        return output

class model_14(nn.Module):
    def __init__(self):
        super(model_14, self).__init__()
        self.dnn_layer = nn.Sequential(
                nn.Linear(28*28, 16),
                nn.BatchNorm1d(16),
                nn.LeakyReLU(),
                nn.Linear(16, 4),
                nn.BatchNorm1d(4),
                nn.LeakyReLU(),
                nn.Linear(4, 10))
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.dnn_layer(x)
        return output

class model_15(nn.Module):
    def __init__(self):
        super(model_15, self).__init__()
        self.dnn_layer = nn.Sequential(
                nn.Linear(28*28, 16),
                nn.BatchNorm1d(16),
                nn.LeakyReLU(),
                nn.Linear(16, 8),
                nn.BatchNorm1d(8),
                nn.LeakyReLU(),
                nn.Linear(8, 10))
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.dnn_layer(x)
        return output
    
class model_16(nn.Module):
    def __init__(self):
        super(model_16, self).__init__()
        self.dnn_layer = nn.Sequential(
                nn.Linear(28*28, 24),
                nn.BatchNorm1d(24),
                nn.LeakyReLU(),
                nn.Linear(24, 16),
                nn.BatchNorm1d(16),
                nn.LeakyReLU(),
                nn.Linear(16, 10))
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.dnn_layer(x)
        return output
    
class model_17(nn.Module):
    def __init__(self):
        super(model_17, self).__init__()
        self.dnn_layer = nn.Sequential(
                nn.Linear(28*28, 32),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(),
                nn.Linear(32, 1),
                nn.BatchNorm1d(1),
                nn.LeakyReLU(),
                nn.Linear(1, 10))
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.dnn_layer(x)
        return output

class model_18(nn.Module): #25374
    def __init__(self):
        super(model_18, self).__init__()
        self.dnn_layer = nn.Sequential(
                nn.Linear(28*28, 32),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(),
                nn.Linear(32, 4),
                nn.BatchNorm1d(4),
                nn.LeakyReLU(),
                nn.Linear(4, 10))
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.dnn_layer(x)
        return output

class model_19(nn.Module):
    def __init__(self):
        super(model_19, self).__init__()
        self.dnn_layer = nn.Sequential(
                nn.Linear(28*28, 32),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(),
                nn.Linear(32, 8),
                nn.BatchNorm1d(8),
                nn.LeakyReLU(),
                nn.Linear(8, 10))
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.dnn_layer(x)
        return output
    
class model_20(nn.Module): #25914
    def __init__(self):
        super(model_20, self).__init__()
        self.dnn_layer = nn.Sequential(
                nn.Linear(28*28, 32),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(),
                nn.Linear(32, 16),
                nn.BatchNorm1d(16),
                nn.LeakyReLU(),
                nn.Linear(16, 10))
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.dnn_layer(x)
        return output
    
class model_23(nn.Module): #29910
    def __init__(self):
        super(model_23, self).__init__()
        self.dnn_layer = nn.Sequential(
                nn.Linear(28*28, 36),
                nn.BatchNorm1d(36),
                nn.LeakyReLU(),
                nn.Linear(36, 32),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(),
                nn.Linear(32, 10))
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.dnn_layer(x)
        return output

class model_26(nn.Module): #36121
    def __init__(self):
        super(model_26, self).__init__()
        self.dnn_layer = nn.Sequential(
                nn.Linear(28*28, 45),
                nn.BatchNorm1d(45),
                nn.LeakyReLU(),
                nn.Linear(45, 12),
                nn.BatchNorm1d(12),
                nn.LeakyReLU(),
                nn.Linear(12, 10))
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.dnn_layer(x)
        return output

class model_25(nn.Module): #38518
    def __init__(self):
        super(model_25, self).__init__()
        self.dnn_layer = nn.Sequential(
                nn.Linear(28*28, 48),
                nn.BatchNorm1d(48),
                nn.LeakyReLU(),
                nn.Linear(48, 12),
                nn.BatchNorm1d(12),
                nn.LeakyReLU(),
                nn.Linear(12, 10))
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.dnn_layer(x)
        return output

class model_22(nn.Module): #39738
    def __init__(self):
        super(model_22, self).__init__()
        self.dnn_layer = nn.Sequential(
                nn.Linear(28*28, 48),
                nn.BatchNorm1d(48),
                nn.LeakyReLU(),
                nn.Linear(48, 32),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(),
                nn.Linear(32, 10))
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.dnn_layer(x)
        return output

class model_24(nn.Module): #39738
    def __init__(self):
        super(model_24, self).__init__()
        self.dnn_layer = nn.Sequential(
                nn.Linear(28*28, 54),
                nn.BatchNorm1d(54),
                nn.LeakyReLU(),
                nn.Linear(54, 32),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(),
                nn.Linear(32, 10))
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.dnn_layer(x)
        return output
   
class model_21(nn.Module): #51610
    def __init__(self):
        super(model_21, self).__init__()
        self.dnn_layer = nn.Sequential(
                nn.Linear(28*28, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(),
                nn.Linear(64, 16),
                nn.BatchNorm1d(16),
                nn.LeakyReLU(),
                nn.Linear(16, 10))
    
    def forward(self, x):
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
    ###Load MNIST dataset###
    train_data = torchvision.datasets.MNIST(
                 root = './MNIST_dataset',
                 train = True,
                 transform = torchvision.transforms.ToTensor(),
                 download = DOWNLOAD_DATASET)

    ###DATALOADER###
    train_dataloader = Data.DataLoader(
                       dataset = train_data,
                       batch_size = BATCHSIZE,
                       shuffle = True,
                       num_workers = 0)
    
    testset = torchvision.datasets.MNIST(
              root='./MNIST_dataset', 
              train=False,
              transform = torchvision.transforms.ToTensor(),
              download = DOWNLOAD_DATASET)
    testloader = Data.DataLoader( 
                 dataset = testset, 
                 batch_size = BATCHSIZE,
                 shuffle = False, 
                 num_workers = 0)
    
    ###LOAD MODEL###
    if args.model_num == '1': #811
        train_model = model_1().to(device)
    elif args.model_num == '2':
        train_model = model_2().to(device)
    elif args.model_num == '3':
        train_model = model_3().to(device)
    elif args.model_num == '4':
        train_model = model_4().to(device)
    elif args.model_num == '5': #1599
        train_model = model_5().to(device)
    elif args.model_num == '6':
        train_model = model_6().to(device)
    elif args.model_num == '7': #1704
        train_model = model_7().to(device)
    elif args.model_num == '8':
        train_model = model_8().to(device)
    elif args.model_num == '9':
        train_model = model_9().to(device)
    elif args.model_num == '10':
        train_model = model_10().to(device)
    elif args.model_num == '11': #6274
        train_model = model_11().to(device)
    elif args.model_num == '12': #9854
        train_model = model_12().to(device)
    elif args.model_num == '13':
        train_model = model_13().to(device)
    elif args.model_num == '14':
        train_model = model_14().to(device)
    elif args.model_num == '15':
        train_model = model_15().to(device)
    elif args.model_num == '16':
        train_model = model_16().to(device)
    elif args.model_num == '17':
        train_model = model_17().to(device)
    elif args.model_num == '18': #25374
        train_model = model_18().to(device)
    elif args.model_num == '19':
        train_model = model_19().to(device)
    elif args.model_num == '20': #25914
        train_model = model_20().to(device)
    elif args.model_num == '23': #29910
        train_model = model_23().to(device)
    elif args.model_num == '26': #36121
        train_model = model_26().to(device)
    elif args.model_num == '25': #39738
        train_model = model_25().to(device)
    elif args.model_num == '22': #39738
        train_model = model_22().to(device)
    elif args.model_num == '24': #44652
        train_model = model_24().to(device)
    elif args.model_num == '21': #51610
        train_model = model_21().to(device)

        
    print('parameters : ', parameter_count(train_model))
    ###OPTIMIZER###
    optimizer = torch.optim.Adam(train_model.parameters(), lr=ADAMPARAM['lr'], betas=ADAMPARAM['betas'], eps=ADAMPARAM['eps'],weight_decay = 1e-06)
    
    ###LOSS FUNCTION###
    loss_func = nn.CrossEntropyLoss()
        
    ###RECORD###
    train_loss_history = []
    train_acc_history = []
    test_loss_history = []
    test_acc_history = []
        
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
            #print("Batch: ", b_num, "loss: ", loss.item(), end = '\r')
            epoch_loss += loss.item()
            epoch_acc  += torch.sum(torch.eq(torch.argmax(pred, dim=1), b_y), dim=0).item()
        
        torch.save(train_model, 'model_'+ args.model_num + '.pkl')
        torch.save(optimizer.state_dict(), 'model_'+ args.model_num + '.optim')

        print("Epoch loss: ", epoch_loss / len(train_data))
        print("Epoch acc:  ", epoch_acc  / len(train_data))
        train_loss_history.append(epoch_loss / len(train_data))
        train_acc_history.append(epoch_acc  / len(train_data))
        print("")

    print('Finished Training')
    
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for num, (images, labels) in enumerate(testloader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = train_model(images)
            test_loss += loss_func(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
    print("")   
    print("Test loss: ", test_loss / len(testset))
    print("Test acc:  ", correct  / total)
    test_loss_history.append( test_loss / len(testset) )
    test_acc_history.append( correct  / total )

    ###Train###-------------------------------
    ###LOSS HISTORY###
    plt.figure(1)
    plt.plot(np.arange(EPOCH)+1, train_loss_history)
    plt.savefig('DNN_'+args.model_num+'_loss.png', format='png')
    np.save('DNN_'+args.model_num+'_loss', np.array( [ train_loss_history[len(train_loss_history)-1], test_loss_history[0] ] ))
    #print( train_loss_history[len(train_loss_history)-1], '---' ,test_loss_history[0])
    
    ###ACCURACY HISTORY###
    plt.figure(2)
    plt.plot(np.arange(EPOCH)+1, train_acc_history)
    plt.savefig('DNN_'+args.model_num+'_acc.png', format='png')
    np.save('DNN_'+args.model_num+'_acc', np.array( [ train_acc_history[len(train_acc_history)-1], test_acc_history[0] ] ))
    #print( train_acc_history[len(train_acc_history)-1], '---' ,test_acc_history[0])
    ###---------------------------------------


    
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='number of parameters vs generalization')
    parser.add_argument('--model_num', '-num', type=str, default='26')
    args = parser.parse_args()
    main(args)
