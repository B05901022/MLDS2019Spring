import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import torch.functional as F
import numpy as np

###DEVICE###
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###REPRODUCIBLE###
torch.manual_seed(1)

###HYPERPARAMETER###
EPOCH = 120
BATCHSIZE = 500
ADAMPARAM = {'lr':0.01, 'betas':(0.9, 0.999), 'eps':1e-08}
DOWNLOAD_DATASET = True

###CNN MODEL###
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_layer = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=(3,3), stride=(1,1), padding=1), #32 * 32
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
                nn.MaxPool2d((2,2)), #16 * 16
                nn.Conv2d(32, 64, kernel_size=(3,3), stride=(1,1), padding=1), #16 * 16
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                nn.Conv2d(64, 16, kernel_size=(3,3), stride=(1,1), padding=1), #16 * 16
                nn.BatchNorm2d(16),
                nn.LeakyReLU(),
                nn.Conv2d(16, 8, kernel_size=(3,3), stride=(1,1), padding=1), #16 * 16
                nn.BatchNorm2d(8),
                nn.LeakyReLU(),
                nn.MaxPool2d((2,2)), #8 * 8
                )
        self.dnn_layer = nn.Sequential(
                nn.Linear(8*8*8, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(),
                nn.Linear(128, 16),
                nn.BatchNorm1d(16),
                nn.LeakyReLU(),
                nn.Linear(16, 10))
        
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

###Load MNIST dataset###
train_data = torchvision.datasets.CIFAR10(
             root = './CIFAR10_dataset',
             train = True,
             transform = torchvision.transforms.ToTensor(),
             download = DOWNLOAD_DATASET)

###DATALOADER###
train_dataloader = Data.DataLoader(
                   dataset = train_data,
                   batch_size = BATCHSIZE,
                   shuffle = False,
                   num_workers = 0)

testset = torchvision.datasets.CIFAR10(
          root='./CIFAR10_dataset', 
          train=False,
          transform = torchvision.transforms.ToTensor(),
          download = DOWNLOAD_DATASET)
testloader = Data.DataLoader( 
             dataset = testset, 
             batch_size = BATCHSIZE,
             shuffle = False, 
             num_workers = 0)

train_model = Net().to(device)
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

shuffle = torch.randint(0, 10, (100,BATCHSIZE))
#x = torch.tensor([1., 2.], device=cuda0)
shuffle_rate = 0
shuffleMap = torch.randint(0,2,(BATCHSIZE,))
#shuffleMap = torch.randint(0,3,(BATCHSIZE,))

for b_num, (b_x, b_y) in enumerate(train_dataloader):
    if(shuffleMap[b_num].item() == 1):
        shuffle_rate += torch.nonzero(shuffle[b_num]-b_y).size(0)
        #print("shuffle rate: ", shuffle_rate )

print("shuffle rate: ", shuffle_rate / len(train_data) )

shuffle = torch.tensor(shuffle, device = 'cuda:0')



for e in range(EPOCH):
        
    print("Epoch ", e)
    epoch_loss = 0
    epoch_acc  = 0
        
    for b_num, (b_x, b_y) in enumerate(train_dataloader):
        #shuffle_rate += torch.nonzero(shuffle[b_num]-b_y).size(0)
        b_x = b_x.to(device)
        b_y = b_y.to(device)
        #shuffle[b_num] = shuffle[b_num].to(device)
        optimizer.zero_grad()
        pred = train_model(b_x)
        if shuffleMap[b_num].item() == 1:
            loss = loss_func(pred, shuffle[b_num])
        else:
            loss = loss_func(pred, b_y)
        loss.backward()
        optimizer.step()
        print("Batch: ", b_num, "loss: ", loss.item(), end = '\r')
        epoch_loss += loss.item()
        if shuffleMap[b_num].item() == 1:
            epoch_acc  += torch.sum(torch.eq(torch.argmax(pred, dim=1), shuffle[b_num]), dim=0).item()
        else:
            epoch_acc  += torch.sum(torch.eq(torch.argmax(pred, dim=1), b_y), dim=0).item()
        
    print("")   
    print("Epoch loss: ", epoch_loss / len(train_data))
    print("Epoch acc:  ", epoch_acc  / len(train_data))
    train_loss_history.append(epoch_loss / len(train_data))
    train_acc_history.append(epoch_acc  / len(train_data))
    
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
    print("Test epoch loss: ", test_loss / len(testset))
    print("Test epoch acc:  ", correct  / total)
    test_loss_history.append( test_loss / len(testset) )
    test_acc_history.append( correct  / total )
    

print('Finished Training')

"""
correct = 0
total = 0
with torch.no_grad():
    for num, (images, labels) in enumerate(testloader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = train_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
"""
###Train###-------------------------------
###LOSS HISTORY###
plt.figure(1)
plt.plot(np.arange(EPOCH)+1, train_loss_history, color='blue', label='train')
plt.plot(np.arange(EPOCH)+1, test_loss_history, color='orange', label='test')

    
###ACCURACY HISTORY###
plt.figure(2)
plt.plot(np.arange(EPOCH)+1, train_acc_history, color='blue', label='train')
plt.plot(np.arange(EPOCH)+1, test_acc_history, color='orange', label='test')
###---------------------------------------

###Test###-------------------------------
###LOSS HISTORY###
plt.figure(3)
plt.plot(np.arange(EPOCH)+1, test_loss_history)

    
###ACCURACY HISTORY###
plt.figure(4)
plt.plot(np.arange(EPOCH)+1, test_acc_history)
###---------------------------------------





