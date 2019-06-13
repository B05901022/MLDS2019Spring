from agent_dir.agent import Agent
#import scipy
import numpy as np
#import skimage
import torch
import torch.nn as nn
import random
#import math
import torch.nn.functional as F
from environment import Environment
from tqdm import tqdm
import collections
"""     
class Replay_Memory(object):
    def __init__(self,capacity=128):
        self.memory=([])
        self.capacity=capacity
        self.position=0
    def push(self,state_transfer):
        if self.memory.shape[0]<self.capacity:
            self.memory.append(state_transfer)
            self.position+=1
        else:
            self.position=self.position%self.capacity
            self.memory[self.position]=state_transfer
            self.position+=1
    def sample(self):
        return self.memory[np.random.randint(0,self.capacity)]
"""
np.random.seed(11037)
torch.cuda.manual_seed(11037)
torch.manual_seed(11037)
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def prepro(o):
    o = np.transpose(o, (2,0,1))
    o = np.expand_dims(o, axis = 0)
    return o
class DQN_Model(nn.Module):
    
    def __init__(self):
        super (DQN_Model,self).__init__()
        self.conv1=nn.Conv2d(4,32,kernel_size=7,stride=2,padding=0)
        self.conv2=nn.Conv2d(32,64,kernel_size=5,stride=2,padding=0)
        self.conv3=nn.Conv2d(64,64,kernel_size=4,stride=2,padding=0)
        self.dense1=nn.Linear(64*8*8,512)
        self.dense2=nn.Linear(512,3)
        
    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=F.relu(self.conv3(x))
        x=x.view(x.size(0),-1)
        x=F.relu(self.dense1(x))
        x=F.relu(self.dense2(x))
        x=torch.sigmoid(x)
        return x
    
class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        

        super(Agent_DQN,self).__init__(env)

        if args.test_dqn:
            print("Loading trained model...")
            print("Model : "+args.model_name+'_'+str(args.test_episode))
            self.model=torch.load('./model/hw4-2'+args.model_name+'_'+str(args.test_episode)+'.pkl')
            #you can load your model here
            #print('loading trained model')
            self.model=self.model.to(device)
        elif args.train_dqn:
            if args.load_model:
                model=torch.load(args.model_name+".pkl")
                self.policy_model=model["policy_model"] #The shape seems (84*84*4), and the models may need to be modified.
                self.target_model=model["target_model"]
                if model["train_method"]=="Epsilon":
                    self.eps_start=model["eps_start"]
                    self.eps_decay=model["eps_decay"]
                    self.eps_end=model["eps_end"]
                    self.target_update = model["target_update"]     
                    self.train_method= model["train_method"]  #Must be "Epsilon" or "Boltzmann"
                    self.optim = model["optim"]           #Must be "RMSprop","Adam" or "SGD"
                    self.lr= model["lr"] 
                    self.sample_freqency= model["sample_freqency"] 
                    self.model_name= model["model_name"] 
                    self.step=model["step"]
                    self.replay_buffer_capacity = model["capacity"]
                    self.training_curve            = []
                    self.step                   = model["step"]
            else:
                self.policy_model=DQN_Model().to(device) #The shape seems (84*84*1), and the models may need to be modified.
                self.target_model=DQN_Model().to(device)
                self.update_target_model()
                self.step                   = 0
                self.episode                = args.dqn_episode
                self.batchsize              = args.dqn_batchsize
                self.replay_buffer          = collections.deque([])
                self.replay_buffer_capacity = args.dqn_capacity
                self.training_curve            = []
                self.eps_start              = args.dqn_eps_start
                self.eps_decay              = args.dqn_eps_decay
                self.eps_end                = args.dqn_eps_end
                self.target_update          = args.dqn_target_update        
                self.train_method           = args.dqn_train_method #Must be "Epsilon" or "Boltzmann"
                self.optim                  = args.dqn_optim           #Must be "RMSprop","Adam" or "SGD"
                self.lr                     = args.dqn_lr
            #self.training_curve         = args.dqn_training_curve
                self.sample_freqency        = args.dqn_sample_frequency
                self.model_name             = args.dqn_model_name
            
            """
            self.policy_model=nn.ModuleList([nn.Sequential(nn.Conv2d(1,3,kernel_size=4,stride=2,padding=0), # (1,3,39,39)
                                                          nn.ReLU(),
                                                          nn.Conv2d(3,3,kernel_size=4,stride=2,padding=0), # (1,3,18,18)
                                                          ),
                                            nn.Sequential(nn.Linear(3*18*18,128),
                                                          nn.ReLU(),
                                                          ),
                                            nn.Sequential(nn.Linear(80*80,128),
                                                          nn.ReLU(),
                                                          ),
                                            nn.Sequential(nn.Linear(256,1),
                                                          nn.Sigmoid(),
                                                          ),
                                            ]).cuda()
            self.target_model=self.policy_model
            """
            
            
            if self.optim=="RMSprop":
                self.optimizer=torch.optim.RMSprop(self.policy_model.parameters(),lr=self.lr,alpha=0.9)
            elif self.optim=="Adam":
                self.optimizer=torch.optim.Adam(self.policy_model.parameters(),lr=self.lr,betas=(0.9,0.999))
            elif self.optim=="SGD":
                self.optimizer=torch.optim.SGD(self.policy_model.parameters(),lr=self.lr)
        ##################
        # YOUR CODE HERE #
        ##################
        #print(self.policy_model)


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        batch_size   = self.batchsize
        sampled_data = None
        latest_r    = collections.deque([],maxlen=100)
        for episode in range(self.episode):
            o = self.env.reset()
            o = prepro(o)
            unclipped_r = 0
            done = 0
            while not done:
                action = self.make_action(o)
                #print(self.env.step(action+1))
                o_next,r,done,_ = self.env.step(action+1)  #(0,1,2) to (1,2,3)
                o_next=prepro(o_next)
                unclipped_r+=r
                r=np.sign(r) #Change r to make it be -1 or +1
                state_transfer=(o,action,r,o_next,done)
                o=o_next
                self.step+=1
                if self.train_method=="Epsilon":
                    self.update_epsilon()
                self.replay_buffer.append(state_transfer)
                
                if len(self.replay_buffer)>batch_size and \
                self.step%self.sample_freqency==0:
                    sampled_data=random.sample(self.replay_buffer,batch_size)
                    loss=self.update_param_DQN(sampled_data)
                    print ("Loss : %3f " % (loss))
                if self.step%(self.target_update*self.sample_freqency)==0:
                    print("Update target...")
                    self.update_target_model()
                    
                    
            #Game Over
            latest_r.append(unclipped_r)
            print("Episode :"+str(episode+1)+" Mean :"+str(np.mean(latest_r))\
                  +" Latest :"+str(unclipped_r))
            self.training_curve.append(np.mean(latest_r))
            
            unclipped_r=0
            if (episode+1)%500==0:
                parameter={"policy_model":self.policy_model,
                           "target_model":self.target_model,
                           "train_method":self.train_method,
                           "eps_start": self.eps_start,
                           "eps_decay": self.eps_decay,
                           "eps_end": self.eps_end,
                           "optim":self.optim,        
                           "lr":self.lr,
                           "sample_freqency":self.sample_freqency,
                           "model_name":self.model_name,
                           "step":self.step,
                           "capacity":self.replay_buffer_capacity,
                           "step":self.step
                          }
                torch.save(parameter,self.model_name+".pkl")
           # for times in range(self.target_update):
                
        #pass
    def update_epsilon(self):
        if self.eps_start>self.eps_end:
            self.eps_start=self.eps_start-self.eps_decay
        else:
            pass

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        #1:NO-OP 2:Left 3:Right
        q_value=self.policy_model(torch.Tensor(observation).to(device))
        #print(q_value)
        if self.train_method=="Epsilon":
            c=np.random.uniform()
            if c>self.eps_start:
                action=torch.argmax(q_value).item()
            else:
                action=np.random.randint(0,3)
        elif self.tain_method=="Boltzmann":
            probability=F.Softmax(q_value,dim=1)
            cumulated=0
            threshold=np.random.uniform()
            for i in range(probability.shape[1]):
                cumulated+=probability[0,i]
                if cumulated>threshold:
                    action=i
                    break
        return action
        #self.env.get_random_action()
    def update_target_model(self):
        self.target_model.load_state_dict(self.policy_model.state_dict())
    def update_param_DQN(self,sampled_data):
        self.optimizer.zero_grad()
        loss=0
        
        batch_o=[]
        batch_o_next=[]
        batch_r=[]
        batch_done=[]
        
        for data in sampled_data:
            batch_o.append(data[0])
            batch_o_next.append(data[3])
            batch_r.append(data[2])
            batch_done.append(data[4])
        q_list=[]
        q_target_list=[]
        output=self.policy_model(torch.Tensor(batch_o).squeeze().to(device))
        q_target=self.target_model(torch.Tensor(batch_o_next).squeeze().to(device)).detach()
        action_next=torch.argmax(q_target, dim=1)
        for i in range(len(sampled_data)):
            q_list.append(output[i,sampled_data[i][1]])
            q_target_list.append(q_target[i,action_next[i]])
        q_list=torch.stack(q_list)
        q_target_list=torch.stack(q_target_list)
        batch_done=torch.Tensor(batch_done).to(device)
        batch_r=torch.Tensor(batch_r).to(device)
        loss=F.mse_loss(q_list,(1-batch_done)*0.99*q_target_list+batch_r)

        loss.backward()
        self.optimizer.step()
        return loss.item()     

       
