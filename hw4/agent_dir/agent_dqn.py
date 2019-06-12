from agent_dir.agent import Agent
import torch
import torch.nn as nn
import numpy as np
import collections
import torch.nn.functional as F
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

class DQN_Model(nn.Module):
    
    def __init__(self):
        super (DQN,self).__init__()
        self.conv1=nn.Conv2d(1,3,kernel_size=4,stride=2,padding=0)
        self.conv2=nn.Conv2d(3,3,kernel_size=4,stride=2,padding=0)
        self.flatten=nn.Linear(3*18*18,128)
        self.dense1=nn.Linear(80*80,128)
        self.dense2=nn.Linear(256,1)
        
    def forward(self,x):
        x1=self.conv1(x)
        x1=F.ReLU(x1)
        x1=self.conv2(x1)
        x1=self.flatten(x1)
        x1=F.ReLU(x1)
        x2=self.dense1(x)
        x2=F.ReLU(x2)
        x=torch.cat((x1,x2),dim=1)
        x=self.dense2(x)
        x=F.Sigmoid(x)
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
            self.last_frame=None
        elif args.train_dqn:
            self.episode                = args.dqn_episode
            self.batchsize              = args.dqn_batchsize
            self.replay_buffer          = collections.deque()
            self.replay_buffer_capacity = args.dqn_capacity
            self.traincurve             = []
            self.eps_start              = args.dqn_eps_start
            self.eps_decay              = args.dqn_eps_decay
            self.eps_end                = args.dqn_eps_end
            self.target_update          = args.dqn_target_update        
            self.train_method           = args.dqn_train_method #Must be "Epsilon" or "Boltzmann"
            self.optim                  = args.dqn_optim           #Must be "RMSprop","Adam" or "SGD"
            self.lr                     = args.dqn_lr
            
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
            
            self.policy_model=DQN_Model().to(device) #The shape seems (84*84*1), and the models may need to be modified.
            self.target_model=DQN_Model().to(device)
            if self.optim=="RMSprop":
                self.optimizer=torch.optim.RMSprop(self.policy_model.parameters(),lr=self.lr,alpha=0.9)
            elif self.optim=="Adam":
                self.optimizer=torch.optim.Adam(self.policy_model.parameters(),lr=self.lr,betas=(0.9,0.999))
            elif self.optim=="SGD":
                self.optimizer=torch.optim.SGD(self.policy_model.parameters(),lr=self.lr)
        ##################
        # YOUR CODE HERE #
        ##################


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
        for episode in range(self.episode):
            for times in range(self.target_update):
                
        pass
    def update_epsilon(self,eps_start,eps_end,eps_decay):
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
        q_value=torch.Tensor(self.policy_model)
        if self.train_method=="Epsilon":
            c=np.random.uniform()
            if c>self.eps_start:
                action=torch.argmax(q_value).item()
            else:
                action=np.random.randint(1,4)
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

       
