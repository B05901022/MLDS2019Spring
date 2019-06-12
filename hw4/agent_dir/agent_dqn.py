from agent_dir.agent import Agent
import torch
import torch.nn as nn
import numpy as np
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
np.random.seed(1024)
torch.cuda.manual_seed(1024)
torch.manual_seed(1024)
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
        x=self.conv1(x)
        x=nn.ReLU(x)
        x=self.conv2(x)
        x=self.flatten(x)
        x=nn.ReLU(x)
        x=self.dense1(x)
        x=nn.ReLU(x)
        x=self.dense2(x)
        x=nn.Sigmoid(x)
        return (x)
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
            self.episode=args.episode
            self.batchsize=args.batchsize
            self.replay_buffer=collections.deque()
            self.replay_buffer_capacity=args.capacity
            self.traincurve=[]
            self.eps_start=args.eps_start
            self.eps_decay=args.eps_decay
            self.eps_end=args.eps_end
            self.target_update=args.target_update        
            self.train_method=args.train_method #Must be "Epsilon" or "Boltzmann"
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
            self.policy_model=DQN()
            self.traget_model=DQN()
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
                
        #pass
    def update_epsilon(eps_start,eps_end,eps_decay):
        if eps_start>eps_end:
            eps_start=eps_start-eps_decay
        return eps_start

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

       
