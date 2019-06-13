from agent_dir.agent import Agent
import torch
import torch.nn as nn
import numpy as np
import collections
import torch.nn.functional as F
import math

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

class NoisyLinear(nn.Module):
    # === Factorised Gaussian NoisyNet ===
    #
    # Reference: https://github.com/hengyuan-hu/rainbow/blob/master/model.py
    #

    def __init__(self, in_feature, out_feature, bias=True, sigma_0=0.5):
        super(NoisyLinear, self).__init__()
        self.in_feature   = in_feature
        self.out_feature  = out_feature
        self.weight       = nn.Parameter(torch.Tensor(out_feature, in_feature))
        self.bias         = nn.Parameter(torch.Tensor(out_feature))
        self.noisy_weight = nn.Parameter(torch.Tensor(out_featurem in_feature))
        self.noisy_bias   = nn.Parameter(torch.Tensor(out_feature))
        self.reset_parameters()
        
        self.noise_std    = sigma_0 / math.sqrt(in_feature)
        self.in_noise     = torch.FloatTensor(in_feature).to(device)
        self.out_noise    = torch.FloatTensor(out_feature).to(device)
        self.noise        = None
        self.sample_noise()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.noisy_weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        self.noisy_bias.data.uniform_(-stdv, stdv)

    def sample_noise(self):
        self.in_noise.normal_(0, self.noise_std)
        self.out_noise.normal_(0, self.noise_std)
        self.noise = torch.mm(self.out_noise.view(-1, 1), self.in_noise.view(1, -1))

    def forward(self, x):
        normal_y = F.linear(x, self.weight, self.bias)
        if not x.volatile:
            #x.volatile = True in inference
            self.sample_noise

        noisy_weight = self.noisy_weight * torch.autograd.Variable(self.noise)
        noisy_bias   = self.noisy_bias   * torch.autograd.Variable(self.out_noise)
        noisy_y  = F.linear(x, noisy_weight, noisy_bias)

        return normal_y + noisy_y

class DQN_Model(nn.Module):
    
    def __init__(self, dueling=True, noisy=True):
        super (DQN_Model,self).__init__()

        self.dueling = dueling
        self.noisy   = noisy

        # === Model ===

        """
        input shape: (1,4,84,84)
        """
        
        self.conv_layer = nn.Sequential(nn.Conv2d(4,  32, kernel_size=5, stride=2), #(1, 32, 40, 40)
                                        nn.ReLU(),
                                        nn.Conv2d(32, 64, kernel_size=5, stride=2), #(1, 64, 18, 18)
                                        nn.ReLU(),
                                        nn.Conv2d(64, 64, kernel_size=5, stride=2), #(1, 64, 7,  7 )
                                        )
        if self.noisy:
            self.linr_layer = NoisyLinear(64*7*7, 512)

            if self.dueling:
                self.q_v    = NoisyLinear(512, 1)
                self.q_A    = NoisyLinear(512, 3)
            else:
                self.q      = NoisyLinear(512, 3)

        else:
            self.linr_layer = nn.Linear(64*7*7, 512)

            if self.dueling:
                self.q_V    = nn.Linear(512, 1)
                self.q_A    = nn.Linear(512, 3)
            else:
                self.q      = nn.Linear(512, 3)


    def forward(self,x):
        x = self.conv_layer(x) #(1, 64, 7, 7)
        x = x.view(1, -1) #(1, 64*7*7)
        x = self.linr_layer(x) #(1, 512)
        if self.dueling:
            #Alternative Q-function
            V = self.q_V(x) #(1, 1)
            A = self.q_A(x) #(1, 3)
            A = A - torch.mean(A, dim=1, keepdim=True)
            return V + A
        else:
            A = self.q(x)
            return A 
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
            self.model=torch.load('./model/hw4-2'+args.dqn_model_name+'_'+str(args.dqn_test_episode)+'.pkl')
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
            self.current_update         = args.dqn_current_update
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
            self.target_model.load_state_dict(self.policy_model.state_dict())
            if self.optim=="RMSprop":
                self.optimizer_p=torch.optim.RMSprop(self.policy_model.parameters(),lr=self.lr,alpha=0.9)
                self.optimizer_t=torch.optim.RMSprop(self.target_model.parameters(),lr=self.lr,alpha=0.9)
            elif self.optim=="Adam":
                self.optimizer_p=torch.optim.Adam(self.policy_model.parameters(),lr=self.lr,betas=(0.9,0.999))
                self.optimizer_t=torch.optim.Adam(self.target_model.parameters(),lr=self.lr,betas=(0.9,0.999))
            elif self.optim=="SGD":
                self.optimizer_p=torch.optim.SGD(self.policy_model.parameters(),lr=self.lr)
                self.optimizer_t=torch.optim.SGD(self.target_model.parameters(),lr=self.lr)
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

        """
        Observation shape: (84,84,4)
        """

        o = self.env.reset()
        print(o)

        
        return

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

       
