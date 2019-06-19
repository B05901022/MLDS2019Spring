from agent_dir.agent import Agent
#import scipy
import numpy as np
#import skimage
import torch
import torch.nn as nn
import random
#import math
import torch.nn.functional as F
import math
import os
import yaml
import random
from environment import Environment
from tqdm import tqdm

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
random.seed(11037)
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def prepro(o):
    # (84,84,4)
    #o = np.array(o).astype(np.float32) / 255.0 #(84,84,4)
    #o = np.mean(o, axis=2)                     #(84,84)   greyscale
    o = np.transpose(o, (2,0,1))               
    o = np.expand_dims(o, axis=0)
    #o = np.expand_dims(o, axis=0)
    return o

class NoisyLinear(nn.Linear):
    # === Factorised Gaussian NoisyNet ===
    
    def __init__(self, in_feature, out_feature, bias=True, sigma_0=0.5):
        super(NoisyLinear, self).__init__(in_feature, out_feature, bias=True)
        self.in_feature   = in_feature
        self.out_feature  = out_feature
        self.sigma_w = nn.Parameter(nn.init.uniform_(torch.rand(self.out_feature, self.in_feature),
                                                     a = -1/self.in_feature, b=1/self.out_feature))
        self.sigma_b = nn.Parameter(torch.Tensor(out_feature).fill_(sigma_0/self.in_feature))
        
        self.register_buffer("epsilon_i", torch.zeros(1,  self.in_feature))
        self.register_buffer("epsilon_j", torch.zeros(self.out_feature, 1))
    
    def remove_noise(self):
        torch.zeros(self.epsilon_i.size(), out=self.epsilon_i)
        torch.zeros(self.epsilon_j.size(), out=self.epsilon_j)
        
    def sample_noise(self):
        torch.randn(self.epsilon_i.size(), out=self.epsilon_i)
        torch.randn(self.epsilon_j.size(), out=self.epsilon_j)

    def forward(self, x, fixed_noise=False):
        if not fixed_noise:
            self.sample_noise()
        
        func = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))
        
        epsilon_i = func(self.epsilon_i)
        epsilon_j = func(self.epsilon_j)
        epsilon_w = torch.mul(epsilon_j, epsilon_i)
        epsilon_b = epsilon_j.squeeze(dim=1)

        return F.linear(x, self.weight+self.sigma_w*epsilon_w, self.bias+self.sigma_b*epsilon_b)
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
        
        self.conv_layer = nn.Sequential(nn.Conv2d(4,  32, kernel_size=7, stride=2), 
                                        nn.ReLU(),
                                        nn.Conv2d(32, 64, kernel_size=5, stride=2), 
                                        nn.ReLU(),
                                        nn.Conv2d(64, 64, kernel_size=4, stride=2), 
                                        nn.ReLU(),
                                        )
        if self.noisy:
            self.linr_layer = nn.Sequential(NoisyLinear(64*8*8, 512),
                                            nn.ReLU(),
                                            )

            if self.dueling:
                self.q_V    = NoisyLinear(512, 1)
                self.q_A    = NoisyLinear(512, 3)
            else:
                self.q      = NoisyLinear(512, 3)

        else:
            self.linr_layer = nn.Sequential(nn.Linear(64*8*8, 512),
                                            nn.ReLU(),
                                            )

            if self.dueling:
                self.q_V    = nn.Linear(512, 1)
                self.q_A    = nn.Linear(512, 3)
            else:
                self.q      = nn.Linear(512, 3)


    def forward(self,x, fixed_noise=False):
        x = self.conv_layer(x) #(1, 64, 7, 7)
        x = x.view(x.size(0), -1) #(1, 64*7*7)
        x = self.linr_layer(x) #(1, 512)
        if self.noisy:
            if self.dueling:
                #Alternative Q-function
                V = self.q_V(x, fixed_noise) #(1, 1)
                A = self.q_A(x, fixed_noise) #(1, 3)
                A = A - torch.mean(A, dim=1, keepdim=True)
                return V + A
            else:
                A = self.q(x, fixed_noise)
                return A 
        else:
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

def test(test_agent, test_env, test_episode, test_seed):
    rewards = []
    test_env.seed(test_seed)
    for i in tqdm(range(test_episode)):
        state = test_env.reset()
        test_agent.init_game_setting()
        done = False
        episode_reward = 0.0

        #playing one game
        while(not done):
            action = test_agent.make_action(state, test=True)
            state, reward, done, _ = test_env.step(action)
            episode_reward += reward

        rewards.append(episode_reward)
    print('Run %d episodes'%(test_episode))
    print('Mean:', np.mean(rewards))
    return np.mean(rewards)

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

            self.policy_model = torch.load('./model/hw4-2/' + args.dqn_model_name + '_' + str(args.dqn_test_episode) + 'p.pkl')
            if args.dqn_noisy:
                self.policy_model.linr_layer[0].remove_noise()
                if args.dqn_dueling:
                    self.policy_model.q_V.remove_noise()
                    self.policy_model.q_A.remove_noise()
                else:
                    self.policy_model.q.remove_noise()
            self.policy_model.to(device)
            print("Model : "+args.dqn_model_name+'_'+str(args.dqn_test_episode))

        elif args.train_dqn:
            self.dqn_model_name         = args.dqn_model_name
            self.episode                = args.dqn_episode
            self.batchsize              = args.dqn_batchsize
            self.traincurve             = []
            self.policy_update          = args.dqn_policy_update
            self.target_update          = args.dqn_target_update
            self.DDQN                   = args.dqn_ddqn
            self.gamma                  = args.dqn_gamma
            self.test_args              = args
            self.test_seed              = 11037
            self.testcurve              = []

        ##################
        # YOUR CODE HERE #
        ##################
            if args.dqn_load_model:
                """
                things to save:
                    policy_model
                    target_model
                    optimizer
                    load_params:
                        'step_count':   int
                        'epsilon':      float
                        #'replay_buffer':collections.deque #Not a good idea...

                """
                self.noisy   = args.dqn_noisy
                self.dueling = args.dqn_dueling
                self.policy_model = torch.load('./model/hw4-2/' + args.dqn_model_name + '_' + str(args.dqn_test_episode) + 'p.pkl')
                self.target_model = torch.load('./model/hw4-2/' + args.dqn_model_name + '_' + str(args.dqn_test_episode) + 't.pkl')
                load_params       = yaml.load(open('./model/hw4-2/' + args.dqn_model_name + '_' + str(args.dqn_test_episode) + '.yaml'))
                self.step_count   = load_params['step_count']
                self.traincurve   = load_params['traincurve']
                self.testcurve    = load_params['testcurve']
                self.train_method = args.dqn_train_method
                self.eps = None
                if (self.train_method == 'Epsilon') and (not self.noisy):
                    self.eps       = load_params['epsilon']
                    self.eps_end   = args.dqn_eps_end
                    self.eps_decay = args.dqn_eps_decay

                self.replay_buffer_capacity = args.dqn_capacity
                self.replay_buffer = collections.deque([], maxlen=self.replay_buffer_capacity) #load_params['replay_buffer']
                self.optim = args.dqn_optim
                self.lr    = args.dqn_lr
                if   self.optim == "RMSprop":
                    print('Optimizer: RMSprop')
                    self.optimizer = torch.optim.RMSprop(self.policy_model.parameters(),lr=self.lr,alpha=0.9)

                elif self.optim == "Adam":
                    print('Optimizer: Adam')
                    self.optimizer = torch.optim.Adam(self.policy_model.parameters(),lr=self.lr,betas=(0.9,0.999))
                elif self.optim == "SGD":
                    print('Optimizer: SGD')
                    self.optimizer = torch.optim.SGD(self.policy_model.parameters(),lr=self.lr)
                else:
                    print('No optimizer type given, use default optimizer Adam')
                    self.optimizer = torch.optim.Adam(self.policy_model.parameters(),lr=self.lr,betas=(0.9,0.999))
                state_dict = torch.load('./model/hw4-2/' + args.dqn_model_name + '_' + str(args.dqn_test_episode) + '.optim')
                self.optimizer.load_state_dict(state_dict)

                self.env.clip_reward = False
            else:
                if os.path.isfile('./model/hw4-2/'+args.dqn_model_name+'_200.pkl'):
                    confirm = input('Model already exists. Overwrite it? [y/N]')
                    if confirm != 'y':
                        print('Process aborted.')
                        exit()
                    else:
                        print('Overwriting model: ', args.dqn_model_name)

                self.noisy   = args.dqn_noisy
                self.dueling = args.dqn_dueling
                self.policy_model = DQN_Model(dueling=args.dqn_dueling, noisy=args.dqn_noisy)
                self.target_model = DQN_Model(dueling=args.dqn_dueling, noisy=args.dqn_noisy)
                self.target_model.load_state_dict(self.policy_model.state_dict())
                self.step_count   = 0
                self.train_method = args.dqn_train_method
                self.eps = None
                if (self.train_method == 'Epsilon') and (not self.noisy):
                    self.eps = args.dqn_eps_start
                    self.eps_end   = args.dqn_eps_end
                    self.eps_decay = args.dqn_eps_decay

                self.replay_buffer_capacity = args.dqn_capacity
                self.replay_buffer = collections.deque([], maxlen=self.replay_buffer_capacity)

                self.optim = args.dqn_optim
                self.lr    = args.dqn_lr
                if   self.optim == "RMSprop":
                    print('Optimizer: RMSprop')
                    self.optimizer = torch.optim.RMSprop(self.policy_model.parameters(),lr=self.lr,alpha=0.9)
                elif self.optim == "Adam":
                    print('Optimizer: Adam')
                    self.optimizer = torch.optim.Adam(self.policy_model.parameters(),lr=self.lr,betas=(0.9,0.999))
                elif self.optim == "SGD":
                    print('Optimizer: SGD')
                    self.optimizer = torch.optim.SGD(self.policy_model.parameters(),lr=self.lr)
                else:
                    print('No optimizer type given, use default optimizer Adam')
                    self.optimizer = torch.optim.Adam(self.policy_model.parameters(),lr=self.lr,betas=(0.9,0.999))

                self.env.clip_reward = False

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

        """
        Observation shape: (84,84,4)
        """
        
        self.policy_model = self.policy_model.to(device)
        self.target_model = self.target_model.to(device)

        batchsize = self.batchsize
        batchdata = None
        latest_r  = collections.deque([], maxlen=100)

        fancy_loading_boiii = collections.deque('>-------------')

        for episode in range(self.episode):
            o = self.env.reset()
            o = prepro(o)
            unclipped_episode_r = 0
            while True:
                if self.step_count % 10 == 0:
                    fancy_loading_boiii.rotate(1)
                print(''.join(fancy_loading_boiii), end='\r')
                a = self.make_action(o)
                o_nxt, r, done, _ = self.env.step(a+1) # (0,1,2) --> (1,2,3)
                o_nxt = prepro(o_nxt)
                unclipped_episode_r += r

                r = np.sign(r) #clip reward
                state_replay = (o, a, r, o_nxt, done)
                o = o_nxt
                self.step_count += 1

                if (self.train_method == 'Epsilon') and (not self.noisy):
                    self.update_epsilon()

                self.replay_buffer.append(state_replay)

                if ( len(self.replay_buffer) >= batchsize ) and ( ( self.step_count % self.policy_update ) == 0 ):
                    if self.DDQN:
                        batchdata = random.sample(self.replay_buffer, batchsize)
                        loss = self.update_DDQN(batchdata)
                        batchdata = None
                        print('\t\tLoss : %4f' % loss, end='\r')
                    else:
                        batchdata = random.sample(self.replay_buffer, batchsize)
                        loss = self.update_DQN(batchdata)
                        batchdata = None
                        print('\t\tLoss : %4f' % loss, end='\r')
                if ( self.step_count % self.target_update == 0 ):
                    self.target_model.load_state_dict(self.policy_model.state_dict())

                if done:
                    latest_r.append(unclipped_episode_r)
                    print("Episode : %5d Mean : %4f Latest : %4f" % \
                          (episode+1, np.mean(latest_r), unclipped_episode_r), end = '\n')
                    self.traincurve.append(np.mean(latest_r))
                    break
            
            unclipped_episode_r = 0
            batchdata = None
            if ((episode+1) % 200 == 0):
                """
                things to save:
                    policy_model
                    target_model
                    optimizer
                    load_params:
                        'step_count':   int
                        'epsilon':      float
                        #'replay_buffer':collections.deque #Not a good idea...
                """
                #print()
                """
                print('Testing model...')
                test_env = Environment('BreakoutNoFrameskip-v4', self.test_args, atari_wrapper=True, test=True)
                test_env.clip_reward = False
                result = test(test_agent=self,
                              test_env=test_env,
                              test_episode=100,
                              test_seed=self.test_seed,
                              )
                self.testcurve.append(result)
                """
                print('Saving model...')
                torch.save(self.policy_model, './model/hw4-2/' + self.dqn_model_name + '_' + str(episode+1) + 'p.pkl')
                torch.save(self.target_model, './model/hw4-2/' + self.dqn_model_name + '_' + str(episode+1) + 't.pkl')
                torch.save(self.optimizer.state_dict(), './model/hw4-2/' + self.dqn_model_name + '_' + str(episode+1) + '.optim')
                save_params = {'step_count': self.step_count,
                               'epsilon':    self.eps,
                               'traincurve': self.traincurve,
                               'testcurve':  self.testcurve,
                               }
                yaml.dump(save_params, open('./model/hw4-2/' + self.dqn_model_name + '_' + str(episode+1) + '.yaml', 'w'))
                print('Model saved.')
        pass

    def update_DQN(self, batchdata):

        self.optimizer.zero_grad()
        loss = 0
        
        if self.noisy and self.dueling:
            self.policy_model.q_V.sample_noise()
            self.target_model.q_V.sample_noise()

        batch_o     = []
        batch_a     = []
        batch_r     = []
        batch_o_nxt = []
        batch_d     = []

        for replay_data in batchdata:
            batch_o.append(replay_data[0])
            batch_a.append(replay_data[1])
            batch_r.append(replay_data[2])
            batch_o_nxt.append(replay_data[3])
            batch_d.append(replay_data[4])

        q_list   = []
        tgt_list = []
        policy_output = self.policy_model(torch.Tensor(batch_o).squeeze(1).to(device)) #(batchsize, 3)
        target_output = self.target_model(torch.Tensor(batch_o_nxt).squeeze(1).to(device)).detach() #(batchsize, 3)
        a_nxt = torch.argmax(target_output, dim=1)

        for i in range(self.batchsize):
            q_list.append(policy_output[i, batch_a[i]])
            tgt_list.append(target_output[i, a_nxt[i]])

        q_list   = torch.stack(q_list)
        tgt_list = torch.stack(tgt_list)
        batch_d  = torch.Tensor(batch_d).to(device)
        batch_r  = torch.Tensor(batch_r).to(device)
        loss = F.mse_loss(q_list, (1-batch_d)*0.99*tgt_list + batch_r)

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_DDQN(self, batchdata):

        self.optimizer.zero_grad()
        loss = 0
        if self.noisy and self.dueling:
            self.policy_model.q_V.sample_noise()
            self.target_model.q_V.sample_noise()

        batch_o     = []
        batch_a     = []
        batch_r     = []
        batch_o_nxt = []
        batch_d     = []

        for replay_data in batchdata:
            batch_o.append(replay_data[0])
            batch_a.append(replay_data[1])
            batch_r.append(replay_data[2])
            batch_o_nxt.append(replay_data[3])
            batch_d.append(replay_data[4])

        q_list   = []
        tgt_list = []
        policy_output = self.policy_model(torch.Tensor(batch_o).squeeze(1).to(device),     fixed_noise=self.noisy)
        policy_nextop = self.policy_model(torch.Tensor(batch_o_nxt).squeeze(1).to(device), fixed_noise=self.noisy)
        policy_nextop = torch.argmax(policy_nextop, dim=1)
        target_output = self.target_model(torch.Tensor(batch_o_nxt).squeeze(1).to(device), fixed_noise=self.noisy).detach()

        for i in range(self.batchsize):
            q_list.append(policy_output[i, batch_a[i]])
            tgt_list.append(target_output[i, policy_nextop[i]])

        q_list   = torch.stack(q_list)
        tgt_list = torch.stack(tgt_list)
        batch_d  = torch.Tensor(batch_d).to(device)
        batch_r  = torch.Tensor(batch_r).to(device)
        loss = F.mse_loss(q_list, (1-batch_d)*self.gamma*tgt_list + batch_r)

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_epsilon(self):
        if self.eps >= self.eps_end:
            self.eps -= self.eps_decay
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

    def make_action(self, observation, test=False):
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

        if not test:
            q_value = self.policy_model(torch.Tensor(observation).to(device))
            if self.noisy:
                """
                prob = F.softmax(q_value, dim=1)[0]
                random_num = np.random.rand()
                if random_num < prob[0]:
                    action = 0
                elif random_num < prob[0] + prob[1]:
                    action = 1
                else:
                    action = 2
                return action
                """
                action = torch.argmax(q_value)
                return action.item()
            else:
                if self.train_method == 'Epsilon':
                    if np.random.rand() < self.eps:
                        action = np.random.randint(3)
                        return action
                    else:
                        action = torch.argmax(q_value)
                        return action.item()
                elif self.train_method == 'Boltzmann':
                    prob = F.softmax(q_value, dim=1)[0]
                    random_num = np.random.rand()
                    if random_num < prob[0]:
                        action = 0
                    elif random_num < prob[0] + prob[1]:
                        action = 1
                    else:
                        action = 2
                    return action
        else:
            observation = prepro(observation)
            q_value     = self.policy_model(torch.Tensor(observation).to(device), fixed_noise=True)
            action      =  torch.argmax(q_value).item()+1
            return action
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

       
