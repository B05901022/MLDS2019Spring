from agent_dir.agent import Agent
#import scipy
import numpy as np
import skimage
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from environment import Environment

np.random.seed(11037)

def prepro(o,
           image_size=[105,80],#[80,80],
           ):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
    
    Input: 
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array 
        Grayscale image, shape: (80, 80, 1)
    
    """
    
    #y = o.astype(np.uint8)
    #resized = scipy.misc.imresize(y, image_size)

    y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]   #gray scale
    resized = skimage.transform.resize(y, image_size)[17:-8,:]            #delete score board
    
    return np.expand_dims(resized.astype(np.float32),axis=2)              #shape (height, wodth) -> (1, height, wodth)

def optimizer_sel(model_name, model_params, optim, lr, load):
    if optim == 'Adam':
        print('Optimizer: Adam')
        if load:
            state_dict = torch.load('./model/hw4-1/' + model_name+'.optim')
            return torch.optim.Adam(model_params, lr=lr, betas=(0.9,0.999)).load_state_dict(state_dict)
        else:
            return torch.optim.Adam(model_params, lr=lr, betas=(0.9,0.999))
    elif optim == 'RMSprop':
        print('Optimizer: RMSprop')
        if load:
            state_dict = torch.load('./model/hw4-1/' + model_name+'.optim')
            return torch.optim.RMSprop(model_params, lr=lr, alpha=0.99).load_state_dict(state_dict)
        else:
            return torch.optim.RMSprop(model_params, lr=lr, alpha=0.99)
    elif optim == 'SGD':
        print('Optimizer: SGD')
        return torch.optim.SGD(model_params, lr=lr)
    
def test_agent(test_agent, test_env, test_episode, test_seed):
    
    test_rwd = []
    test_env.seed = test_seed
    
    for episode in range(test_episode):
        state = test_env.reset()
        test_agent.init_game_setting()
        done = False
        episode_rwd = 0
        
        while not done:
            action = test_agent.make_action(state, test=True)
            state, rwd, done, _ = test_env.step(action)
            episode_rwd += rwd
            
        test_rwd.append(episode_rwd)
        
    print('Run ', test_episode, ' episodes', end=' ')
    print('Mean: ', np.mean(test_rwd))
    
    return np.mean(test_rwd)
    
class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)

        if args.test_pg:
            #you can load your model here
            print('loading trained model')
            self.model = torch.load('./model/hw4-1/' + args.model_name + '.pkl')
            self.last_frame = None

        ##################
        # YOUR CODE HERE #
        ##################
        else:
            #train model
            
            ### initialize ###
            self.last_frame = None
            self.gamma      = args.gamma
            self.batchsize  = args.batchsize
            self.episode    = args.episode
            self.ppo        = args.ppo
            self.vanilla_pg = args.vanilla_pg
            self.traincurve = []
            self.loss_func  = F.binary_cross_entropy
            
            ### for test ###
            self.test_args    = args
            self.test_episode = 30
            self.test_seed    = 11037
            
            ### for save ###
            self.model_name = args.model_name
            
            ### model ###
            if args.load_model:
                self.model = torch.load('./model/hw4-1/' + self.model_name + '.pkl').cuda()
                self.optimizer = optimizer_sel(self.model_name,
                                               self.model.parameters(),
                                               args.optim,
                                               args.lr,
                                               load=True,
                                               )
                
            else:
                self.model = nn.ModuleList([nn.Sequential(nn.Linear(80*80*1, 256),
                                            nn.ReLU(),
                                            nn.Linear(256,1),
                                            nn.Sigmoid(),
                                            )]).cuda()
                self.optimizer = optimizer_sel(None,
                                                self.model.parameters(),
                                                args.optim,
                                                args.lr,
                                                load=False,
                                                )
                


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.last_frame = None
        pass


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################

        ### initialize reward list ###
        total_rwd = []
        steps_rwd = []
        prob_list = []
        act_list  = []
        
        rwd_mean  = 0
        rwd_std   = 0
        
        n_rwd     = 0
        batch_iter= 1
        
        loss = 0
        best_result = -21
        latest_rwd  = [] #for learning curve
        
        if self.ppo:
            pass
        else:
            for episode in range(self.episode):
                
                ### initialize game ###
                
                o = self.env.reset()                    #reset game
                self.last_frame = prepro(o)             #initialize first frame
                action = self.env.action_space.sample() #random initial action
                o, _, _, _ = self.env.step(action)      #first observation
                episode_rwd = []
                
                while True:
                    
                    o = prepro(o)
                    residual_state = o - self.last_frame
                    self.last_frame = o
                    
                    action, p = self.make_action(residual_state, test=False)
                    o, rwd, done, _ = self.env.step(action)
                    
                    prob_list.append(p)
                    act_list.append(action)
                    
                    if rwd != 0:
                        """
                        Someone gets point.
                        """
                        if not self.vanilla_pg:
                            T = len(steps_rwd)
                            for steps in range(T):
                                steps_rwd_now = ( self.gamma**(T-steps) ) * rwd
                                steps_rwd[steps] += steps_rwd_now
                                rwd_mean += steps_rwd_now
                                rwd_std  += steps_rwd_now ** 2
                                
                            steps_rwd.append(rwd)
                            rwd_mean += rwd
                            rwd_std  += rwd ** 2
                            n_rwd    += T+1
                            
                            episode_rwd.append(rwd)
                            total_rwd.extend(steps_rwd)
                            steps_rwd = []
                        else:
                            T = len(steps_rwd)
                            steps_rwd = [rwd] * (T+1)
                            episode_rwd.append(rwd)
                            total_rwd.extend(steps_rwd)
                            steps_rwd = []
                        
                    else:
                        steps_rwd.append(rwd)
                    
                    if done:
                        if batch_iter != self.batchsize:
                            batch_iter += 1
                            self.init_game_setting()
                            if len(latest_rwd) < 30:
                                latest_rwd.append(np.sum(episode_rwd))
                            else:
                                latest_rwd.pop(0)
                                latest_rwd.append(np.sum(episode_rwd))
                                self.traincurve.append(np.mean(latest_rwd))
                            break
                        else:
                            batch_iter = 1
                            self.init_game_setting()
                            if len(latest_rwd) < 30:
                                latest_rwd.append(np.sum(episode_rwd))
                            else:
                                latest_rwd.pop(0)
                                latest_rwd.append(np.sum(episode_rwd))
                                self.traincurve.append(np.mean(latest_rwd))
                                print('Episode: ',episode, ' Mean: ', np.mean(latest_rwd), end=' ')
                            
                            #update parameters
                            self.optimizer.zero_grad()
                            
                            if not self.vanilla_pg:
                                batch_mean = rwd_mean / n_rwd
                                batch_std  = math.sqrt(rwd_std  / n_rwd - batch_mean ** 2)
                                
                                total_rwd = torch.Tensor(total_rwd).cuda()
                                total_rwd = (total_rwd - batch_mean) / batch_std
                            else:
                                total_rwd = torch.Tensor(total_rwd).cuda()
                            
                            act_list = torch.Tensor(act_list).cuda()
                            prob_list= torch.stack(prob_list).view(-1).cuda()
                            
                            loss = self.loss_func(input=prob_list,
                                                  target=act_list,
                                                  weight=total_rwd,
                                                  reduction='sum',
                                                  )
                            """
                            BCELoss:
                                return -1 * torch.sum(
                                                    torch.dot(weight,
                                                              target * torch.log(input) +
                                                              (1-target) * torch.log(1-input),
                                                              )
                                                     )
                            """
                            loss /= self.batchsize
                            loss.backward()
                            self.optimizer.step()
                            
                            ### initialize for next batch ###
                            total_rwd = []
                            steps_rwd = []
                            prob_list = []
                            act_list  = []
        
                            rwd_mean  = 0
                            rwd_std   = 0
        
                            n_rwd     = 0
                            batch_iter= 1
                            
                            print('Loss: ', loss.item(), end='\r')
                            break
                        
                if episode % 200 == 0 and episode != 0:
                    print()
                    print('Testing')
                    test_env = Environment('Pong-v0', self.test_args, test=True)
                    result = test_agent(test_agent=self,
                                        test_env=test_env,
                                        test_episode=self.test_episode,
                                        test_seed=self.test_seed,
                                        )
                    if result > best_result:
                        best_result = result
                        torch.save(self.model, './model/hw4-1/'+ self.model_name + str(episode) + '.pkl')
                        torch.save(self.optimizer.state_dict, './model/hw4-1/'+ self.model_name + str(episode) + '.optim')
                        np.save('./training_curve/hw4-1/' + self.model_name + str(episode) + '.npy', np.array(self.traincurve))
                        print('Testing finished.')
                        print('Model saved.')
                        print('Result: ', best_result)
        np.save('./training_curve/hw4-1/' + self.model_name + '.npy', np.array(self.traincurve))
        return


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        
        if test:
            if type(self.last_frame) == type(None):
                observation = prepro(observation)
                self.last_frame = observation
            else:
                observation = prepro(observation)
                observation = observation - self.last_frame
                self.last_frame = observation
            observation = torch.Tensor(observation).view(1,-1).cuda()
            prob = self.model[0](observation)
            if np.random.rand() < prob[0,0].item():
                action = 3
            else:
                action = 2
            return action
        else:
            observation = torch.Tensor(observation).view(1,-1).cuda()
            prob = self.model[0](observation)
            if np.random.rand() < prob[0,0].item():
                action = 3
            else:
                action = 2 
            
            return action, prob #self.env.get_random_action()

