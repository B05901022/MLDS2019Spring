from agent_dir.agent import Agent
import scipy
import numpy as np
import skimage
import torch
import torch.nn as nn

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
            state_dict = torch.load(model_name+'.optim')
            return torch.optim.Adam(model_params, lr=lr, betas=(0.9,0.999)).load_state_dict(state_dict)
        else:
            return torch.optim.Adam(model_params, lr=lr, betas=(0.9,0.999))
    elif optim == 'RMSprop':
        print('Optimizer: RMSprop')
        if load:
            state_dict = torch.load(model_name+'.optim')
            return torch.optim.RMSprop(model_params, lr=lr, alpha=0.99).load_state_dict(state_dict)
        else:
            return torch.optim.RMSprop(model_params, lr=lr, alpha=0.99)
    elif optim == 'SGD':
        print('Optimizer: SGD')
        return torch.optim.SGD(model_params, lr=lr)
    
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
            self.model = torch.load(args.model_name+'.pkl')
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
            
            ### model ###
            if args.load_model:
                self.model = torch.load(args.model_name+'.pkl')
                self.optimizer = optimizer_sel(args.model_name,
                                               self.model.parameters(),
                                               args.optim,
                                               args.lr,
                                               load=True,
                                               )
                
            else:
                self.model = nn.Sequential(nn.Linear(80*80*1, 256),
                                           nn.ReLU(),
                                           nn.Linear(256,1),
                                           nn.Sigmoid(),
                                           )
                self.optimizer = ooptimizer_sel(None,
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
        rwd_srd   = 1
        rwd_eps   = 1e-10
        
        loss = 0
        best_result = -21
        latest_rwd  = []
        
        if self.ppo:
            pass
        else:
            for episode in range(self.episode):
                
                ### initialize game ###
                
                o = self.env.reset()                    #reset game
                self.last_frame = prepro(o)             #initialize first frame
                action = self.env.action_space.sample() #random initial action
                o, _, _, _ = self.env.step(action)      #first observation
                
                while True:
                    
                    o = prepro(o)
                    residual_state = prepro(self.last_frame)
                    self.last_frame = o
                    
                    action, p = self.make_action(residual_state, test=False)
                    o, rwd, done, _ = self.env.step(action)
                    
                    prob_list.append(p)
                    act_list.append(action)
                    
                    
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
        
        else:
            
        
        return self.env.get_random_action()

