def add_arguments(parser):
    '''
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('--load_model', type=bool, default=False,    help='Load pre-trained model')
    parser.add_argument('--ppo',        type=bool, default=False,   help='Proximal policy optimization')
    parser.add_argument('--vanilla_pg', type=bool, default=False,   help='If use vanilla policy gradient')
    
    
    parser.add_argument('--dqn_model_name', type=str,  default='DQN_CV1',help='Name of your model')
    parser.add_argument('--dqn_optim',      type=str,  default='Adam',  help='Optimizer type (SGD,Adam,RMSprop)')
    parser.add_argument('--dqn_train_method', type=str, default="Epsilon",   help='Choose Epsilon or Boltzmann')
    parser.add_argument('--model_name', type=str,  default='PG_CV3',help='Name of your model')
    parser.add_argument('--optim',      type=str,  default='Adam',  help='Optimizer type')
    parser.add_argument('--dqn_lr',         type=float,default=0.0001,  help='Learning rate')
    parser.add_argument('--dqn_eps_start',      type=float,default=1,    help='Epsilon start value')
    parser.add_argument('--dqn_eps_decay',      type=float,default=0.0001,    help='Epsilon decay value')
    parser.add_argument('--dqn_eps_end',      type=float,default=0.025,    help='Epsilon minimum value')
    parser.add_argument('--lr',         type=float,default=0.0001,  help='Learning rate')
    parser.add_argument('--gamma',      type=float,default=0.99,    help='Discout factor')
    parser.add_argument('--dqn_episode',    type=int,  default=10000,   help='Episodes')
    parser.add_argument('--dqn_capacity',    type=int,  default=10000,   help='How many samples can the replay buffer keep?')
    parser.add_argument('--dqn_target_update',    type=int,  default=10000,   help='How many times can the policy model update after the target model updates?')
    parser.add_argument('--dqn_sample_frequency',    type=int,  default=4,   help='Sample frequency')
    parser.add_argument('--batchsize',  type=int,  default=1,       help='Batch size')
    parser.add_argument('--episode',    type=int,  default=10000,   help='Episodes')
    parser.add_argument('--test_episode',    type=int,  default=10000,   help='Eest episode')
    parser.add_argument('--dqn_batchsize',  type=int,  default=64,       help='Batch size')
    parser.add_argument('--dqn_episode',type=int, default=1600,    help='Testing/Retraining episode')
    
    
    
    
    
    return parser
