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
    parser.add_argument('--ppo',        type=bool, default=False,   help='Proximal policy optimization')
    parser.add_argument('--vanilla_pg', type=bool, default=False,   help='If use vanilla policy gradient')
    parser.add_argument('--test_episode',type=int, default=1600,    help='Testing/Retraining episode')

    parser.add_argument('--dqn_model_name',     '-dmn', type=str,  default='DQN_xxD',help='Name of your model')
    parser.add_argument('--dqn_load_model',     '-dlm', type=bool, default=False,    help='Load pre-trained model')
    parser.add_argument('--dqn_dueling',        '-ddu', type=bool, default=False,    help='If using Dueling DQN')
    parser.add_argument('--dqn_noisy',          '-dn',  type=bool, default=False,    help='If using Noisy DQN')
    parser.add_argument('--dqn_ddqn',           '-ddo', type=bool, default=False,    help='If using Double DQN')
    parser.add_argument('--dqn_test_episode',   '-dte', type=int,  default=20000,    help='Testing/Loading on which episode')
    parser.add_argument('--dqn_episode',        '-de',  type=int,  default=100000,   help='Episodes')
    parser.add_argument('--dqn_batchsize',      '-db',  type=int,  default=32,       help='Batch size')
    parser.add_argument('--dqn_capacity',       '-dc',  type=int,  default=10000,    help='How big is the buffer')
    parser.add_argument('--dqn_train_method',   '-dtm', type=str,  default='Epsilon',help='Must be \'Epsilon\' or \'Boltzmann\'')
    parser.add_argument('--dqn_eps_start',      '-des', type=float,default=1.0,      help='Episilon greedy start point')
    parser.add_argument('--dqn_eps_end',        '-dee', type=float,default=0.025,    help='Episilon greedy end point')
    parser.add_argument('--dqn_eps_decay',      '-ded', type=int,  default=1e-6,     help='Episilon decay per step')
    parser.add_argument('--dqn_policy_update',  '-dpu', type=int,  default=4,        help='Policy update step')
    parser.add_argument('--dqn_target_update',  '-dtu', type=int,  default=1000,     help='Target update step')
    parser.add_argument('--dqn_optim',          '-do',  type=str,  default='Adam',   help='Currently supporting RMSprop, Adam, SGD')
    parser.add_argument('--dqn_lr',             '-dlr', type=float,default=1.5e-4,   help='Learning rate')
    parser.add_argument('--dqn_gamma',          '-dg',  type=float,default=0.99,     help='Discount factor')
    parser.add_argument('--test_episode',    type=int,  default=10000,   help='Eest episode')
    parser.add_argument('--dqn_batchsize',  type=int,  default=64,       help='Batch size')
    parser.add_argument('--dqn_episode',type=int, default=1600,    help='Testing/Retraining episode')
    

