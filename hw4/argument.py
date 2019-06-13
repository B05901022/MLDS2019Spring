def add_arguments(parser):
    '''
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('--load_model', type=bool, default=True,    help='Load pre-trained model')
    parser.add_argument('--model_name', type=str,  default='PG_CV3',help='Name of your model')
    parser.add_argument('--optim',      type=str,  default='Adam',  help='Optimizer type')
    parser.add_argument('--lr',         type=float,default=0.0001,  help='Learning rate')
    parser.add_argument('--gamma',      type=float,default=0.99,    help='Discout factor')
    parser.add_argument('--batchsize',  type=int,  default=1,       help='Batch size')
    parser.add_argument('--episode',    type=int,  default=10000,   help='Episodes')
    parser.add_argument('--ppo',        type=bool, default=False,   help='Proximal policy optimization')
    parser.add_argument('--vanilla_pg', type=bool, default=False,   help='If use vanilla policy gradient')
    parser.add_argument('--test_episode',type=int, default=1600,    help='Testing/Retraining episode')

    parser.add_argument('--dqn_model_name',     type=str,  default='DQN_CV1',help='Name of your model')
    parser.add_argument('--dqn_test_episode',   type=int,  default=1000,     help='Testing/Loading on which episode')
    parser.add_argument('--dqn_episode',        type=int,  default=10000,    help='Episodes')
    parser.add_argument('--dqn_batchsize',      type=int,  default=32,       help='Batch size')
    parser.add_argument('--dqn_capacity',       type=int,  default=10000,    help='How big is the buffer')
    parser.add_argument('--dqn_eps_start',      type=int,  default=200,      help='Episilon greedy start')
    parser.add_argument('--dqn_eps_decay',      type=int,  default=500,      help='Episilon greedy decay')
    parser.add_argument('--dqn_eps_end',        type=int,  default=1500,     help='Episilon greedy end')
    parser.add_argument('--dqn_current_update', type=int,  default=4,        help='Current update step')
    parser.add_argument('--dqn_target_update',  type=int,  default=1000,     help='Target update step')
    parser.add_argument('--dqn_train_method',   type=str,  default='Epsilon',help='Must be \'Epsilon\' or \'Boltzmann\'')
    parser.add_argument('--dqn_optim',          type=str,  default='Adam',   help='Currently supporting RMSprop, Adam, SGD')
    parser.add_argument('--dqn_lr',             type=float,default=1.5e-4,   help='Learning rate')

    return parser
