def add_arguments(parser):
    '''
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('--load_model', type=bool, default=False, help='Load pre-trained model')
    parser.add_argument('--model_name', type=str,  default='PG_1',help='Name of your model')
    parser.add_argument('--optim',      type=str,  default='Adam',help='Optimizer type')
    parser.add_argument('--lr',         type=float,default=0.001, help='Learning rate')
    parser.add_argument('--gamma',      type=float,default=0.99,  help='Discout factor')
    parser.add_argument('--batchsize',  type=int,  default=32,    help='Batch size')
    parser.add_argument('--episide',    type=int,  default=1000,  help='Episodes')
    parser.add_argument('--ppo',        type=bool, default=False, help='Proximal policy optimization')
    
    return parser
