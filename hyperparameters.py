import numpy as np

class hparams:

    def __init__(self):
        #Neural Net hyperparameters
        self.alpha = 0.0005
        self.beta = 0.0005
        self.gamma = 0.99
        self.max_size = 100000
        self.tau = 0.001
        self.layer1_size = 256
        self.layer2_size = 256
        self.batch_size = 64
        self.reward_scale = 7.0
        #Learning Loop paramaters
        self.faff_max = 500 #maximum time spent not improving
        self.running_mean = 100 #number of terms to be averaged over in the running mean
        self.verbose = 'e' #How much the code should print:
                           #e - print everything
                           #o - only after a reinitialisation
                           #n - no output
        #Automation Run Parameters
        self.window_rate = 0.1 #window decrease rate (between 1 and 0)
        self.pc_max = 5 #max number of reinitialisations before window decrease
        self.max_window_exp = 15 #maximum number of window changes
        #Environment Parameters
        self.reward_reset = False #start with fresh reward?
        self.guessing_run_list = np.array([0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0], dtype=np.bool)
                                           #Toggle 1 or 0 for which paramaters to
                                           #start with previous result
        self.env_shape = 100 #number of z points
        self.action_space_N = 14 # number of parameters or twice the number of operators
        self.hh = 0.05 #external weight/2
        self.shifts = np.array([0.0, 1.5, 1.5, 0.5, 2.5, 1.5, 2.5,
                                0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) #minimum values
        self.guess_sizes = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) #starting windows
        self.neutral_list = [] #which parameters can be positive OR negative
        self.positive_list = list(range(int(self.action_space_N/2))) + [7, 8, 9, 12]
                               #which parameters are positive
        self.negative_list = [10, 11, 13] #which parameters can be negative
        self.block_type = ['s1', 's2', 't1', 't2', 't2', 't2', 't2']
                                       #type s1: g_a (scalar, s channel),
                                       #type s2: g_a_symm (spin > 0, s channel),
                                       #type t1: g_b (scalar, t channel),
                                       #type t2: g_b_symm (spin > 0, t channel)
        self.spin_list = [0, 2, 0, 1, 1, 2, 3] #spin partition
        self.zre = [0.696, 0.794, 0.892, 0.402, 0.402, 0.402, 0.598, 0.402,
                    0.206, 0.696, 0.304, 0.5, 0.304, 0.304, 0.598, 0.402,
                    0.598, 0.5, 0.794, 0.5, 0.696, 0.794, 0.402, 0.402, 0.794,
                    0.598, 0.304, 0.402, 0.206, 0.794, 0.892, 0.696, 0.598,
                    0.696, 0.402, 0.598, 0.206, 0.794, 0.794, 0.5, 0.5, 0.108,
                    0.206, 0.5, 0.696, 0.402, 0.304, 0.696, 0.5, 0.5, 0.5,
                    0.696, 0.5, 0.206, 0.402, 0.598, 0.304, 0.598, 0.402, 0.696,
                    0.696, 0.794, 0.794, 0.402, 0.696, 0.304, 0.598, 0.304,
                    0.304, 0.206, 0.696, 0.5, 0.5, 0.598, 0.402, 0.696, 0.598,
                    0.402, 0.304, 0.402, 0.892, 0.108, 0.696, 0.206, 0.402,
                    0.696, 0.206, 0.402, 0.402, 0.5, 0.402, 0.892, 0.598, 0.794,
                    0.5, 0.206, 0.206, 0.598, 0.304, 0.5]
        self.zim = [0.402, 0.402, 0.01, 0.402, 0.402, -0.402, 0.598, -0.206,
                    0.108, 0.5, -0.206, 0.5, 0.402, -0.304, -0.598, 0.5, -0.304,
                    -0.304, -0.108, 0.206, -0.01, -0.304, 0.108, 0.598, 0.304,
                    0.598, 0.108, 0.206, -0.108, 0.108, 0.108, 0.01, -0.206,
                    -0.206, 0.304, 0.01, -0.402, -0.304, 0.402, -0.01, 0.598,
                    -0.108, 0.402, 0.108, 0.206, -0.108, -0.206, 0.01, 0.108,
                    -0.598, -0.206, -0.01, 0.01, 0.108, -0.598, -0.108, 0.5,
                    0.108, -0.108, -0.108, 0.108, -0.206, -0.01, -0.01, 0.206,
                    0.108, -0.5, -0.304, -0.402, -0.01, 0.304, -0.206, 0.206,
                    -0.206, -0.304, -0.5, -0.402, 0.01, -0.304, -0.206, 0.01,
                    -0.108, -0.108, 0.402, -0.108, 0.108, 0.206, -0.304, 0.598,
                    -0.598, 0.01, -0.01, 0.01, 0.01, 0.402, -0.01, 0.402, 0.598,
                    0.5, -0.598]
