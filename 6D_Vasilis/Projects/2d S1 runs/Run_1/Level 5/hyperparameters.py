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
        self.verbose = 'o' #How much the code should print:
                           #e - print everything
                           #o - only after a reinitialisation
                           #n - no output

        #Automation Run Parameters
        self.window_rate = 0.1 #window decrease rate (between 1 and 0)
        self.pc_max = 5 #max number of reinitialisations before window decrease
        self.max_window_exp = 3 #maximum number of window changes

        #Environment Parameters
        self.guessing_run = True #start with a guessing_run?
        self.guessing_run_list = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.bool)
        self.env_shape = 1 #previous number of z-points, now z-integrated crossing equations squared
        self.action_space_N = 18 # number of parameters or twice the number of operators
        self.hh = 0.05 #external weight/2
        self.shifts = np.array([0.0, 3.0, 1.8, 1.8, 3.0, 0.8, 1.8, 3.0, 2.8,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) #minimum values
        self.guess_sizes = np.array([0.2, 0.2, 1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                                     0.01, 0.01, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]) #starting windows
        self.negative_list = [14, 17] #which parameters can be negative
        self.block_type = [1, 1, 2, 3, 3, 4, 4, 4, 4]
                                       #type 1: g_a (scalar, s channel),
                                       #type 2: g_a_symm (spin 1+, s channel),
                                       #type 3: g_b (scalar, t channel),
                                       #type 4: g_b_symm (spin 1+, t channel)
        self.spin_list = [0, 0, 2, 0, 0, 1, 2, 2, 3] #spin partition
        self.angle_lower = - 0.8    # z-integration: angle lower limit
        self.angle_upper = 0.8      # z-integration: angle upper limit
        self.rho_lower = 0.08       # z-integration: radius lower limit
        self.rho_upper = 0.3        # z-integration: radius upper limit


        # discrete z-sampling depository
        self.zre = [0.28174921068496483, 0.37171247358548106,
                    0.8192128071960839, 0.4898011929480619,
                    0.36433333577284366, 0.5868862180746248,
                    0.1788536137559479, 0.7875658375070528,
                    0.3982651859962612, 0.59147793957184806,
                    0.1828924058811349, 0.5120705311439774, 0.7717284619883115,
                    0.8938241431801536, 0.3614054863635266, 0.3969950442593259,
                    0.7853354521175738, 0.8478307341398701,
                    0.5960905532156612, 0.6819515049295498,
                    0.33445072118149716, 0.27428023071122043,
                    0.4397354432717353,  0.66683935212647291,
                    0.15849022656792844, 0.764473758219389, 0.5683473609003777,
                    0.6644931312772897, 0.427820993289316,  0.6137587608379179,
                    0.9144842118149716, 0.90429023661128043,
                    0.5597351232717853, 0.2553935272647291,
                    0.44849032652792344, 0.206473458229389, 0.8883463619403777,
                    0.2222231416772897, 0.627224996281316, 0.3192587686379179]
        self.zim = [-0.1196286082458976, 0.5676688331652797,
                    -0.2703465796624232, 0.4855793121800304, 0.4563978685566148,
                    0.24983554459868318, 0.3601740000968346,
                    -0.31502423428972365, 0.5598326981357635,
                    0.7163000253660893, -0.3538921464788704,
                    -0.11488450883731992, 0.41681982170391996,
                    -0.2744724930409729, 0.7024297082801009, 0.1296675719681288,
                    -0.6039594174704517, 0.1879761887390891, 0.580773414100678,
                    -0.27357637156350245, -0.6450679710159976,
                    0.3235023806135529, 0.6920486164050496, 0.7390133873283762,
                    0.3234845941947617, 0.1482965786801569, -0.5320906447157876,
                    -0.574914109853286, -0.70040542536128, -0.7528122650004181,
                    -0.3331639710159976, 0.400123866135529, 0.5920477164057496,
                    0.4191132873263762, 0.3934849941997617, 0.2282961784431569,
                    -0.1344906417337876, -0.604912105863286, -0.21040532556198,
                    -0.4528144450044181]
