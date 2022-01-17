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
        self.faff_max = 500 #maximum time spent not improving 500 before
        self.running_mean = 100 #number of terms to be averaged over in the running mean
        self.verbose = 'o' #How much the code should print:
                           #e - print everything
                           #o - only after a reinitialisation
                           #n - no output

        #Automation Run Parameters
        self.window_rate = 0.5 #window decrease rate (between 1 and 0) 0.1 before
        self.pc_max = 5 #max number of reinitialisations before window decrease 5 before
        self.max_window_exp = 20 #maximum number of window changes

        #Environment Parameters
        self.guessing_run = True #start with a guessing_run?
        self.guessing_run_list = np.array([1, 1, 1, 1, 1, 1,
                                           1, 1, 1, 1, 1, 1], dtype=np.bool)
        self.env_shape = 40  #number of z-points
        self.action_space_N = 12  # number of parameters or twice the number of operators
        self.central = 25     # central charge
        self.ell_max_chi = 10    # half of upper cutoff on ell sum for chi
        self.shifts = np.array([9.0, 11.0, 5.6, 5.6, 7.6, 7.6,
                                10.0, 5.0, 0.0, 0.0, 0.0, 0.0]) #minimum values
        self.guess_sizes = np.array([0.0, 0.0, 5.0, 5.0, 5.0, 5.0,
                                     10.0, 10.0, 10.0, 10.0, 10.0, 10.0]) #starting windows
        #self.negative_list = [0, 0, 0, 0, 0, 0] #which parameters are negative: in this context none
        self.block_type = [1, 1, 3, 3, 3, 3]
                                       #type 1: B[0,2] short multiplets
                                       #type 2: D[0,4] short multiplets
                                       #type 3: L[0,0] long multiplets
        self.spin_list = [2, 4, 0, 0, 2, 2]         #spin partition
        self.angle_lower = 0.5      # z-integration: angle lower limit
        self.angle_upper = 0.7      # z-integration: angle upper limit
        self.rho_lower = 0.2       # z-integration: radius lower limit
        self.rho_upper = 0.3        # z-integration: radius upper limit
        #self.angle_lower = 0.01    # z-integration: angle lower limit
        #self.angle_upper = 0.8     # z-integration: angle upper limit
        #self.rho_lower = 0.08      # z-integration: radius lower limit
        #self.rho_upper = 0.3       # z-integration: radius upper limit


        # discrete z-sampling depository
        self.zre = [0.468807640826418, 0.517146862754101, 0.533880739764984,
                    0.453358614936510, 0.478785663077804, 0.545299241427676,
                    0.509526073915755, 0.525685871104942, 0.529967603947159,
                    0.490719474514400, 0.497418844956683, 0.513375687826775,
                    0.513338708802137, 0.470132722513035, 0.450573845646593,
                    0.544736759885518, 0.526597400358350, 0.454734015028994,
                    0.515004621891340, 0.489781950078955, 0.517929964965511,
                    0.462581960721652, 0.531423448384517, 0.537204112413190,
                    0.470485486062178, 0.463814970256187, 0.545219150889574,
                    0.484211076584189, 0.486825883715483, 0.509857851483674,
                    0.482213819916561, 0.493498147430609, 0.477481962040439,
                    0.468851424164388, 0.498786821625030, 0.535178896079752,
                    0.538797472525637, 0.478714309941896, 0.523252997635023,
                    0.512468217980186]
        self.zim = [0.524265297033090, 0.537187906398895, 0.536290603507858,
                    0.456717769527056, 0.530601416663103, 0.520607574262190,
                    0.471808137750576, 0.510959262121600, 0.539551815014013,
                    0.507717187056383, 0.498818027614816, 0.474143275660736,
                    0.455771561469064, 0.481144581971504, 0.520181918456855,
                    0.521104793089804, 0.505713415207269, 0.508474352372741,
                    0.544539365604019, 0.523029641848955, 0.544310693999159,
                    0.547157647548605, 0.546679687104277, 0.454502007761790,
                    0.539390601487434, 0.503218195531775, 0.480655924289362,
                    0.542922414080476, 0.530636351383066, 0.466891830540449,
                    0.518785967996353, 0.525913063707380, 0.454761222979719,
                    0.549035389651293, 0.549607182583849, 0.467336167003651,
                    0.541994984267299, 0.539059381885018, 0.470160265471706,
                    0.494633229182767]