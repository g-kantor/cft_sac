import numpy as np


class hparams:

    def __init__(self):
        # Neural Net hyperparameters
        self.alpha = 0.0005
        self.beta = 0.0005
        self.gamma = 0.99
        self.max_size = 100000
        self.tau = 0.001
        self.layer1_size = 256
        self.layer2_size = 256
        self.batch_size = 64
        self.reward_scale = 7.0
        # Learning Loop paramaters
        self.faff_max = 500  # maximum time spent not improving
        self.running_mean = 100  # number of terms to be averaged over in the running mean
        self.verbose = 'e'  # How much the code should print:
        # e - print everything
        # o - only after a reinitialisation
        # n - no output
        # Automation Run Parameters
        self.window_rate = 0.1  # window decrease rate (between 1 and 0)
        self.pc_max = 5  # max number of reinitialisations before window decrease
        self.max_window_exp = 15  # maximum number of window changes
        # Environment Parameters
        self.reward_reset = True  # start with fresh reward?
        # self.guessing_run_list = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #                                    0, 0, 0, 0, 0, 0], dtype=np.bool)
        # Toggle 1 or 0 for which paramaters to
        # start with previous result
        self.inv_c_charge = 1 / 25  # This is the inverse central charge (we can set it to 0 nicely i.e. infinite c)
        self.maxl = 4  # Spin cutoff. This MUST be an even number
        self.num_of_spins = int(0.5 * self.maxl + 1)
        self.num_of_opecoeffs = 2 * self.num_of_spins  # This is the number of unknown OPE coefficients
        self.num_of_deltas = 2 * self.num_of_spins  # This is the number of known and unknown conformal dimensions
        self.env_shape = 40  # This is number of z points
        self.action_space_N = self.num_of_opecoeffs + self.num_of_deltas  # This is the total number of parameters

        self.shifts_opecoeffs = np.full(self.num_of_opecoeffs, 0.2, dtype=np.float32)
        self.shifts_deltas = np.array([0, 0, 0, 6, 8, 10])
        self.shifts = np.concatenate((self.shifts_opecoeffs, self.shifts_deltas))

        self.guessing_run_list_opes = np.full(self.num_of_opecoeffs, 1, dtype=np.bool)
        self.guessing_run_list_deltas = np.array([0, 0, 0, 1, 1, 1])
        self.guessing_run_list = np.concatenate((self.guessing_run_list_opes, self.guessing_run_list_deltas))

        self.guess_sizes_opes = np.full(self.num_of_opecoeffs, 1, dtype=np.float32)
        self.guess_sizes_deltas = np.array([0, 0, 0, 1, 1, 1])
        self.guess_sizes = np.concatenate((self.guess_sizes_opes, self.guess_sizes_deltas))

        self.neutral_list = []  # which parameters can be positive OR negative
        self.positive_list = list(range(int(self.action_space_N)))  # list(range(int(self.action_space_N)))  # which parameters are positive
        self.negative_list = []

        self.block_type = ['ope D[0,4]', 'ope B[0,2]', 'ope B[0,2]', 'ope L[0,0]', 'ope L[0,0]', 'ope L[0,0]',
                           'delta D[0,4]', 'delta B[0,2]', 'delta B[0,2]', 'delta L[0,0]', 'delta L[0,0]', 'delta L[0,0]']
        # self.block_type = ['D[0,4]', 'B[0,2]', 'B[0,2]', 'L[0,0]', 'L[0,0]', 'L[0,0]', 'L[0,0]']
        # type s1: g_a (scalar, s channel),
        # type s2: g_a_symm (spin > 1, s channel),
        # type t1: g_b (scalar, t channel),
        # type t2: g_b_symm (spin > 1, t channel)
        self.spin_list = [0, 2, 4, 0, 2, 4]  # spin partition
        # self.zre = [0.28174921068496483, 0.37171247358548106,
        #             0.8192128071960839, 0.4898011929480619,
        #             0.36433333577284366, 0.5868862180746248,
        #             0.1788536137559479, 0.7875658375070528,
        #             0.3982651859962612, 0.59147793957184806,
        #             0.1828924058811349, 0.5120705311439774, 0.7717284619883115,
        #             0.8938241431801536, 0.3614054863635266, 0.3969950442593259,
        #             0.7853354521175738, 0.8478307341398701,
        #             0.5960905532156612, 0.6819515049295498,
        #             0.33445072118149716, 0.27428023071122043,
        #             0.4397354432717353, 0.66683935212647291,
        #             0.15849022656792844, 0.764473758219389, 0.5683473609003777,
        #             0.6644931312772897, 0.427820993289316, 0.6137587608379179,
        #             0.9144842118149716, 0.90429023661128043,
        #             0.5597351232717853, 0.2553935272647291,
        #             0.44849032652792344, 0.206473458229389, 0.8883463619403777,
        #             0.2222231416772897, 0.627224996281316, 0.3192587686379179]
        # self.zim = [-0.1196286082458976, 0.5676688331652797,
        #             -0.2703465796624232, 0.4855793121800304, 0.4563978685566148,
        #             0.24983554459868318, 0.3601740000968346,
        #             -0.31502423428972365, 0.5598326981357635,
        #             0.7163000253660893, -0.3538921464788704,
        #             -0.11488450883731992, 0.41681982170391996,
        #             -0.2744724930409729, 0.7024297082801009, 0.1296675719681288,
        #             -0.6039594174704517, 0.1879761887390891, 0.580773414100678,
        #             -0.27357637156350245, -0.6450679710159976,
        #             0.3235023806135529, 0.6920486164050496, 0.7390133873283762,
        #             0.3234845941947617, 0.1482965786801569, -0.5320906447157876,
        #             -0.574914109853286, -0.70040542536128, -0.7528122650004181,
        #             -0.3331639710159976, 0.400123866135529, 0.5920477164057496,
        #             0.4191132873263762, 0.3934849941997617, 0.2282961784431569,
        #             -0.1344906417337876, -0.604912105863286, -0.21040532556198,
        #             -0.4528144450044181]
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
