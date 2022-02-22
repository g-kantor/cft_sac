import numpy as np
from data_z_sample import z_data

class hparams:

    def __init__(self):
        #---Neural Net hyperparameters---
        self.alpha = 0.0005
        self.beta = 0.0005
        self.gamma = 0.99
        self.max_size = 100000
        self.tau = 0.001
        self.layer1_size = 256
        self.layer2_size = 256
        self.batch_size = 256
        self.reward_scale = 7.0

        #---Learning Loop paramaters---
        self.faff_max = 500 #maximum time spent not improving
        self.running_mean = 100 #number of terms to be averaged over in the running mean
        self.verbose = 'e' #How much the code should print:
                           #e - print everything
                           #o - only after a reinitialisation
                           #n - no output

        #---Automation Run Parameters---
        self.window_rate = 0.3 #window decrease rate (between 1 and 0)
        self.pc_max = 5 #max number of reinitialisations before window decrease
        self.max_window_exp = 15 #maximum number of window changes
        self.create_outp_file = True
        self.file_name = ''

        #---Environment Parameters---
        self.reward_reset = True #start with fresh reward?
        self.same_spin_hierarchy = True #same operators with the same spin should be ordered
        self.inv_c_charge = 0.0  # This is the inverse central charge (we can set it to 0 nicely i.e. infinite c)
        self.hh = 0.05 #external weight/2
        self.guessing_run_list_deltas = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.bool)
                                                                        #start from scratch?
        self.guessing_run_list_opes = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.bool)
                                                                        #start from scratch?
        self.guess_sizes_deltas = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
                                                                        #initial search window size for deltas
        self.guess_sizes_opes = np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0])
                                                                        #initial search window size for OPE coeffs
        self.shifts_deltas = np.array([8.0, 9.0, 11.0, 13.0, 15.0, 5.6, 5.6, 5.6, 7.6, 7.6, 9.6])
                                                                        #set minimum values for deltas
        self.shifts_opecoeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                                                                        #set minimum values for OPE coeffs
        self.block_type = ['d4', 'b2', 'b2', 'b2', 'b2', 'l', 'l', 'l', 'l', 'l', 'l']
        self.block_type_printed = {'d4': 'D[0,4]', 'b2': 'B[0,2]', 'l': 'L[0,0]'}
                                                                        # type d4: D[0,4] short multiplets
                                                                        # type b2: B[0,2] short multiplets
                                                                        # type l: L[0,0] long multiplets
                                                                        # the order d4, b2, l MUST be preserved
        self.spin_list_short_d = np.array([0])
        self.spin_list_short_b = np.array([2, 4, 6, 8])
        self.spin_list_long = np.array([0, 0, 0, 2, 2, 4]) #spins HAVE to be given in ascending order
        self.num_of_operators_short_d = 1
        self.num_of_operators_short_b = 4
        self.num_of_operators_long = 6

        #---Non-Adjustable Parameters---
        z_init = z_data()
        self.env_shape = z_init.env_shape #number of z points
        self.ell_max = np.amax(np.concatenate((self.spin_list_short_d, self.spin_list_short_b,
                                               self.spin_list_long), axis=None))
                                                                        # Spin cutoff. This MUST be an even number
        self.multiplet_index = [np.arange(self.num_of_operators_short_d),
                                np.arange(self.num_of_operators_short_d,
                                          (self.num_of_operators_short_b +
                                           self.num_of_operators_short_d)),
                                np.arange((self.num_of_operators_short_b +
                                           self.num_of_operators_short_d),
                                          (self.num_of_operators_short_b +
                                           self.num_of_operators_short_d +
                                           self.num_of_operators_long))]
        self.action_space_N = len(self.guessing_run_list_opes) + \
                              len(self.guessing_run_list_deltas)        # This is the total
                                                                        #number of parameters
        self.shifts = np.concatenate((self.shifts_deltas, self.shifts_opecoeffs))
        self.guessing_run_list = np.concatenate((self.guessing_run_list_deltas,
                                                 self.guessing_run_list_opes))
        self.guess_sizes = np.concatenate((self.guess_sizes_deltas, self.guess_sizes_opes))
        self.spin_list_short = np.concatenate((self.spin_list_short_d, self.spin_list_short_b))
        self.spin_list = np.concatenate((self.spin_list_short_d, self.spin_list_short_b,
                                         self.spin_list_long))
