from blocks import conf_blocks
from hyperparameters import hparams
import numpy as np
from numpy import linalg as LA

class env:
    def __init__(self):
        #---Instantiating the relevant classes---
        self.cft = conf_blocks()
        hp = hparams()

        #---Defining paramaters to be used in the class---
        self.env_shape = hp.env_shape
        self.action_space_N = hp.action_space_N
        self.npstatus = np.zeros(self.env_shape)
        self.nptrack = np.zeros(self.action_space_N, dtype=np.float32)
        self.reward = 0
        self.abs_error = 0
        self.done = False
        self.max = 1.0
        self.shifts = hp.shifts
        self.guess_sizes = hp.guess_sizes
        self.guessing_run_list = hp.guessing_run_list
        self.multiplet_index = hp.multiplet_index
        self.block_type = hp.block_type
        self.spin_list = hp.spin_list
        self.dyn_shift = hp.dyn_shift

        self.inho_value = self.cft.inhomo_z_vector()
        self.short_d_hypers = self.cft.short_coeffs_d_multiplet()
        self.short_b_hypers = self.cft.short_coeffs_b_multiplet_array()

        self.same_spin_hierarchy = hp.same_spin_hierarchy
        self.spin_and_block = [(self.spin_list[i], self.block_type[i])
                               for i in range(self.action_space_N//2)]
        self.spin_and_block_s = [(self.spin_list[i-self.action_space_N//2+1],
                                 self.block_type[i-self.action_space_N//2+1])
                                 for i in range(self.action_space_N//2)]
        self.dup_list = [self.spin_and_block[i] == self.spin_and_block_s[i]
                         for i in range(self.action_space_N//2)]

    #---Creating a function to ensure same spin multiplets have increasing weights
    def find_duplicates(self):
        flag_current = False
        flag_next = False
        for i in range(self.action_space_N//2):
            flag_current = self.dup_list[i]
            flag_next_tmp = False


            if flag_next and not flag_current:
                self.nptrack[i] = np.clip(self.nptrack[i], a_min=(self.nptrack[i-1] \
                                          + self.dyn_shift), a_max=None)

            if flag_current and not flag_next:
                #self.nptrack[i] = np.clip(self.nptrack[i], a_min=None,
                #                          a_max=(self.nptrack[i+1]))
                flag_next_tmp = True

            if flag_current and flag_next:
                '''
                if self.nptrack[i - 1] > self.nptrack[i + 1]:
                    self.nptrack[i] = np.clip(self.nptrack[i], a_min=(self.nptrack[i-1] \
                                              + self.dyn_shift), a_max=None)
                else:
                    self.nptrack[i] = np.clip(self.nptrack[i], a_min=(self.nptrack[i-1] \
                                              + self.dyn_shift))
                '''
                self.nptrack[i] = np.clip(self.nptrack[i], a_min=(self.nptrack[i-1] \
                                          + self.dyn_shift), a_max=None)
                flag_next_tmp = True

            flag_next = flag_next_tmp


    def move(self, action, largest, solution):
        #---Getting current set of deltas and OPE coeffs from the nets---
        self.nptrack = np.copy(action)

        #---Rewriting them in the form used by the algorithm---
        self.nptrack = self.shifts + abs(self.guess_sizes * self.nptrack + \
                       (1 - self.guessing_run_list) * solution)

        #---Enforcing the Spin Weight Hierarchy
        if self.same_spin_hierarchy:
            self.find_duplicates()

        #---Creating dictionaries which know the positions of specific multiplets within nptrack---
        delta_dict = {
                      "short_d": self.nptrack[self.multiplet_index[0]],
                      "short_b": self.nptrack[self.multiplet_index[1]],
                      "long": self.nptrack[self.multiplet_index[2]]
                      }
        ope_dict = {
                    "short_d": self.nptrack[self.multiplet_index[0] + self.action_space_N//2],
                    "short_b": self.nptrack[self.multiplet_index[1] + self.action_space_N//2],
                    "long": self.nptrack[self.multiplet_index[2] + self.action_space_N//2]
                    }

        #---Calculating all the different contributions to the constraints---
        short_cons_d_multiplet = ope_dict['short_d'] * self.short_d_hypers
        short_cons_b_multiplet = ope_dict['short_b'].reshape(-1, 1) * self.short_b_hypers
        long_cons = ope_dict['long'].reshape(-1, 1) * \
                    self.cft.long_coeffs_array(delta_dict['long'])

        self.npstatus = self.inho_value + short_cons_d_multiplet + \
                        short_cons_b_multiplet.sum(axis=0) + \
                        long_cons.sum(axis=0)

        #---Defining the reward for the agent as 1/|constraints|---
        self.reward = 1 / LA.norm(self.npstatus)

        #---If current reward is larger than previous largest then trigger a flag---
        if self.reward > largest:
            self.done = True

    def reset_env(self):
        #---Resetting relevant objects---
        self.npstatus = np.zeros(self.env_shape, dtype=np.float32)
        self.nptrack = np.zeros(self.action_space_N, dtype=np.float32)
        self.done = False
