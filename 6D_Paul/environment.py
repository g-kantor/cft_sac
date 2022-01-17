from CrossingIngredients import ConformalBlocks
from hyperparameters import hparams
import numpy as np
from numpy import linalg as LA


class env:
    def __init__(self):
        self.cft = ConformalBlocks()
        hp = hparams()
        self.env_shape = hp.env_shape
        self.action_space_N = hp.action_space_N
        self.npstatus = np.zeros(self.env_shape)
        # self.npstatus_abs = np.zeros(self.env_shape)
        self.nptrack = np.zeros(self.action_space_N, dtype=np.float32)
        self.reward = 0
        self.inv_reward = 100
        # self.abs_error = 0
        self.done = False
        self.max = 1.0
        self.shifts = hp.shifts
        self.guess_sizes = hp.guess_sizes
        self.neutral_list = hp.neutral_list
        self.positive_list = hp.positive_list
        self.negative_list = hp.negative_list
        # self.hh = hp.hh
        self.guessing_run_list = hp.guessing_run_list
        self.zre = hp.zre
        self.zim = hp.zim
        self.block_type = hp.block_type
        self.spin_list = hp.spin_list

        self.inv_c_charge = hp.inv_c_charge
        self.spin_cutoff = hp.maxl
        self.maxk = int(0.5 * hp.maxl)
        self.num_of_spins = hp.num_of_spins
        self.num_of_deltas = hp.num_of_deltas
        self.num_of_opecoeffs = hp.num_of_opecoeffs

        self.ope_coeff_d = self.nptrack[0]
        self.ope_coeffs_b = self.nptrack[1:self.num_of_spins]
        self.ope_coeffs_long = self.nptrack[self.num_of_spins:self.num_of_opecoeffs]
        self.cdims_long = self.nptrack[-self.num_of_spins:]

    def move(self, action, largest, solution):  # remember that there are only
        # real things here
        self.nptrack = np.copy(action)

        for i in range(self.action_space_N):
            if i in self.neutral_list:
                if self.guessing_run_list[i]:
                    self.nptrack[i] = self.shifts[i] + self.guess_sizes[i] * self.nptrack[i]
                else:
                    self.nptrack[i] = self.shifts[i] + self.guess_sizes[i] * self.nptrack[i] + solution[i]
            if i in self.positive_list:
                if self.guessing_run_list[i]:
                    self.nptrack[i] = self.shifts[i] + abs(self.guess_sizes[i] * self.nptrack[i])
                else:
                    self.nptrack[i] = self.shifts[i] + abs(self.guess_sizes[i] * self.nptrack[i] + solution[i])

            if i in self.negative_list:
                if self.guessing_run_list[i]:
                    self.nptrack[i] = self.shifts[i] - abs(self.guess_sizes[i] * self.nptrack[i])
                else:
                    self.nptrack[i] = self.shifts[i] - abs(self.guess_sizes[i] * self.nptrack[i] + solution[i])

        self.ope_coeff_d = self.nptrack[0]
        self.ope_coeffs_b = self.nptrack[1:self.num_of_spins]
        self.ope_coeffs_long = self.nptrack[self.num_of_spins:self.num_of_opecoeffs]
        self.cdims_long = self.nptrack[-self.num_of_spins:]

        for i in range(self.env_shape):
            self.npstatus[i] = self.cft.cons(self.zre[i] + self.zim[i] * 1j, self.zre[i] - self.zim[i] * 1j,
                                             self.inv_c_charge, self.spin_cutoff,
                                             self.ope_coeff_d, self.ope_coeffs_b,
                                             self.ope_coeffs_long, self.cdims_long).real

        self.reward = 1 / LA.norm(self.npstatus)
        self.inv_reward = LA.norm(self.npstatus)
        # self.abs_error = LA.norm(self.npstatus) / (self.npstatus_abs.sum())

        if self.reward > largest:
            self.done = True

    def reset_env(self):
        self.npstatus = np.zeros(self.env_shape, dtype=np.float32)
        self.nptrack = np.zeros(self.action_space_N, dtype=np.float32)
        self.done = False
