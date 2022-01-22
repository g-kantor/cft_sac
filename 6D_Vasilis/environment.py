from CrossingIngredients import conf_blocks
from hyperparameters import hparams
import numpy as np
from numpy import linalg as LA
#import scipy.special as sc
#from scipy.integrate import dblquad

class env:
    def __init__(self):
        self.cft = conf_blocks()
        hp = hparams()
        self.env_shape = hp.env_shape
        self.action_space_N = hp.action_space_N
        self.npstatus = np.zeros(self.env_shape)
        self.nptrack = np.zeros(self.action_space_N, dtype=np.float32)
        self.reward = 0
        self.done = False
        self.max = 1.0
        self.shifts = hp.shifts
        self.guess_sizes = hp.guess_sizes
        #self.negative_list = hp.negative_list
        self.central = hp.central
        self.ell_max_chi = hp.ell_max_chi
        self.guessing_run_list = hp.guessing_run_list
        self.zre = hp.zre
        self.zim = hp.zim
        self.block_type = hp.block_type
        self.spin_list = hp.spin_list
        self.angle_lower = hp.angle_lower
        self.angle_upper = hp.angle_upper
        self.rho_lower = hp.rho_lower
        self.rho_upper = hp.rho_upper
        self.inho_value = self.cft.inhomo_z_vector()
        self.short_coeffs = self.cft.short_cons_coeffs()
        self.short_cons = np.zeros(self.env_shape)

        #(self.nptrack[int(self.action_space_N/2):], self.block_type, self.spin_list)

    def move(self, action, largest, solution):
        self.nptrack = np.copy(action)
        number_operators = int(self.action_space_N / 2)


        for i in range(number_operators):    #move for the deltas
            if self.block_type[i] == 1:  # B[0,2] short multiplets
                self.nptrack[i] = self.spin_list[i] + 7
            elif self.block_type[i] == 2:  # D[0,4] short multiplets
                self.nptrack[i] = 8
            elif self.block_type[i] == 3:  # L[0,0] long multiplets
                if self.guessing_run_list[i]:
                    self.nptrack[i] = self.shifts[i] + abs(self.guess_sizes[i] * self.nptrack[i])
                else:
                    self.nptrack[i] = self.shifts[i] + abs(self.guess_sizes[i] * self.nptrack[i] + solution[i])

        for i in range(number_operators, self.action_space_N):    #move for the OPE-squared coefficients
            #if i in self.negative_list:
            #    if self.guessing_run_list[i]:
            #        self.nptrack[i] = self.shifts[i] - abs(self.guess_sizes[i] * self.nptrack[i])
            #    else:
            #        self.nptrack[i] = self.shifts[i] - abs(self.guess_sizes[i] * self.nptrack[i] + solution[i])
            #else:
            if self.guessing_run_list[i]:
                self.nptrack[i] = self.shifts[i] + abs(self.guess_sizes[i] * self.nptrack[i])
            else:
                self.nptrack[i] = self.shifts[i] + abs(self.guess_sizes[i] * self.nptrack[i] + solution[i])

        for i in range(self.env_shape):
            temp = 0
            for j in range(int(self.action_space_N / 2)):
                temp += self.short_coeffs[i][j] * self.nptrack[int(self.action_space_N / 2) + j]
            self.short_cons[i] = temp

        for i in range(self.env_shape):
            self.npstatus[i] = (self.inho_value[i] +
                                self.short_cons[i] +
                                self.cft.long_cons(self.zre[i] + self.zim[i]*1j, self.zre[i] - self.zim[i]*1j,
                                                   self.nptrack[:int(self.action_space_N / 2)],
                                                   self.nptrack[int(self.action_space_N / 2):],
                                                   self.block_type, self.spin_list)).real

        #for i in range(self.env_shape):
        #    self.npstatus[i] = dblquad(lambda r, a: self.cft.weight(r, a) *
        #                                (self.cft.cons(self.central, self.ell_max_chi,
        #                                0.5 + r * np.exp(a * 1j),
        #                                0.5 + r * np.exp(- a * 1j),
        #                                self.nptrack[:int(self.action_space_N / 2)],
        #                                self.nptrack[int(self.action_space_N / 2):],
        #                                self.block_type,
        #                                self.spin_list).real) ** 2,
        #                                self.angle_lower, self.angle_upper, lambda r: self.rho_lower,
        #                                lambda r: self.rho_upper)[0]

        #for i in range(self.env_shape):
        #    self.npstatus[i] = dblquad(lambda r, a: self.cft.weight(r, a) *
        #                                (self.cft.cons(self.central, self.ell_max_h, self.ell_max_chi,
        #                                4 * r * np.exp(a * 1j) * (1 + r * np.exp(a * 1j)) ** (-2),
        #                                4 * r * np.exp(- a * 1j) * (1 + r * np.exp(- a * 1j)) ** (-2),
        #                                self.nptrack[:int(self.action_space_N / 2)],
        #                                self.nptrack[int(self.action_space_N / 2):],
        #                                self.block_type,
        #                                self.spin_list).real) ** 2,
        #                                self.angle_lower, self.angle_upper, lambda r: self.rho_lower,
        #                                lambda r: self.rho_upper)[0]

        self.reward = 1 / LA.norm(self.npstatus)

        if self.reward > largest:
            self.done = True

    def reset_env(self):
        self.npstatus = np.zeros(self.env_shape, dtype=np.float32)
        self.nptrack = np.zeros(self.action_space_N, dtype=np.float32)
        self.reward = 0
        self.done = False
