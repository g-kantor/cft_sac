from blocks import conf_blocks
from hyperparameters import hparams
import numpy as np
from numpy import linalg as LA

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
        self.negative_list = hp.negative_list
        self.hh = hp.hh
        self.guessing_run = hp.guessing_run
        self.zre = hp.zre
        self.zim = hp.zim
        self.block_type = hp.block_type
        self.spin_list = hp.spin_list

    def move(self, action, largest, solution): #remember that there are only
                                               #real things here
        self.nptrack = np.copy(action)

        for i in range(self.action_space_N):
            if i in self.negative_list: #add here all the OPE squares which are
                                        #negative
                if self.guessing_run:
                    self.nptrack[i] = self.shifts[i] + self.guess_sizes[i] * \
                                      self.nptrack[i]
                else:
                    self.nptrack[i] = self.shifts[i] + self.guess_sizes[i] * \
                                      self.nptrack[i] + solution[i]
            else:
                if self.guessing_run:
                    self.nptrack[i] = self.shifts[i] + abs(self.guess_sizes[i] \
                                      * self.nptrack[i])
                else:
                    self.nptrack[i] = self.shifts[i] + abs(self.guess_sizes[i] \
                                      * self.nptrack[i] + solution[i])

        for i in range(self.env_shape):
            self.npstatus[i] = self.cft.cons(self.zre[i] + self.zim[i]*1j,
                                             self.zre[i] - self.zim[i]*1j,
                                             self.nptrack[:int(self.action_space_N/2)],
                                             self.nptrack[int(self.action_space_N/2):],
                                             self.hh, self.block_type,
                                             self.spin_list).real

        self.reward = 1 / LA.norm(self.npstatus)

        if self.reward > largest:
            self.done = True

    def reset_env(self):
        self.npstatus = np.zeros(self.env_shape, dtype=np.float32)
        self.nptrack = np.zeros(self.action_space_N, dtype=np.float32)
        self.reward = 0
        self.done = False
