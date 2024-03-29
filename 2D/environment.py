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
        self.npstatus_abs = np.zeros(self.env_shape)
        self.nptrack = np.zeros(self.action_space_N, dtype=np.float32)
        self.reward = 0
        self.abs_error = 0
        self.done = False
        self.max = 1.0
        self.shifts = hp.shifts
        self.guess_sizes = hp.guess_sizes
        self.neutral_list = hp.neutral_list
        self.positive_list = hp.positive_list
        self.negative_list = hp.negative_list
        self.hh = hp.hh
        self.guessing_run_list = hp.guessing_run_list
        self.zre = hp.zre
        self.zim = hp.zim
        self.block_type = hp.block_type
        self.spin_list = hp.spin_list
        self.same_spin_hierarchy = hp.same_spin_hierarchy
        self.spin_and_block = [(self.spin_list[i], self.block_type[i])
                               for i in range(self.action_space_N//2)]
        self.spin_and_block_s = [(self.spin_list[i-self.action_space_N//2+1],
                                 self.block_type[i-self.action_space_N//2+1])
                                 for i in range(self.action_space_N//2)]
        self.dup_list = [self.spin_and_block[i] == self.spin_and_block_s[i]
                         for i in range(self.action_space_N//2)]

    def find_duplicates(self):
        flag_current = False
        flag_next = False
        for i in range(self.action_space_N//2):
            flag_current = self.dup_list[i]
            flag_next_tmp = False

            if flag_next and not flag_current:
                self.nptrack[i] = np.clip(self.nptrack[i], a_min=self.nptrack[i-1],
                                          a_max=None)
            if flag_current and not flag_next:
                self.nptrack[i] = np.clip(self.nptrack[i], a_min=None,
                                          a_max=self.nptrack[i+1])
                flag_next_tmp = True

            if flag_current and flag_next:
                if self.nptrack[i - 1] > self.nptrack[i + 1]:
                    self.nptrack[i] = np.clip(self.nptrack[i], a_min=self.nptrack[i-1],
                                              a_max=None)
                else:
                    self.nptrack[i] = np.clip(self.nptrack[i], a_min=self.nptrack[i-1],
                                              a_max=self.nptrack[i+1])
                flag_next_tmp = True

            flag_next = flag_next_tmp

    def move(self, action, largest, solution): #remember that there are only
                                               #real things here
        self.nptrack = np.copy(action)

        for i in range(self.action_space_N):
            if i in self.neutral_list:
                if self.guessing_run_list[i]:
                    self.nptrack[i] = self.shifts[i] + self.guess_sizes[i] * \
                                      self.nptrack[i]
                else:
                    self.nptrack[i] = self.shifts[i] + self.guess_sizes[i] * \
                                      self.nptrack[i] + solution[i]
            if i in self.positive_list:
                if self.guessing_run_list[i]:
                    self.nptrack[i] = self.shifts[i] + abs(self.guess_sizes[i] \
                                      * self.nptrack[i])
                else:
                    self.nptrack[i] = self.shifts[i] + abs(self.guess_sizes[i] \
                                      * self.nptrack[i] + solution[i])

            if i in self.negative_list:
                if self.guessing_run_list[i]:
                    self.nptrack[i] = self.shifts[i] - abs(self.guess_sizes[i] \
                                      * self.nptrack[i])
                else:
                    self.nptrack[i] = self.shifts[i] - abs(self.guess_sizes[i] \
                                      * self.nptrack[i] + solution[i])

        if self.same_spin_hierarchy:
            self.find_duplicates()

        for i in range(self.env_shape):
            [self.npstatus[i], self.npstatus_abs[i]] = self.cft.cons(self.zre[i] + self.zim[i]*1j,
                                                                     self.zre[i] - self.zim[i]*1j,
                                                                     self.nptrack[:int(self.action_space_N/2)],
                                                                     self.nptrack[int(self.action_space_N/2):],
                                                                     self.hh, self.block_type,
                                                                     self.spin_list)

        self.reward = 1 / LA.norm(self.npstatus)
        self.abs_error = LA.norm(self.npstatus)/(self.npstatus_abs.sum())

        if self.reward > largest:
            self.done = True

    def reset_env(self):
        self.npstatus = np.zeros(self.env_shape, dtype=np.float32)
        self.nptrack = np.zeros(self.action_space_N, dtype=np.float32)
        self.done = False
