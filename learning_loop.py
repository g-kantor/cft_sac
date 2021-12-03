import numpy as np
from sac_agent import Agent
from environment import env
from hyperparameters import hparams
import matplotlib.pyplot as plt
from numpy import linalg as LA

class learning:
    def __init__(self, g_run):
        self.env = env()
        hp = hparams()
        self.agent = Agent(alpha=hp.alpha, beta=hp.beta,
                           input_dims=[self.env.env_shape],
                           env=self.env, gamma=hp.gamma,
                           n_actions=self.env.action_space_N,
                           max_size=hp.max_size, tau=hp.tau,
                           layer1_size=hp.layer1_size,
                           layer2_size=hp.layer2_size,
                           batch_size=hp.batch_size,
                           reward_scale=hp.reward_scale)
        self.load_checkpoint = False
        self.done = False
        self.fdone = False
        self.observation = self.env.npstatus
        self.env.reset_env()
        self.j = 0
        self.faff = 0
        self.faff_max = hp.faff_max
        self.running_mean = hp.running_mean
        self.reward_reset = hp.reward_reset
        f = open('current_result.txt', 'r')
        tmp1 = f.readlines()
        tmp2 = []
        f.close()
        for i in range(len(tmp1) - 1):
            if i==0:
                tmp2.append(float(tmp1[i].split('=')[1]))
            else:
                if i==(int(len(self.env.nptrack)/2) + 1):
                    pass
                else:
                    tmp2.append(float(tmp1[i].split('->')[1]))
        self.rewards = [tmp2[0]]
        self.solution = np.array(tmp2[1:]) - self.env.shifts
        self.strsol = []
        self.avg_rewards = []
        self.best_rewards = []
        self.best_accuracy = 0.0
        self.parameter_data_sets = []
        for i in range(hp.action_space_N):
            self.parameter_data_sets.append([])
        self.env.guessing_run_list = g_run
        self.verbose = hp.verbose
        self.first_run = False
        self.block_type = hp.block_type
        self.spin_list = hp.spin_list

    def loop(self, iteration, rate):
        self.productivity_counter = False
        self.env.guess_sizes = rate**iteration * self.env.guess_sizes
        if self.first_run:
            if self.reward_reset:
                self.rewards = [0.0]

        while not self.fdone:
            self.j += 1
            self.action = self.agent.choose_action(self.observation)
            self.env.move(self.action, max(self.rewards), self.solution)
            self.observation_ = self.env.npstatus
            self.reward = self.env.reward
            self.agent.remember(self.observation, self.action, self.reward,
                                self.observation_, self.done)
            self.agent.learn()
            self.observation = self.observation_
            self.rewards.append(self.reward)
            self.avg_rewards.append(np.mean(self.rewards[-self.running_mean:]))
            for i in range(self.env.action_space_N):
                self.parameter_data_sets[i].append(self.env.nptrack[i])

            if self.env.done:
                for i in range(len(self.env.nptrack)):
                    self.strsol.append('(s=' + str(self.spin_list[i%(int(len(self.env.nptrack)/2))]) \
                                        + ', ch=' + self.block_type[i%(int(len(self.env.nptrack)/2))] \
                                        + ')' + ' -> ' + str(self.env.nptrack[i]) + '\n')
                    if i == (int(len(self.env.nptrack)/2) - 1):
                        self.strsol.append('-------------------------\n')
                self.strsol.append('acc = ' + str(self.env.abs_error))
                self.solution = np.copy(self.env.nptrack - self.env.shifts)
                file = open('current_result.txt', 'w')
                file.write('reward = ' + str(self.env.reward) + '\n') #FIRST LINE IS ACCURACY
                file.writelines(self.strsol)
                file.close()
                self.strsol = []
                self.best_rewards.append(self.reward)
                self.best_accuracy = self.env.abs_error
                self.env.reset_env()
                self.faff = 0
                self.productivity_counter = True
            else:
                self.faff += 1

            if self.faff==self.faff_max:
                self.fdone = True

            if self.verbose == 'e':
                print(self.solution + self.env.shifts)
                print('step %.1f'% self.j, 'avg reward %.10f' % \
                      np.mean(self.rewards[-25:]), 'current reward %.10f' % \
                      self.reward, 'max reward %.10f' % max(self.rewards),
                      'max accuracy %.10f' % self.best_accuracy,
                      'faff %.1f' % self.faff)
            if self.verbose == 'o':
                if self.fdone:
                    print(self.solution + self.env.shifts)
                    print('step %.1f'% self.j, 'avg reward %.10f' % \
                          np.mean(self.rewards[-25:]), 'current reward %.10f' % \
                          self.reward, 'max reward %.10f' % max(self.rewards),
                          'max accuracy %.10f' % self.best_accuracy,
                          'faff %.1f' % self.faff)
