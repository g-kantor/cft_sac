import numpy as np
from sac_agent import Agent
from environment import env
from hyperparameters import hparams
from numpy import linalg as LA

class learning:
    def __init__(self, g_run, file_path):
        #---Instantiating the relevant classes---
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

        #---Defining paramaters to be used in the class---
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
        self.strsol = []
        self.avg_rewards = []
        self.best_rewards = []
        self.env.guessing_run_list = g_run
        self.verbose = hp.verbose
        self.first_run = False
        self.block_type = hp.block_type
        self.block_type_printed = hp.block_type_printed
        self.spin_list = hp.spin_list
        self.create_outp_file = hp.create_outp_file
        self.file_path = file_path


    def read_files(self):
        #---Reading previous results from output file---
        if not self.create_outp_file: #no unique output file
            f = open('outputs/current_result.txt', 'r')
            tmp1 = f.readlines()
            tmp2 = []
            f.close()
            for i in range(len(tmp1)):
                if i==0:
                    tmp2.append(float(tmp1[i].split('=')[1]))
                else:
                    if i==(int(len(self.env.nptrack)/2) + 1):
                        pass
                    else:
                        tmp2.append(float(tmp1[i].split('->')[1]))
            self.rewards = [tmp2[0]]
            self.solution = np.array(tmp2[1:]) - self.env.shifts

        else: #if in hyperparameters a new output file has been enabled

            if self.first_run: #if it's the first run of the algorithm
                self.rewards = [0.0]
                self.solution = np.zeros(self.env.action_space_N)

            else: #if the algorithm has already completed a loop
                f = open(self.file_path, 'r')
                tmp1 = f.readlines()
                tmp2 = []
                f.close()
                for i in range(len(tmp1)):
                    if i==0:
                        tmp2.append(float(tmp1[i].split('=')[1]))
                    else:
                        if i==(int(len(self.env.nptrack)/2) + 1):
                            pass
                        else:
                            tmp2.append(float(tmp1[i].split('->')[1]))
                self.rewards = [tmp2[0]]
                self.solution = np.array(tmp2[1:]) - self.env.shifts

    def write_result(self):
        #---Write the current best results to an output file---
        for i in range(len(self.env.nptrack)):

        #---First we create a specific output template---
            self.strsol.append('(spin=' + str(self.spin_list[i%(len(self.env.nptrack)//2)]) \
                                + ', mult=' + self.block_type_printed[
                                self.block_type[i%(len(self.env.nptrack)//2)]] \
                                + ')' + ' -> ' + str(self.env.nptrack[i]) + '\n')
            if i == (int(len(self.env.nptrack)/2) - 1):
                self.strsol.append('-------------------------\n')
        self.solution = np.copy(self.env.nptrack - self.env.shifts)

        #---Determine the correct file to write to---
        if not self.create_outp_file:
            file = open('outputs/current_result.txt', 'w')
        else:
            file = open(self.file_path, 'w')

        #---Finally, write to the file---
        file.write('reward = ' + str(self.env.reward) + '\n') #FIRST LINE IS ACCURACY
        file.writelines(self.strsol)
        file.close()

    def cmd_line_out(self, iteration):
        #---Specify what information should be shown in the terminal---
        if self.verbose == 'e':
            print(self.solution + self.env.shifts)
            print('step %.1f'% self.j, 'avg reward %.10f' % \
                  np.mean(self.rewards[-25:]), 'current reward %.10f' % \
                  self.reward, 'max reward %.10f' % max(self.rewards),
                  'zoom %.d' % iteration,
                  'faff %.1f' % self.faff)
        if self.verbose == 'o':
            if self.fdone:
                print(self.solution + self.env.shifts)
                print('step %.1f'% self.j, 'avg reward %.10f' % \
                      np.mean(self.rewards[-25:]), 'current reward %.10f' % \
                      self.reward, 'max reward %.10f' % max(self.rewards),
                      'zoom %.d' % iteration,
                      'faff %.1f' % self.faff)



    def loop(self, iteration, rate):
        #---Defining some objects before the loop---
        self.productivity_counter = False
        self.env.guess_sizes = rate**iteration * self.env.guess_sizes
        if self.first_run: #relevant for when no unique output file
            if self.reward_reset:
                self.rewards = [0.0]

        while not self.fdone:
            #---Running the learning loop---
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


            if self.env.done: #If the reward is higher than the previous
                #---Write result to output file and reset/update objects---
                self.write_result()
                self.strsol = []
                self.best_rewards.append(self.reward)
                self.env.reset_env()
                self.faff = 0
                self.productivity_counter = True

            else: #If the reward is smaller
                self.faff += 1

            if self.faff==self.faff_max: #If faff limit is reached, abort
                self.fdone = True

            self.cmd_line_out(iteration)
