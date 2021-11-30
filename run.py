from learning_loop import learning
from hyperparameters import hparams
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np

if __name__ == '__main__':
    hp = hparams()
    lrn = learning(g_run=hp.guessing_run_list)
    window_scale_exp = 0
    max_window_exp = hp.max_window_exp
    rate = hp.window_rate
    pc = 0
    '''
    running_data = []
    tmp = []
    plot_data = []
    for i in range(int(hp.action_space_N/2)):
        tmp.append([])
        plot_data.append([])
        running_data.append([])
    '''

    while window_scale_exp < max_window_exp:
        lrn.loop(window_scale_exp, rate)
        #for i in range(int(hp.action_space_N/2)):
        #    running_data[i] = running_data[i] + lrn.parameter_data_sets[i]
        if not lrn.productivity_counter:
            pc += 1
        if pc == hp.pc_max:
            window_scale_exp += 1
            pc = 0
            hp.guessing_run_list = np.zeros(hp.action_space_N, dtype=np.bool)

        del lrn
        lrn = learning(g_run=hp.guessing_run_list)

'''
    figure(figsize=(10, 10), dpi=100)

    for k in range(int(hp.action_space_N/2)):
        for i in running_data[k]:
            tmp[k].append(i)
            plot_data[k].append(np.mean(np.array(tmp[k])[-hp.running_mean:]))

    for i in range(int(hp.action_space_N/2)):
        plt.plot(plot_data[i])
    plt.show()
'''
