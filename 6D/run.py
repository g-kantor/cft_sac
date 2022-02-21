from learning_loop import learning
from hyperparameters import hparams
import numpy as np
import time

if __name__ == '__main__':
    #---Instantiating the relevant classes---
    hp = hparams()

    #---Defining paramaters to be used---
    window_scale_exp = 0
    max_window_exp = hp.max_window_exp
    rate = hp.window_rate
    pc = 0
    best_accuracy = 0.0

    #---Defining a file path to be used later in the code---
    if hp.create_outp_file:
        if hp.file_name != '':
            file_path = 'outputs/' + str(hp.file_name) + '.txt'
        else:
            file_path = 'outputs/' + 'current_result_' + str(time.time()).split('.')[0] + '.txt'
    else:
        file_path = ''

    #---Instantiating learning class (has to be done after defining file path)---
    lrn = learning(hp.guessing_run_list, file_path)
    lrn.first_run = True
    lrn.read_files()

    #---Looping until a certain window size is reached---
    while window_scale_exp < max_window_exp:
        lrn.best_accuracy = best_accuracy
        lrn.loop(window_scale_exp, rate)
        best_accuracy = lrn.best_accuracy

        #---Tallying the number of times faff_max was reached, then at pc_max, windows are decreased---
        if not lrn.productivity_counter:
            pc += 1
        if pc == hp.pc_max:
            window_scale_exp += 1
            pc = 0
            hp.guessing_run_list = np.zeros(hp.action_space_N, dtype=np.bool)

        #---After finishing a learning loop, the class is reinstantiated---
        del lrn
        lrn = learning(hp.guessing_run_list, file_path)
        lrn.read_files()
