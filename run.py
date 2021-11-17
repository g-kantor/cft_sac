from learning_loop import learning
from hyperparameters import hparams
import matplotlib.pyplot as plt

if __name__ == '__main__':
    lrn = learning()
    hp = hparams()
    window_scale_exp = 0
    max_window_exp = hp.max_window_exp
    rate = hp.window_rate
    pc = 0
    running_data = []

    while window_scale_exp < max_window_exp:
        lrn.loop(window_scale_exp, rate)
        running_data = running_data + lrn.avg_rewards
        if not lrn.productivity_counter:
            pc += 1
        if pc == hp.pc_max:
            window_scale_exp += 1
            pc = 0

        del lrn
        lrn = learning()

    plt.plot(running_data)
    plt.show()
