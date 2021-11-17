from learning_loop import learning
from hyperparameters import hparams

if __name__ == '__main__':
    lrn = learning()
    hp = hparams()
    window_scale_exp = 0
    rate = hp.window_rate
    pc = 0

    while True:
        lrn.loop(window_scale_exp, rate)
        if not lrn.productivity_counter:
            pc += 1
        if pc == hp.pc_max:
            window_scale_exp += 1
            pc = 0

        del lrn
        lrn = learning()
