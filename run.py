from learning_loop import learning

if __name__ == '__main__':
    lrn = learning()
    window_scale_exp = 0
    rate = 0.1 #how quickly the window should decrease, number between 0 and 1,
               #1 being slowest.
    pc = 0

    while True:
        lrn.loop(window_scale_exp, rate)
        if not lrn.productivity_counter:
            pc += 1
        if pc == 5: #how long the iterator should wait before decreasing the
                    #window
            window_scale_exp += 1
            pc = 0

        del lrn
        lrn = learning()
