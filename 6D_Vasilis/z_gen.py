import scipy.special as sc
import mpmath as mp
import numpy as np
import random
from CrossingIngredients import conf_blocks

n_x = 15
n_y = 15

reals_q1 = np.linspace(0.51, 0.73, n_x)
reals_q2 = np.linspace(0.01, 0.99, n_x)
reals_q3 = np.linspace(-0.01, -0.99, n_x)
reals_q4 = np.linspace(-0.01, -0.99, n_x)
#reals = np.concatenate((reals_q1, reals_q2, reals_q3, reals_q4))
reals = reals_q1
imag_q1 = np.linspace(0.11, 0.69, n_y)
imag_q2 = np.linspace(-0.01, -0.99, n_y)
imag_q3 = np.linspace(-0.01, -0.99, n_y)
imag_q4 = np.linspace(0.01, 0.99, n_y)
#imag = np.concatenate((imag_q1, imag_q2, imag_q3, imag_q4))
imag = imag_q1
delta = 9.5
spin = 2
x = np.zeros(16*n_x*n_y)
y = np.zeros(16*n_x*n_y)
zre = []
zim = []
cntr = 0

blocks = conf_blocks()

for ii in range(n_x):
    for jj in range(n_y):
        absdiff = np.log10(abs(blocks.a_atomic(delta, spin,
                                            1.0 - (reals[ii] + imag[jj] * 1.0j), 1.0 - (reals[ii] - imag[jj] * 1.0j)) -
                               blocks.a_atomic_mpversion(delta, spin,
                                            1.0 - (reals[ii] + imag[jj] * 1.0j), 1.0 - (reals[ii] - imag[jj] * 1.0j))) +
                           abs(blocks.a_atomic(delta, spin,
                                                    reals[ii] + imag[jj] * 1.0j, reals[ii] - imag[jj] * 1.0j) -
                               blocks.a_atomic_mpversion(delta, spin,
                                                    reals[ii] + imag[jj] * 1.0j, reals[ii] - imag[jj] * 1.0j)))
        #absdiff = np.log10(abs(sc.hyp2f1(hs, hs, 2*hs, 1.0 - (reals[ii] + imag[jj] * 1.0j)) - \
        #            complex(mp.hyp2f1(hs, hs, 2*hs, 1.0 - (reals[ii] + imag[jj] * 1.0j)))) + \
        #            abs(sc.hyp2f1(hs, hs, 2*hs, reals[ii] + imag[jj] * 1.0j) - \
        #                        complex(mp.hyp2f1(hs, hs, 2*hs, reals[ii] + imag[jj] * 1.0j))))
        if absdiff < -14:
            zre.append(reals[ii])
            zim.append(imag[jj])

        cntr += 1
        print(cntr)

shuffled_indices = list(range(len(zre)))
random.shuffle(shuffled_indices)
shuffled_zre = np.array(zre)[shuffled_indices].tolist()
shuffled_zim = np.array(zim)[shuffled_indices].tolist()

print('LENGTH')
print(len(zre))
print(shuffled_zre[:200])
print(shuffled_zim[:200])
