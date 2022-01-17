import scipy.special as sc
import mpmath as mp
import numpy as np
import random

n_x = 11
n_y = 11

reals_q1 = np.linspace(0.01, 0.99, n_x)
reals_q2 = np.linspace(0.01, 0.99, n_x)
reals_q3 = np.linspace(-0.01, -0.99, n_x)
reals_q4 = np.linspace(-0.01, -0.99, n_x)
reals = np.concatenate((reals_q1, reals_q2, reals_q3, reals_q4))
imag_q1 = np.linspace(0.01, 0.99, n_y)
imag_q2 = np.linspace(-0.01, -0.99, n_y)
imag_q3 = np.linspace(-0.01, -0.99, n_y)
imag_q4 = np.linspace(0.01, 0.99, n_y)
imag = np.concatenate((imag_q1, imag_q2, imag_q3, imag_q4))
hs = 15.0
x = np.zeros(16*n_x*n_y)
y = np.zeros(16*n_x*n_y)
zre = []
zim = []
cntr = 0

for ii in range(4*n_x):
    for jj in range(4*n_y):
        absdiff = np.log10(abs(sc.hyp2f1(hs, hs, 2*hs, 1.0 - (reals[ii] + imag[jj] * 1.0j)) - \
                    complex(mp.hyp2f1(hs, hs, 2*hs, 1.0 - (reals[ii] + imag[jj] * 1.0j)))) + \
                    abs(sc.hyp2f1(hs, hs, 2*hs, reals[ii] + imag[jj] * 1.0j) - \
                                complex(mp.hyp2f1(hs, hs, 2*hs, reals[ii] + imag[jj] * 1.0j))))
        if absdiff < -5:
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
print(shuffled_zre[:100])
print(shuffled_zim[:100])
