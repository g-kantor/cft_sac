import numpy as np
import scipy.special as sc
import mpmath as mp
from hyperparameters import hparams

class conf_blocks:
    #Computes conformal blocks and a-crossing equation for the 6d (2,0) SCFT in 1507.05637

    def __init__(self):
        hp = hparams()
        self.central = hp.central
        self.env_shape = hp.env_shape
        self.action_space_N = hp.action_space_N
        self.ell_max_chi = hp.ell_max_chi
        self.block_type = hp.block_type
        self.spin_list = hp.spin_list
        self.zre = hp.zre
        self.zim = hp.zim

    def f_nm(self, n, m, ell, delta, delta12, delta34, z, zb):
        n = int(n)
        m = int(m)
        delta = float(delta)
        delta12 = float(delta12)
        delta34 = float(delta34)
        z = complex(z)
        zb = complex(zb)

        res_zzb = (z * zb) ** ((delta - ell) / 2) * (z - zb) ** (-3) * (
                  ((- z / 2) ** ell) * (z ** (n + 3)) * (zb ** m) *
                    complex(sc.hyp2f1(((delta + ell - delta12) / 2) + n,
                            ((delta + ell + delta34) / 2) + n, delta + ell + 2 * n, z)) *
                    complex(sc.hyp2f1(((delta - ell - delta12) / 2) - 3 + m, ((delta - ell + delta34) / 2) - 3 + m,
                            delta - ell - 6 + 2 * m, zb)))

        res_zbz = (z * zb) ** ((delta - ell) / 2) * (z - zb) ** (-3) * (
                  ((- zb / 2) ** ell) * (zb ** (n + 3)) * (z ** m) *
                    complex(sc.hyp2f1(((delta + ell - delta12) / 2) + n,
                            ((delta + ell + delta34) / 2) + n, delta + ell + 2 * n, zb)) *
                    complex(sc.hyp2f1(((delta - ell - delta12) / 2) - 3 + m, ((delta - ell + delta34) / 2) - 3 + m,
                            delta - ell - 6 + 2 * m, z)))

        res = res_zzb - res_zbz
        return res

    def f_mpversion(self, n, m, ell, delta, delta12, delta34, z, zb):
        n = int(n)
        m = int(m)
        delta = float(delta)
        delta12 = float(delta12)
        delta34 = float(delta34)
        z = complex(z)
        zb = complex(zb)

        res_zzb = (z * zb) ** ((delta - ell) / 2) * (z - zb) ** (-3) * (
                  ((- z / 2) ** ell) * (z ** (n + 3)) * (zb ** m) *
                    complex(mp.hyp2f1(((delta + ell - delta12) / 2) + n,
                            ((delta + ell + delta34) / 2) + n, delta + ell + 2 * n, z)) *
                    complex(mp.hyp2f1(((delta - ell - delta12) / 2) - 3 + m, ((delta - ell + delta34) / 2) - 3 + m,
                            delta - ell - 6 + 2 * m, zb)))

        res_zbz = (z * zb) ** ((delta - ell) / 2) * (z - zb) ** (-3) * (
                  ((- zb / 2) ** ell) * (zb ** (n + 3)) * (z ** m) *
                    complex(mp.hyp2f1(((delta + ell - delta12) / 2) + n,
                            ((delta + ell + delta34) / 2) + n, delta + ell + 2 * n, zb)) *
                    complex(mp.hyp2f1(((delta - ell - delta12) / 2) - 3 + m, ((delta - ell + delta34) / 2) - 3 + m,
                            delta - ell - 6 + 2 * m, z)))

        res = res_zzb - res_zbz
        return res

    def g_l_delta(self, ell, delta, delta12, delta34, z, zb):
        ell = int(ell)
        delta = float(delta)
        delta12 = float(delta12)
        delta34 = float(delta34)
        z = complex(z)
        zb = complex(zb)

        res_g = self.f_nm(0, 0, ell, delta, delta12, delta34, z, zb) - \
            (ell + 3) * ((ell + 1) ** (-1)) * self.f_nm(-1, 1, ell, delta, delta12, delta34, z, zb) + \
            (delta - 4) * (ell + 3) * ((16 * (delta - 2) * (ell + 1)) ** (-1)) * \
            (delta - ell - delta12 - 4) * (delta - ell + delta12 - 4) * (delta - ell + delta34 - 4) * \
            (delta - ell - delta34 - 4) * \
            (((delta - ell - 5) * ((delta - ell - 4) ** 2) * (delta - ell - 3)) ** (-1)) * \
            self.f_nm(0, 2, ell, delta, delta12, delta34, z, zb) - \
            (delta - 4) * ((delta - 2) ** (-1)) * (delta - ell - delta12) * (delta + ell + delta12) * \
            (delta + ell + delta34) * (delta + ell - delta34) * \
            ((16 * (delta + ell - 1) * ((delta + ell) ** 2) * (delta + ell + 1)) ** (-1)) * \
            self.f_nm(1, 1, ell, delta, delta12, delta34, z, zb) + \
            2 * (delta - 4) * (ell + 3) * delta12 * delta34 * \
            (((delta + ell) * (delta + ell - 2) * (delta + ell - 4) * (delta + ell - 6)) ** (-1)) * \
            self.f_nm(0, 1, ell, delta, delta12, delta34, z, zb)
        return res_g

    def g_l_delta_mpversion(self, ell, delta, delta12, delta34, z, zb):
        ell = int(ell)
        delta = float(delta)
        delta12 = float(delta12)
        delta34 = float(delta34)
        z = complex(z)
        zb = complex(zb)

        res_g = self.f_mpversion(0, 0, ell, delta, delta12, delta34, z, zb) - \
            (ell + 3) * ((ell + 1) ** (-1)) * self.f_mpversion(-1, 1, ell, delta, delta12, delta34, z, zb) + \
            (delta - 4) * (ell + 3) * ((16 * (delta - 2) * (ell + 1)) ** (-1)) * \
            (delta - ell - delta12 - 4) * (delta - ell + delta12 - 4) * (delta - ell + delta34 - 4) * \
            (delta - ell - delta34 - 4) * \
            (((delta - ell - 5) * ((delta - ell - 4) ** 2) * (delta - ell - 3)) ** (-1)) * \
            self.f_mpversion(0, 2, ell, delta, delta12, delta34, z, zb) - \
            (delta - 4) * ((delta - 2) ** (-1)) * (delta - ell - delta12) * (delta + ell + delta12) * \
            (delta + ell + delta34) * (delta + ell - delta34) * \
            ((16 * (delta + ell - 1) * ((delta + ell) ** 2) * (delta + ell + 1)) ** (-1)) * \
            self.f_mpversion(1, 1, ell, delta, delta12, delta34, z, zb) + \
            2 * (delta - 4) * (ell + 3) * delta12 * delta34 * \
            (((delta + ell) * (delta + ell - 2) * (delta + ell - 4) * (delta + ell - 6)) ** (-1)) * \
            self.f_mpversion(0, 1, ell, delta, delta12, delta34, z, zb)
        return res_g

    def a_atomic(self, delta, ell, z, zb):
        ell = int(ell)
        delta = float(delta)
        z = complex(z)
        zb = complex(zb)

        res_at = 4 * (((z ** 6) * (zb ** 6) * (delta - ell - 2) * (delta + ell + 2)) ** (-1)) * \
                 self.g_l_delta(ell, delta + 4, 0, -2, z, zb)
        return res_at

    def a_atomic_mpversion(self, delta, ell, z, zb):
        ell = int(ell)
        delta = float(delta)
        z = complex(z)
        zb = complex(zb)

        res_at = 4 * (((z ** 6) * (zb ** 6) * (delta - ell - 2) * (delta + ell + 2)) ** (-1)) * \
                 self.g_l_delta_mpversion(ell, delta + 4, 0, -2, z, zb)
        return res_at

    def b_l(self, central, ell):
        central = float(central)
        ell = int(ell)
        half_ell = int(ell / 2)

        res_b = (ell + 1) * (ell + 3) * ((ell + 2) ** 2) * sc.factorial(ell / 2) * \
            sc.factorial2(half_ell + 2, exact=True) * sc.factorial2(half_ell + 3, exact=True) * \
            sc.factorial2(ell + 5, exact=True) * \
            ((18 * sc.factorial2(ell + 2, exact=True) * sc.factorial2(2 * ell + 5, exact=True)) ** (-1)) + \
            (8 / central) * ((sc.factorial2(2 * ell + 5, exact=True)) ** (-1)) * \
            (2 ** (- half_ell - 1)) * (ell * (ell + 7) + 11) * sc.factorial2(ell + 3, exact=True) * \
            sc.gamma(half_ell + 2)
        return res_b

    def h(self, central, z):
        central = float(central)
        z = complex(z)

        res_h = - ((1 / 3) * (z ** 3) - (z - 1) ** (-1) - (z - 1) ** (- 2) - (3 * (z - 1) ** 3) ** (-1) - z ** (-1)) - \
            (8 / central) * (z - (z - 1) ** (-1) + np.log(1 - z)) + (- 1 / 6 + 8 / central )
        return res_h

    def a_chi(self, central, ell_max, z, zb):
        z = complex(z)
        zb = complex(zb)
        central = float(central)
        ell_max = int(ell_max)      # half of upper cutoff on ell sum

        res_a_chi = 0
        for ell_half in range(0, ell_max + 1):
            res_a_chi += (2 ** (2 * ell_half)) * self.b_l(central, 2 * ell_half) * \
                         self.a_atomic(2 * ell_half + 4, 2 * ell_half, z, zb)
        return res_a_chi

    def inhomo(self, central, ell_max_chi, z, zb):
        z = complex(z)
        zb = complex(zb)
        central = float(central)
        ell_max_chi = int(ell_max_chi)

        res_inhomo = - ((z - zb) ** (-3)) * (
                      (self.h(central, 1 - zb) - self.h(central, 1 - z)) * (((z - 1) * (zb - 1)) ** (-1)) +
                      (self.h(central, zb) - self.h(central, z)) * ((z * zb) ** (-1))) - \
            (z - 1) * (zb - 1) * self.a_chi(central, ell_max_chi, 1 - z, 1 - zb) + \
            z * zb * self.a_chi(central, ell_max_chi, z, zb)
        return res_inhomo

    def inhomo_z_vector(self):          #evaluates outside the move loop the inhomo part on the z-sampling
        res_inhomo_z_vector = np.zeros(self.env_shape)
        for i in range(self.env_shape):
            res_inhomo_z_vector[i] = (self.inhomo(self.central, self.ell_max_chi,
                                                 self.zre[i] + self.zim[i] * 1j, self.zre[i] - self.zim[i] * 1j)).real

        return res_inhomo_z_vector

    def short_cons_coeffs(self):   #evaluates outside the move loop the D, B part on the z-sampling
        short_coeffs = np.zeros((self.env_shape, int(self.action_space_N / 2)))
        for i_z in range(self.env_shape):
            z = self.zre[i_z] + self.zim[i_z] * 1j
            zb = self.zre[i_z] - self.zim[i_z] * 1j
            for i_multiplet in range(int(self.action_space_N / 2)):
                if self.block_type[i_multiplet] == 1:  # B[0,2] short multiplets
                    short_coeffs[i_z][i_multiplet] += (z * zb * self.a_atomic(self.spin_list[i_multiplet] + 7,
                                                                         self.spin_list[i_multiplet], z, zb) -
                                                  (z - 1) * (zb - 1) * self.a_atomic(self.spin_list[i_multiplet] + 7,
                                                                                     self.spin_list[i_multiplet],
                                                                                     1 - z, 1 - zb)).real
                elif self.block_type[i_multiplet] == 2:  # D[0,4] short multiplets
                    short_coeffs[i_z][i_multiplet] += (z * zb * self.a_atomic(6, 0, z, zb) -
                                                  (z - 1) * (zb - 1) * self.a_atomic(6, 0, 1 - z, 1 - zb)).real

        return short_coeffs

    #def short_cons(self, ope_coeffs):   #evaluates outside the move loop the D, B part on the z-sampling
    #    short_c = np.zeros((self.env_shape, int(self.action_space_N / 2)))
    #    for i_z in range(self.env_shape):
    #        z = self.zre[i_z] + self.zim[i_z] * 1j
    #        zb = self.zre[i_z] - self.zim[i_z] * 1j
    #        for i_multiplet in range(int(self.action_space_N / 2)):
    #            if self.block_type[i_multiplet] == 1:  # B[0,2] short multiplets
    #                short_c[i_z][i_multiplet] += ope_coeffs[i_multiplet] * \
    #                    (z * zb * self.a_atomic(self.spin_list[i_multiplet] + 7, self.spin_list[i_multiplet], z, zb) -
    #                        (z - 1) * (zb - 1) * self.a_atomic(self.spin_list[i_multiplet] + 7,
    #                                                           self.spin_list[i_multiplet], 1 - z, 1 - zb)).real
    #            elif self.block_type[i_multiplet] == 2:  # D[0,4] short multiplets
    #                short_c[i_z] += ope_coeffs[i_multiplet] * \
    #                    (z * zb * self.a_atomic(6, 0, z, zb)
    #                    - (z - 1) * (zb - 1) * self.a_atomic(6, 0, 1 - z, 1 - zb)).real
    #
    #    return short_c

    def long_cons(self, z, zb, deltas, ope_coeffs, block_type, spin_list):
        long_c = 0
        for i in range(len(deltas)):
            if block_type[i] == 3:  # L[0,0] long multiplets
                long_c += ope_coeffs[i] * (z * zb * self.a_atomic(deltas[i], spin_list[i], z, zb) -
                                           (z - 1) * (zb - 1) * self.a_atomic(deltas[i], spin_list[i], 1 - z, 1 - zb))
        return long_c

    #def cons(self, central, ell_max_chi, z, zb, deltas, ope_coeffs, block_type, spin_list):
    #    central = float(central)
    #    ell_max_chi = int(ell_max_chi)
    #    c = self.inhomo(central, ell_max_chi, z, zb)
    #    for i in range(len(deltas)):
    #        if block_type[i] == 1:  # B[0,2] short multiplets
    #            c += ope_coeffs[i] * (z * zb * self.a_atomic(spin_list[i] + 7, spin_list[i], z, zb) -
    #                                  (z - 1) * (zb - 1) * self.a_atomic(spin_list[i] + 7, spin_list[i], 1 - z, 1 - zb))
    #        if block_type[i] == 2:  # D[0,4] short multiplets
    #            c += ope_coeffs[i] * (z * zb * self.a_atomic(6, 0, z, zb) -
    #                                  (z - 1) * (zb - 1) * self.a_atomic(6, 0, 1 - z, 1 - zb))
    #        if block_type[i] == 3:  # L[0,0] long multiplets
    #            c += ope_coeffs[i] * (z * zb * self.a_atomic(deltas[i], spin_list[i], z, zb) -
    #                                  (z - 1) * (zb - 1) * self.a_atomic(deltas[i], spin_list[i], 1 - z, 1 - zb))
    #    return c

    def weight(self, r, a):
        r = float(r)
        a = float(a)
        # weight_function = r * (1 + r ** 2 - 2 * r * np.cos(a)) * ((1 + np.exp(- a * 1j) * r)**(-3)) *
        # ((1 + np.exp(a * 1j) * r)**(-3)) # here input the weight function of the z-sampling/integration
        weight_function = 1  # here input the weight function of the z-sampling/integration
        return weight_function
