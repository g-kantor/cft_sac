import numpy as np
import mpmath as mp
import scipy.special as sc
from hyperparameters import hparams
from data_z_sample import z_data

class conf_blocks:
    """Computes conformal block and crossing equation for 6d (2,0) SCFT as in 1507.05637"""

    def __init__(self):
        hp = hparams()
        self.spin_list_short_b = hp.spin_list_short_b
        self.spin_list_long = hp.spin_list_long
        self.inv_c_charge = hp.inv_c_charge
        self.action_space_N = hp.action_space_N
        self.ell_max = hp.ell_max
        zd = z_data()
        self.env_shape = zd.env_shape
        self.z = zd.z
        self.z_conj = zd.z_conj
        self.hyper_geometric = "scipy"

    """
    def h(self, z):
        #Compute h(z) defined in eqn (3.21) of 1507.05637
        res_a = - ((1 / 3) * (z ** 3) - (z - 1) ** (-1) - (z - 1) ** (-2) - (1 / 3) * (z - 1) ** (-3) - z ** (-1)) \
                - 8 * self.inv_c_charge * (z - (z - 1) ** (-1) + np.log(1 - z)) - (1 / 6) + 8 * self.inv_c_charge
        return res_a
    """

    def c_h(self, z, zb):
        """Compute C(z,zb) defined in eqn (2.15) of 2105.13361"""
        res_a = (z - zb) ** (-3) * (z * zb) ** (-1) * (self.h(z) - self.h(zb))
        return res_a

    def f_nm(self, n, m, ell, delta, z, zb, delta12, delta34):
        """Compute the function defined after (B.1) of 1507.05637"""
        prefactor = (z * zb) ** ((delta - ell) / 2) * (z - zb) ** (-3)
        if self.hyper_geometric == "scipy":
            res_zzb = prefactor * (
                    ((- z / 2) ** ell) * (z ** (n + 3)) * (zb ** m) *
                    sc.hyp2f1(((delta + ell - delta12) / 2) + n,
                              ((delta + ell + delta34) / 2) + n, delta + ell + 2 * n, z) *
                    sc.hyp2f1(((delta - ell - delta12) / 2) - 3 + m, ((delta - ell + delta34) / 2) - 3 + m,
                              delta - ell - 6 + 2 * m, zb))

            res_zbz = prefactor * (
                    ((- zb / 2) ** ell) * (zb ** (n + 3)) * (z ** m) *
                    sc.hyp2f1(((delta + ell - delta12) / 2) + n,
                              ((delta + ell + delta34) / 2) + n, delta + ell + 2 * n, zb) *
                    sc.hyp2f1(((delta - ell - delta12) / 2) - 3 + m, ((delta - ell + delta34) / 2) - 3 + m,
                              delta - ell - 6 + 2 * m, z))
        else:
            res_zzb = prefactor * (
                    ((- z / 2) ** ell) * (z ** (n + 3)) * (zb ** m) *
                    mp.hyp2f1(((delta + ell - delta12) / 2) + n,
                              ((delta + ell + delta34) / 2) + n, delta + ell + 2 * n, z) *
                    mp.hyp2f1(((delta - ell - delta12) / 2) - 3 + m, ((delta - ell + delta34) / 2) - 3 + m,
                              delta - ell - 6 + 2 * m, zb))

            res_zbz = prefactor * (
                    ((- zb / 2) ** ell) * (zb ** (n + 3)) * (z ** m) *
                    mp.hyp2f1(((delta + ell - delta12) / 2) + n,
                              ((delta + ell + delta34) / 2) + n, delta + ell + 2 * n, zb) *
                    mp.hyp2f1(((delta - ell - delta12) / 2) - 3 + m, ((delta - ell + delta34) / 2) - 3 + m,
                              delta - ell - 6 + 2 * m, z))

        res = res_zzb - res_zbz
        return res

    def g_l_delta(self, ell, delta, z, zb, delta12=0, delta34=-2):
        """This computes the function defined in (B.1) of 1507.05637"""
        # Should we add error handling for possibility of denominator = 0?

        numerator3 = (delta - 4) * (ell + 3) * (delta - ell - delta12 - 4) * (delta - ell + delta12 - 4) \
                     * (delta - ell + delta34 - 4) * (delta - ell - delta34 - 4)
        denominator3 = 16 * (delta - 2) * (ell + 1) * (delta - ell - 5) * (delta - ell - 4) ** 2 * (delta - ell - 3)
        g_l_delta_coeff_3 = numerator3 / denominator3

        numerator4 = - (delta - 4) * (delta + ell - delta12) * (delta + ell + delta12) * (delta + ell + delta34) \
                     * (delta + ell - delta34)
        denominator4 = 16 * (delta - 2) * (delta + ell - 1) * (delta + ell) ** 2 * (delta + ell + 1)
        g_l_delta_coeff_4 = numerator4 / denominator4

        numerator5 = 2 * (delta - 4) * (ell + 3) * delta12 * delta34
        denominator5 = (delta + ell) * (delta + ell - 2) * (delta + ell - 4) * (delta + ell - 6)
        g_l_delta_coeff_5 = numerator5 / denominator5

        res_a = self.f_nm(0, 0, ell, delta, z, zb, delta12, delta34) \
                - (ell + 3) / (ell + 1) * self.f_nm(-1, 1, ell, delta, z, zb, delta12, delta34) \
                + g_l_delta_coeff_3 * self.f_nm(0, 2, ell, delta, z, zb, delta12, delta34) \
                + g_l_delta_coeff_4 * self.f_nm(1, 1, ell, delta, z, zb, delta12, delta34) \
                + g_l_delta_coeff_5 * self.f_nm(0, 1, ell, delta, z, zb, delta12, delta34)
        return res_a

    def b_l(self, ell):
        """"Computes the b_l coefficients given in (4.9) of 1507.05637"""
        half_ell = ell//2  # This is needed to avoid error in factorial2 as it doesn't like float variables

        numerator1 = (ell + 1) * (ell + 3) * (ell + 2) ** 2 * sc.factorial(half_ell) \
                     * sc.factorial2(half_ell + 2, exact=True) * sc.factorial2(half_ell + 3, exact=True) \
                     * sc.factorial2(ell + 5, exact=True)
        numerator2 = self.inv_c_charge * 8 * (2 ** (-(half_ell + 1))) * (ell * (ell + 7) + 11) \
                     * sc.factorial2(ell + 3, exact=True) * sc.gamma(half_ell + 2)
        denominator1 = 18 * sc.factorial2(ell + 2, exact=True) * sc.factorial2(2 * ell + 5, exact=True)
        denominator2 = sc.factorial2(2 * ell + 5, exact=True)
        res_a = numerator1 / denominator1 + numerator2 / denominator2
        return res_a

    def a_atomic(self, delta, ell, z, zb):
        """Computes the function 'a' in (4.5) of 1507.05637"""
        res_a = 4 * ((z ** 6) * (zb ** 6) * (delta - ell - 2) * (delta + ell + 2)) ** (-1) \
                * self.g_l_delta(ell, delta + 4, z, zb)
        return res_a

    def h_atomic(self, beta, z):
        res = z**(beta - 1) * (1 - beta)**(-1) * sc.hyp2f1(beta - 1, beta, 2 * beta, z)
        return res

    def h(self, z):
        #Compute h(z) defined in eqn (4.9) of 1507.05637
        res = self.h_atomic(0, z) + 8 * self.inv_c_charge * self.h_atomic(2, z) + \
              np.sum(self.b_l(k) * self.h_atomic(k + 4, z) for k in range(0, 25, 2))
        return res

    def a_chi(self, z, zb):
        """Computes the function given in (4.11) of 1507.05637"""
        res_a = np.sum((2 ** k) * self.b_l(k)
                       * self.a_atomic(k + 4, k, z, zb) for k in range(0, 25, 2))
        return res_a

    def inhomo_z_vector(self):
        """Computes the RHS of (4.13) in 1507.05637 for each point in the z-sampling"""
        # evaluates outside the move loop the inhomo part on the z-sampling

        res_inhomo_z_vector = - ((self.z - self.z_conj) ** (-3)) * (
                (self.h(1 - self.z_conj) - self.h(1 - self.z)) * (((self.z - 1) * (self.z_conj - 1)) ** (-1))
                + (self.h(self.z_conj) - self.h(self.z)) * ((self.z * self.z_conj) ** (-1))) \
                              - (self.z - 1) * (self.z_conj - 1) * self.a_chi(1 - self.z, 1 - self.z_conj) \
                              + self.z * self.z_conj * self.a_chi(self.z, self.z_conj)

        return res_inhomo_z_vector.real

    def short_coeffs_b_multiplet(self, ell):  # computes the spin ell B multiplet conformal block
        res = (self.z * self.z_conj * self.a_atomic(ell + 6, ell, self.z, self.z_conj)
               - (self.z - 1) * (self.z_conj - 1)
               * self.a_atomic(ell + 6, ell, 1 - self.z, 1 - self.z_conj)).real
        return res

    def short_coeffs_b_multiplet_array(self):
        b_multiplet_array = self.short_coeffs_b_multiplet(self.spin_list_short_b[0])
        if self.spin_list_short_b.size == 1:
            return b_multiplet_array
        else:
            b_multiplet_spins = self.spin_list_short_b[1:]
            for ell in b_multiplet_spins:
                b_multiplet_array = np.vstack((b_multiplet_array, self.short_coeffs_b_multiplet(ell)))

        return b_multiplet_array

    def short_coeffs_d_multiplet(self):
        return (self.z * self.z_conj * self.a_atomic(6, 0, self.z, self.z_conj)
                - (self.z - 1) * (self.z_conj - 1) * self.a_atomic(6, 0, 1 - self.z, 1 - self.z_conj)).real

    def long_cons(self, delta, ell):
        long_c = (self.z * self.z_conj * self.a_atomic(delta, ell, self.z, self.z_conj)
                  - (self.z - 1) * (self.z_conj - 1) * self.a_atomic(delta, ell, 1 - self.z, 1 - self.z_conj)).real
        return long_c

    def long_coeffs_array(self, deltas):
        long_c = self.long_cons(deltas[0], self.spin_list_long[0])
        for x in range(1, deltas.size):
            long_c = np.vstack((long_c, self.long_cons(deltas[x], self.spin_list_long[x])))
        return long_c
