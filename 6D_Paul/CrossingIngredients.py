import numpy as np
import scipy.special as sc


class ConformalBlocks:
    """Computes conformal block and crossing equation for 6d (2,0) SCFT as in 1507.05637"""

    def h(self, z, inv_c_charge):
        """Compute h(z) defined in eqn (3.21) of 1507.05637"""
        res_a = - (1 / 3 * (z ** 3) - (z - 1) ** (-1) - (z - 1) ** (-2) - 1 / 3 * (z - 1) ** (-3) - z ** (-1)) \
                - 8 * inv_c_charge * (z - (z - 1) ** (-1) + np.log(1 - z)) - 1 / 6 + 8 * inv_c_charge
        return res_a

    def c_h(self, z, zb, inv_c_charge):  # This is eqn (2.15) of 2105.13361
        """Compute C(z,zb) defined in eqn (2.15) of 2105.13361"""
        res_a = (z - zb) ** (-3) * (z * zb) ** (-1) * (self.h(z, inv_c_charge) - self.h(zb, inv_c_charge))
        return res_a

    def curlyfinter(self, n, m, spin, delta, z, zb, delta12, delta34):
        res_a = (-1 / 2) ** spin * z ** (spin + n + 3) * zb ** m * complex(sc.hyp2f1(0.5 * (delta + spin - delta12) + n,
                                                                                     0.5 * (delta + spin + delta34) + n,
                                                                                     delta + spin + 2 * n,
                                                                                     z)) * complex(
            sc.hyp2f1(0.5 * (delta - spin - delta12) - 3 + m,
                      0.5 * (delta - spin + delta34) - 3 + m,
                      delta - spin - 6 + 2 * m, zb))
        return res_a

    def curlyf(self, n, m, spin, delta, z, zb, delta12, delta34):
        """Compute the function defined after (B.1) of 1507.05637"""
        res_a = (z * zb) ** (1 / 2 * (delta - spin)) * (z - zb) ** (-3) \
                * (self.curlyfinter(n, m, spin, delta, z, zb, delta12, delta34) - self.curlyfinter(n, m, spin, delta,
                                                                                                   zb, z, delta12,
                                                                                                   delta34))
        return res_a

    def coeff3(self, spin, delta, delta12, delta34):
        numerator3 = (delta - 4) * (spin + 3) * (delta - spin - delta12 - 4) * (delta - spin + delta12 - 4) \
                     * (delta - spin + delta34 - 4) * (delta - spin - delta34 - 4)
        denominator3 = 16 * (delta - 2) * (spin + 1) * (delta - spin - 5) * (delta - spin - 4) ** 2 * (delta - spin - 3)
        res_a = numerator3 / denominator3
        return res_a

    def coeff4(self, spin, delta, delta12, delta34):
        numerator4 = - (delta - 4) * (delta + spin - delta12) * (delta + spin + delta12) * (delta + spin + delta34) \
                     * (delta + spin - delta34)
        denominator4 = 16 * (delta - 2) * (delta + spin - 1) * (delta + spin) ** 2 * (delta + spin + 1)
        res_a = numerator4 / denominator4
        return res_a

    def coeff5(self, spin, delta, delta12, delta34):
        numerator5 = 2 * (delta - 4) * (spin + 3) * delta12 * delta34
        denominator5 = (delta + spin) * (delta + spin - 2) * (delta + spin - 4) * (delta + spin - 6)
        res_a = numerator5 / denominator5  # Should we add error handling for possibility of denominator = 0?
        return res_a

    def curlyg(self, spin, delta, z, zb, delta12=0, delta34=-2):
        """This takes the coeff and the curlyf functions and forms (B.1) of 1507.05637"""
        res_a = self.curlyf(0, 0, spin, delta, z, zb, delta12, delta34) \
                - (spin + 3) / (spin + 1) * self.curlyf(-1, 1, spin, delta, z, zb, delta12, delta34) \
                + self.coeff3(spin, delta, delta12, delta34) * self.curlyf(0, 2, spin, delta, z, zb, delta12, delta34) \
                + self.coeff4(spin, delta, delta12, delta34) * self.curlyf(1, 1, spin, delta, z, zb, delta12, delta34) \
                + self.coeff5(spin, delta, delta12, delta34) * self.curlyf(0, 1, spin, delta, z, zb, delta12, delta34)
        return res_a

    def bcoeff(self, spin, inv_c_charge):  # Compute the b_l coefficients given in (4.9) of 1507.05637
        halfspin = int(1 / 2 * spin)  # This is needed to avoid error in factorial2 as it doesn't like float variables
        numerator1 = (spin + 1) * (spin + 3) * (spin + 2) ** 2 * sc.factorial(halfspin) \
                     * sc.factorial2(halfspin + 2, exact=True) * sc.factorial2(halfspin + 3, exact=True) \
                     * sc.factorial2(spin + 5, exact=True)
        numerator2 = inv_c_charge * 8 * 2 ** (-(halfspin + 1)) * (spin * (spin + 7) + 11) \
                     * sc.factorial2(spin + 3, exact=True) * sc.gamma(halfspin + 2)
        denominator1 = 18 * sc.factorial2(spin + 2, exact=True) * sc.factorial2(2 * spin + 5, exact=True)
        denominator2 = sc.factorial2(2 * spin + 5, exact=True)
        res_a = numerator1 / denominator1 + numerator2 / denominator2
        return res_a

    def a_atomic(self, delta, spin, z, zb):
        res_a = 4 * ((z ** 6) * (zb ** 6) * (delta - spin - 2) * (delta + spin + 2)) ** (-1) \
                * self.curlyg(spin, delta + 4, z, zb)
        return res_a

    def a_chi(self, inv_c_charge, spin_cutoff, z, zb):
        res_a = sum((2 ** (2 * k)) * self.bcoeff(2 * k, inv_c_charge)
                    * self.a_atomic(2 * k + 4, 2 * k, z, zb) for k in range(0, int(0.5 * spin_cutoff + 1)))
        return res_a

    def multiplet_block_d(self, z, zb, ope_coeff_d):
        res_a = ope_coeff_d * self.a_atomic(6, 0, z, zb)
        return res_a

    def multiplet_block_b(self, z, zb, spin_cutoff, ope_coeffs_b):
        res_a = sum(ope_coeffs_b[k] * self.a_atomic(2 * (k + 1) + 6, 0, z, zb) for k in range(0, int(0.5 * spin_cutoff)))
        return res_a

    # def multiplet_block_long(self, z, zb, spin_cutoff, num_of_deltas, cdims_long, ope_coeffs_long):
    #     res_a = sum(sum(ope_coeffs_long[spin_counter, delta_counter] * self.a_atomic(cdims_long[delta_counter],
    #                                                                                  2 * spin_counter, z, zb)
    #                     for delta_counter in range(num_of_deltas)) for spin_counter in range(0, int(0.5 * spin_cutoff + 1)))
    #     return res_a

    def multiplet_block_long(self, z, zb, spin_cutoff, cdims_long, ope_coeffs_long):
        res_a = sum(ope_coeffs_long[k] * self.a_atomic(cdims_long[k], 2 * k, z, zb)
                    for k in range(0, int(0.5 * spin_cutoff + 1)))
        return res_a

    def channel(self, z, zb, inv_c_charge, spin_cutoff, ope_coeff_d, ope_coeffs_b, ope_coeffs_long, cdims_long):
        res_a = z * zb * self.a_chi(inv_c_charge, spin_cutoff, z, zb) \
                + z * zb * self.multiplet_block_d(z, zb, ope_coeff_d) \
                + z * zb * self.multiplet_block_b(z, zb, spin_cutoff, ope_coeffs_b) \
                + z * zb * self.multiplet_block_long(z, zb, spin_cutoff, cdims_long, ope_coeffs_long) \
                + self.c_h(z, zb, inv_c_charge)
        return res_a

    def cons(self, z, zb, inv_c_charge, spin_cutoff, ope_coeff_d, ope_coeffs_b, ope_coeffs_long, cdims_long):
        crossing = self.channel(z, zb, inv_c_charge, spin_cutoff, ope_coeff_d, ope_coeffs_b, ope_coeffs_long, cdims_long) \
                   - self.channel(1 - z, 1 - zb, inv_c_charge, spin_cutoff, ope_coeff_d, ope_coeffs_b, ope_coeffs_long, cdims_long)
        return crossing
