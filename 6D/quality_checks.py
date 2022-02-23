import numpy as np
from blocks import conf_blocks
from hyperparameters import hparams
from scipy.misc import derivative
from data_z_sample import z_data
from numpy import linalg as LA

class quality:

    def __init__(self):
        self.cft = conf_blocks()
        hp = hparams()
        z_d = z_data()
        self.env_shape = self.cft.env_shape
        self.action_space_N = hp.action_space_N
        self.inv_central = hp.inv_c_charge
        self.z = z_d.z
        self.z_conj = z_d.z_conj
        self.inho_value = self.cft.inhomo_z_vector()
        self.short_d_hypers = self.cft.short_coeffs_d_multiplet()
        self.short_b_hypers = self.cft.short_coeffs_b_multiplet_array()
        self.spin_list = hp.spin_list
        self.num_of_operators_long = hp.num_of_operators_long
        self.num_of_operators_short_d = hp.num_of_operators_short_d
        self.num_of_operators_short_b = hp.num_of_operators_short_b

    def basic_crossed_block(self, delta, spin, z, zb):
        formula = (z * zb * self.cft.a_atomic(delta, spin, z, zb) -
                   (z - 1) * (zb - 1) * self.cft.a_atomic(delta, spin, 1 - z, 1 - zb)).real
        return formula

    def crossed_blocks(self, deltas):       # matrix F_DELTA[row][col] = crossed_blocks[i_z][delta]
        temp_blocks = np.zeros((self.env_shape,int(self.action_space_N / 2)))
        for i_z in range(self.env_shape):
            z = self.z[i_z]
            zb = self.z_conj[i_z]
            for i_multiplet in range(int(self.num_of_operators_short_d)):   # D[0,4] contribution
                temp_blocks[i_z][i_multiplet] += self.basic_crossed_block(6, 0, z, zb)

            for i_multiplet in range(int(self.num_of_operators_short_d),
                                     int(self.num_of_operators_short_d) + int(self.num_of_operators_short_b)): # B[0,2]
                temp_blocks[i_z][i_multiplet] += self.basic_crossed_block(self.spin_list[i_multiplet] + 6,
                                                                          self.spin_list[i_multiplet], z, zb)

            for i_multiplet in range(int(self.num_of_operators_short_d) + int(self.num_of_operators_short_b),
                                     int(self.num_of_operators_short_d) + int(self.num_of_operators_short_b) + \
                                     int(self.num_of_operators_long)):  # L[0,0]
                temp_blocks[i_z][i_multiplet] += self.basic_crossed_block(deltas[i_multiplet],
                                                                          self.spin_list[i_multiplet], z, zb)

        return temp_blocks

    def derivative_blocks(self, deltas):
        temp_derivative_blocks = np.zeros((self.env_shape, int(self.action_space_N / 2)))
        for i_z in range(self.env_shape):
            z = self.z[i_z]
            zb = self.z_conj[i_z]

            for i_multiplet in range(int(self.num_of_operators_short_d) + int(self.num_of_operators_short_b),
                                     int(self.action_space_N / 2)):  # L[0,0]
                temp_derivative_blocks[i_z][i_multiplet] += \
                    derivative(lambda x: self.basic_crossed_block(x, self.spin_list[i_multiplet], z, zb),
                               deltas[i_multiplet], dx=1e-10)

        return temp_derivative_blocks

    def matM(self, deltas):
        a = self.crossed_blocks(deltas)
        temp_matM = np.matmul(a.transpose(), a)
        return temp_matM

    def vecN(self, deltas):
        temp_vecN = np.matmul(self.inho_value, self.crossed_blocks(deltas))
        return temp_vecN

    def matP(self, deltas):
        a = self.derivative_blocks(deltas)
        temp_mapP = np.matmul(a.transpose(), self.crossed_blocks(deltas))
        return temp_mapP

    def vecQ(self, deltas):
        temp_vecQ = np.matmul(self.inho_value, self.derivative_blocks(deltas))
        return temp_vecQ

    def c_mc_term(self, deltas, ope_coeffs):
        c_mn = np.matmul(self.matM(deltas), ope_coeffs)
        return c_mn

    def abs_c(self, deltas, ope_coeffs):
        abs_c_mn = LA.inv(np.diag(abs(self.c_mc_term(deltas, ope_coeffs)) + abs(self.vecN(deltas))))
        return abs_c_mn

    def delta_pc_term(self, deltas, ope_coeffs):
        delta_pc = np.matmul(self.matP(deltas), ope_coeffs)
        return delta_pc

    def abs_delta_inv(self, deltas, ope_coeffs):
        abs_delta_inv_vector = np.zeros(int(self.action_space_N / 2))
        for i_delta in range(int(self.action_space_N / 2)):
            if abs(self.vecQ(deltas)[i_delta]) + abs(self.delta_pc_term(deltas, ope_coeffs)[i_delta]) != 0:
                abs_delta_inv_vector[i_delta] += (abs(self.vecQ(deltas)[i_delta]) +
                                                  abs(self.delta_pc_term(deltas, ope_coeffs)[i_delta])) ** (-1)
        return abs_delta_inv_vector

    def abs_delta(self, deltas, ope_coeffs):
        abs_delta_pc = np.diag(self.abs_delta_inv(deltas, ope_coeffs))
        return abs_delta_pc

    def c_equations(self, deltas, ope_coeffs):
        c_eq = self.vecN(deltas) + self.c_mc_term(deltas, ope_coeffs)
        return c_eq

    def relative_c_equations(self, deltas, ope_coeffs):
        relative_c_eq = np.matmul(self.c_equations(deltas, ope_coeffs), self.abs_c(deltas, ope_coeffs))
        return relative_c_eq

    def delta_equations(self, deltas, ope_coeffs):
        delta_eq = self.vecQ(deltas) + self.delta_pc_term(deltas, ope_coeffs)
        return delta_eq

    def relative_delta_equations(self, deltas, ope_coeffs):
        relative_delta_eq = np.matmul(self.delta_equations(deltas, ope_coeffs), self.abs_delta(deltas, ope_coeffs))
        return relative_delta_eq


# input the data and print the quality checks

hpar = hparams()
quality = quality()

filename = input('Specify the file for quality check: ')
f = open(filename, 'r')
tmp1 = f.readlines()
tmp2 = []
f.close()
for i in range(len(tmp1)):
    if i == 0:
        tmp2.append(float(tmp1[i].split('=')[1]))
    else:
        if i == (hpar.action_space_N // 2 + 1):
            pass
        else:
            tmp2.append(float(tmp1[i].split('->')[1]))
num_of_ops = hpar.action_space_N // 2
result_delta = np.array(tmp2[1:num_of_ops + 1])
result_ope = np.array(tmp2[num_of_ops + 1:])

abs_quality_check_array = []
for i in range(num_of_ops):
    abs_quality_check_array.append(quality.delta_equations(result_delta, result_ope)[i])
for i in range(num_of_ops, hpar.action_space_N):
    abs_quality_check_array.append(quality.c_equations(result_delta, result_ope)[i - num_of_ops])

rel_quality_check_array = []
for i in range(num_of_ops):
    rel_quality_check_array.append(quality.relative_delta_equations(result_delta, result_ope)[i])
for i in range(num_of_ops, hpar.action_space_N):
    rel_quality_check_array.append(quality.relative_c_equations(result_delta, result_ope)[i - num_of_ops])

str_quality_check = []
for i in range(hpar.action_space_N):
    str_quality_check.append('(spin=' + str(hpar.spin_list[i%(hpar.action_space_N//2)]) \
                             + ', mult=' + hpar.block_type_printed[
                             hpar.block_type[i%(hpar.action_space_N//2)]] \
                             + ')' + ' -> ' + ' relative: ' + str(rel_quality_check_array[i])
                             + ',' + ' absolute: ' + str(abs_quality_check_array[i]) + '\n')
    if i == (num_of_ops - 1):
        str_quality_check.append('-------------------------\n')

file = open('quality_print_out.txt', 'w')
file.writelines(str_quality_check)
file.close()
