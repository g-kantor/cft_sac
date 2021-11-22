import numpy as np
import scipy.special as sc

class conf_blocks:

    def g_a(self, h_, h, hb, z, zb):
        h = float(h)
        z = complex(z)
        hb = float(hb)
        zb = complex(zb)
        res_a = ((z - 1)**(2*h_)) * ((zb - 1)**(2*h_)) * (z**h) * (zb**hb) * \
                complex(sc.hyp2f1(h, h, 2*h, z)) * \
                complex(sc.hyp2f1(hb, hb, 2*hb, zb))
        return res_a

    def g_a_symm(self, h_, h, hb, z, zb):
        h = float(h)
        z = complex(z)
        hb = float(hb)
        zb = complex(zb)
        res_a = ((z - 1)**(2*h_)) * ((zb - 1)**(2*h_)) * ((z**h) * (zb**hb) * \
                complex(sc.hyp2f1(h, h, 2*h, z)) * \
                complex(sc.hyp2f1(hb, hb, 2*hb, zb)) + (zb**h) * (z**hb) * \
                complex(sc.hyp2f1(h, h, 2*h, zb)) * \
                complex(sc.hyp2f1(hb, hb, 2*hb, z)))
        return res_a

    def g_b(self, h_, h, hb, z, zb):
        h = float(h)
        z = complex(z)
        hb = float(hb)
        zb = complex(zb)
        res_b = ((1 - z)**h) * ((1 - zb)**hb) * (z**(2*h_)) * (zb**(2*h_)) * \
                complex(sc.hyp2f1(h, h, 2*h, 1 - z)) * \
                complex(sc.hyp2f1(hb, hb, 2*hb, 1 - zb))
        return res_b

    def g_b_symm(self, h_, h, hb, z, zb):
        h = float(h)
        z = complex(z)
        hb = float(hb)
        zb = complex(zb)
        res_b = (z**(2*h_)) * (zb**(2*h_)) * (((1 - z)**h) * ((1 - zb)**hb) * \
                complex(sc.hyp2f1(h, h, 2*h, 1 - z)) * \
                complex(sc.hyp2f1(hb, hb, 2*hb, 1 - zb)) + ((1 - zb)**h) * \
                ((1 - z)**hb) * complex(sc.hyp2f1(h, h, 2*h, 1 - zb)) * \
                complex(sc.hyp2f1(hb, hb, 2*hb, 1 - z)))
        return res_b

    def cons(self, z, zb, deltas, ope_coeffs, hh, block_type, spin_list):
        c = - (z**(2*hh)) * (zb**(2*hh))
        for i in range(len(deltas)):
            if block_type[i] == 1:
                c += ope_coeffs[i]*self.g_a(hh, (deltas[i])/2, (deltas[i])/2, z, zb)
            if block_type[i] == 2:
                c += ope_coeffs[i]*self.g_a_symm(hh, (deltas[i] + spin_list[i])/2, (deltas[i] - spin_list[i])/2, z, zb)
            if block_type[i] == 3:
                c -= ope_coeffs[i]*self.g_b(hh, (deltas[i])/2, (deltas[i])/2, z, zb)
            if block_type[i] == 4:
                c += ope_coeffs[i]*self.g_b_symm(hh, (deltas[i] + spin_list[i])/2, (deltas[i] - spin_list[i])/2, z, zb)

        return c
