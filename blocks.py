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

    def cons(self, z, zb, deltas, ope_coeffs, hh):
        c = ope_coeffs[0]*self.g_a(hh, (deltas[0])/2, (deltas[0])/2, z, zb) - \
            ope_coeffs[1]*self.g_b(hh, (deltas[1])/2, (deltas[1])/2, z, zb) - \
            ope_coeffs[2]*self.g_b_symm(hh, (deltas[2] + 1)/2,
                                        (deltas[2] - 1)/2, z, zb) - \
            ope_coeffs[3]*self.g_b_symm(hh, (deltas[3] + 2)/2,
                                        (deltas[3] - 2)/2, z, zb) - \
            (z**(2*hh)) * (zb**(2*hh))
        return c
