#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from HPRO.lcaodiag import LCAODiagKernel
import os

energy_window = [-3,7]
nbnd = 80

if not os.path.exists('eig.dat'):

    kernel = LCAODiagKernel()
    kernel.setk([[0.000000000000,  0.000000000000,  0.000000000000],
                [0.500000000000,  0.000000000000,  0.000000000000],
                [0.333333333333,  0.333333333333,  0.000000000000],  
                [0.000000000000,  0.000000000000,  0.000000000000],
                [-0.333333333333,  -0.333333333333,  0.000000000000]],
                [10, 10, 10, 10, 1],
                ['\u0413', 'M', 'K','\u0413', 'K'])
    kernel.load_deeph_mats('./')
    kernel.diag(nbnd=nbnd, efermi=None)
    kernel.write('./')


with open("eig.dat") as f:
    lines = f.readlines()
nk, nb = map(int, lines[2].split())
band_lines = lines[3:]
kpt = (np.array(range(nk)).repeat(nb)).reshape(nk, nb)
band = np.ones((nk, nb))
for k in range(nk):
    band[k]  = np.array(list(map(lambda x: float(x.split()[-1]), band_lines[k*(nb+1):(k+1)*(nb+1)][1:])))
plt.plot(kpt, band, color='royalblue')
plt.xlabel('k-point')
plt.ylabel('Energy (eV)')
plt.ylim(energy_window)
plt.xlim(0, nk-1)
plt.savefig('band.png')

