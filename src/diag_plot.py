import numpy as np
import matplotlib.pyplot as plt

from HPRO.lcaodiag import LCAODiagKernel

kernel = LCAODiagKernel()
kernel.setk([[0.000000000000,  0.000000000000,  0.000000000000],
             [0.500000000000,  0.000000000000,  0.000000000000],
             [0.333333333333,  0.333333333333,  0.000000000000],  
             [0.000000000000,  0.000000000000,  0.000000000000],
             [-0.333333333333,  -0.333333333333,  0.000000000000]],
             [20, 20, 20, 20, 1],
             ['\u0413', 'M', 'K','\u0413', 'K'])
kernel.load_deeph_mats('./')
kernel.diag(nbnd=8, efermi=None)
kernel.write('./')


with open("eig.dat") as f:
    lines = f.readlines()
nk, nb = map(int, lines[2].split())
band_lines = lines[3:]
kpt = np.array(range(nk)).repeat(nb)
band = np.ones((nk, nb))
for k in range(nk):
    band[k]  = np.array(list(map(lambda x: float(x.split()[-1]), band_lines[k*(nb+1):(k+1)*(nb+1)][1:])))
plt.scatter(kpt.flatten(), band.flatten())
plt.savefig('band.png')

# with open("../aohamiltonian_dft/eig.dat") as f:
#     lines = f.readlines()
# nk, nb = map(int, lines[2].split())
# band_lines = lines[3:]
# kpt = np.array(range(nk)).repeat(nb)
# band = np.ones((nk, nb))
# for k in range(nk):
#     band[k]  = np.array(list(map(lambda x: float(x.split()[-1]), band_lines[k*(nb+1):(k+1)*(nb+1)][1:])))
# plt.scatter(kpt.flatten(), band.flatten())

# plt.ylim(-10, 10)