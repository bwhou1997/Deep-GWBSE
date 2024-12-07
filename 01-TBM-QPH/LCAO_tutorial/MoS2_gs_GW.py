import time
from gpaw.response.g0w0 import G0W0
from ase.build import mx2
from gpaw import GPAW, PW, FermiDirac
from ase.parallel import parprint
import numpy as np
import matplotlib.pyplot as plt

def get_angular_projectors(setup, angular, type='bound'):
    """Determine the projector indices which have specified angula
    quantum number.

    angular can be s, p, d, f, or a list of these.
    If type is 'bound', only bound state projectors are considered, otherwise
    all projectors are included.
    """
    # Get the number of relevant j values
    if type == 'bound':
        nj = len([n for n in setup.n_j if n >= 0])
    else:
        nj = len(setup.n_j)

    # Choose the relevant projectors
    projectors = []
    i = 0
    j = 0
    for j in range(nj):
        m = 2 * setup.l_j[j] + 1
        if 'spdf'[setup.l_j[j]] in angular:
            projectors.extend(range(i, i + m))
        j += 1
        i += m

    return projectors


def raw_orbital_LDOS(wfs, a, spin, angular='spdf', nbands=None):
    """Return a list of eigenvalues, and their weight on the specified atom.

    angular can be s, p, d, f, or a list of these.
    If angular is None, the raw weight for each projector is returned.

    An integer value for ``angular`` can also be used to specify a specific
    projector function.

    Setting nbands limits the number of bands included. This speeds up the
    calculation if one has many bands in the calculator but is only interested
    in the DOS at low energies.
    """
    w_k = wfs.kd.weight_k
    nk = len(w_k)
    if not nbands:
        nb = wfs.bd.nbands
    else:
        nb = nbands
        assert nb <= wfs.bd.nbands, ('nbands higher than available number' +
                                     'of bands')

    if a < 0:
        # Allow list-style negative indices; we'll need the positive a for the
        # dictionary lookup later
        a = len(wfs.setups) + a

    I1 = sum(setup.ni for setup in wfs.setups[:a])
    setup = wfs.setups[a]
    I2 = I1 + setup.ni

    energies = np.empty(nb * nk)
    weights_xi = np.empty((nb * nk, setup.ni), dtype=complex)
    x = 0
    for k, w in enumerate(w_k):
        eps_n = wfs.collect_eigenvalues(k=k, s=spin)
        if len(eps_n) > 0:
            energies[x:x + nb] = eps_n[:nb]
        P_nI = wfs.collect_projections(k, spin)
        if P_nI is not None:
            # weights_xi[x:x + nb, :] = w * abs(P_nI[:nb, I1:I2])**2
            weights_xi[x:x + nb, :] = P_nI[:nb, I1:I2]
        x += nb

    wfs.world.broadcast(energies, 0)
    wfs.world.broadcast(weights_xi, 0)

    if angular is None:
        return energies, weights_xi
    elif isinstance(angular, int):
        return energies, weights_xi[:, angular]
    else:
        projectors = get_angular_projectors(setup, angular, type='bound')
        # weights = np.sum(np.take(weights_xi,
                                #  indices=projectors, axis=1), axis=1)
        weights = np.take(weights_xi,
                                 indices=projectors, axis=1)
        return energies, weights


Hartree2eV = 27.21138602


structure = mx2(formula='MoS2', kind='2H', a=3.184, thickness=3.127,
                size=(1, 1, 1), vacuum=3.5)



print('start GS')
start_time_gs = time.time()

calc = GPAW(mode=PW(ecut=300),
            # parallel={'domain': 1},
            xc='PBE',
            basis='dzp',
            kpts={'size': (6, 6, 1), 'gamma': True},
            occupations=FermiDirac(0.01),
            txt='MoS2_out_gs.txt',
            # parallel={'kpt':16,'band':16}
            )

structure.calc = calc
structure.get_potential_energy()
# calc.write('MoS2_gs.gpw', 'all')

calc.diagonalize_full_hamiltonian()
# calc.write('MoS2_fulldiag.gpw', 'all')



# e, P = calc.get_orbital_ldos(a=0, angular='d')
# eig_k = calc.get_eigenvalues(kpt=0)
# setups = calc.wfs.setups # setup for each atom


###### Build Ham #######

wfs = calc.wfs

nb =wfs.bd.nbands
nk = len(wfs.kd.weight_k)
nvalence_band = calc.get_number_of_electrons() // 2
fermi_energy = calc.get_fermi_level()

# shape(nk, nb, -1)
_, weights_Mo = raw_orbital_LDOS(wfs, 0, 0, 'spd')
_, weights_S1 = raw_orbital_LDOS(wfs, 1, 0, 'spd')
energies, weights_S2 = raw_orbital_LDOS(wfs, 2, 0, 'spd')

energies = energies * Hartree2eV

kn_overlap_a = np.concatenate((weights_Mo, weights_S1, weights_S2), axis=1)

# only select Gamma point for debugging:
kn_overlap_a = kn_overlap_a.reshape(nk, nb, -1)[0, :int(nvalence_band)*2, :]
energies = energies.reshape(nk, nb)[0, :int(nvalence_band)*2]

Ham_unit_cell = np.einsum("na, nb, n -> ab", kn_overlap_a.conj(), kn_overlap_a, energies)

# repeat the Ham_unit_cell to by 3x3
# Ham = np.kron(np.ones((24, 24)), Ham_unit_cell)

# get eigenvalues of ham
# eig_Ham = np.linalg.eigvalsh(Ham)
eig_Ham = np.linalg.eigvalsh(Ham_unit_cell)

###############################










# print('start GW')
# start_time_gw = time.time()

# for ecut in [80]:
#     gw = G0W0(calc='MoS2_fulldiag.gpw',
#               bands=(8, 18),
#               ecut=ecut,
#               truncation='2D',
#               nblocksmax=True,
#               q0_correction=True,
#               filename=f'MoS2_g0w0_{ecut}')

#     result = gw.calculate()

# end_time_gw = time.time()
# print(f'GW step completed in {end_time_gw - start_time_gw:.2f} seconds')




