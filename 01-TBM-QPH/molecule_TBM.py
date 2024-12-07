import time
from gpaw.response.g0w0 import G0W0
from ase.build import mx2
from gpaw import GPAW, PW, FermiDirac
from ase.parallel import parprint
import numpy as np
import matplotlib.pyplot as plt
from ase.build import molecule
import scipy as sp
from sklearn.metrics import r2_score

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



class Molecule_TBM():
    """
    # Useful attributes
    self.orbitals = {atom1:{l=0:n1, l=1:n2}, atom2:{}...}
    self.norbitals = [norb_atom1, norb_atom2, ...]
    self.e_n = eigenvalues by DFT
    self.eigvals = eigenvalues by TBM
    self.H_ab = Hamiltonian matrix
    self.S_ab = Overlap matrix

    # Useful methods
    self.plot_ortho() to plot the overlap matrix
    self.plot_Ham() to plot the Hamiltonian matrix

    """    
    def __init__(self, symbol):
        """
        i) Initialize the molecule and calculate the DFT Hamiltonian (GPAW)
        ii) Build the tight-binding Hamiltonian
        iii) Analyse the orbitals
        """
        # Parameters
        self.ecut = 300
        self.vacuum = 5
        self.basis = 'dzp'
        self.orbtype = 'spdf'
        self.Hartree2eV = 27.21138602

        # Build molecule
        self.symbol = symbol
        self.mol = molecule(symbol, vacuum=self.vacuum)
        self.calc = GPAW(mode=PW(ecut=self.ecut), basis=self.basis)
        self.mol.calc = self.calc

        # DFT calculation
        self.mol.get_potential_energy()
        self.wfs = self.calc.wfs
        self.nb = self.wfs.bd.nbands
        self.nk = len(self.wfs.kd.weight_k)
        self.nvalence_band = self.calc.get_number_of_electrons() // 2
        self.fermi_energy = self.calc.get_fermi_level()
        self.natom = len(self.wfs.setups)

        # Collect orbital LDOS
        self.setup = [setup for setup in self.wfs.setups]
        self.weights = []
        self.orbitals = {} # {atom1:{l=0:n1, l=1:n2}, atom2:{}...}
        self.norbitals = [] # [norb_atom1, norb_atom2, ...]
        self.e_n = None

        self.collect_c_ni()
        self.n_overlap_a = np.concatenate(self.weights, axis=1)
        self.delta_nm = None

        self.build_Ham()
        self.analyse_orbitals()

        print('R2 score of energies:', r2_score(self.e_n, self.eigvals))

    def collect_c_ni(self,):
        self.weights = []
        for i in range(self.natom):
            self.e_n, w = raw_orbital_LDOS(self.wfs, i, 0, self.orbtype)
            self.weights.append(w)
            self.norbitals.append(w.shape[1])
        self.e_n = self.e_n * self.Hartree2eV


    def analyse_orbitals(self,):
        self.orbitals = {}
        orb_types = ['s', 'p', 'd', 'f']
        for i in range(self.natom):
            for j, orb_type in enumerate(orb_types):
                self.orbitals[i] = self.orbitals.get(i, {})
                self.orbitals[i][j] = get_angular_projectors(self.setup[i], orb_type)
        
    def build_Ham(self):
        # c_ni = <n|i>, where |n> is eigenvector and |i> is the AO
        # |n> = sum_i c_ni |i>, which is othornomal Bloch basis
        # |i> is not orthonormal. So we need to transform this to a general eigenvalue problem
        self.H_ab = np.einsum("an, nb, n -> ab", self.n_overlap_a.conj().transpose(), self.n_overlap_a, self.e_n)
        self.S_ab = np.einsum("an, nb -> ab", self.n_overlap_a.conj().transpose(), self.n_overlap_a)

        self.eigvals, self.eigvecs = sp.linalg.eigh(self.H_ab, self.S_ab)
    
    def plot_delta_nm(self,):
        """
        Sum_a <n|a><a|m> = delta_nm
        """
        self.delta_nm = np.einsum("na, ma -> nm", self.n_overlap_a, self.n_overlap_a.conj())
        plt.xlabel('n')
        plt.ylabel('m')
        plt.title(r'<n|m>')
        plt.imshow(np.abs(self.delta_nm))
        plt.colorbar()
        plt.show()
    
    def plot_Ham(self,):
        """
        Characterize the Hamiltonian matrix by atomic orbitals
        """
        # I know this is little bit confusing, but it is efficient
        x_label = []
        y_label = []

        for i in range(self.natom):
            for j in range(len(self.orbitals[i])):
                for k in range(len(self.orbitals[i][j])):
                    if j != 0:
                        x_label.append('l=%s'%j)
                        y_label.append('l=%s'%j)
                    else:
                        y_label.append(self.mol[i].symbol+r'$_%s$'%(i+1) + ', l=%s'%j)
                        x_label.append(self.mol[i].symbol+r'$_%s$'%(i+1) + '\nl=%s'%j)
                    pass


        plt.imshow(np.abs(self.H_ab))
        plt.xticks(range(len(x_label)), x_label)
        plt.gca().xaxis.set_ticks_position('top')
        plt.gca().xaxis.set_label_position('top')
        plt.yticks(range(len(y_label)), y_label)
        plt.title('Hamiltonian matrix')
        cbar = plt.colorbar()
        cbar.set_label(r'$|H_{\alpha, \beta}|$ (eV)', labelpad=-20, y=1.05, rotation=0)
        plt.show()




if __name__ == "__main__":
    mol = Molecule_TBM('C6H6')
    mol.plot_Ham()