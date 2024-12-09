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
import os
from ase.collections import g2, s22, dcdft
import json
import h5py as h5

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

    self.calculate_GW_energies() to calculate the GW energies

    """    
    def __init__(self, symbol):
        """
        i) Initialize the molecule and calculate the DFT Hamiltonian (GPAW)
        ii) Build the tight-binding Hamiltonian
        iii) Analyse the orbitals
        """
        # Parameters
        self.ecut = 100
        self.vacuum = 5
        self.basis = 'dzp'
        self.orbtype = 'spdf'
        self.Hartree2eV = 27.21138602
        self.nbands_gpaw = '100%'
        self.pos_def_shift_for_S_ab = 1E-8 # Force S_ab to be positive definite by add a small diagonal term
        self.rotation = [] # rotation parameter
        self.ecut_gw = 50
        self.near_distance_thres = 20 # [Angstrom] for nearest neighbor

        # Build molecule
        self.symbol = symbol
        self.mol = molecule(symbol, vacuum=self.vacuum)

        self.mol.rotate(0, 'z')

        self.calc = GPAW(mode=PW(ecut=self.ecut, force_complex_dtype=True), 
                         basis=self.basis,
                         nbands=self.nbands_gpaw,)
        self.mol.calc = self.calc

        # DFT calculation
        self.mol.get_potential_energy()
        self.wfs = self.calc.wfs
        self.nb = self.wfs.bd.nbands # number of eigenstates in DFT
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

        self.build_Ham_TBM()
        self.analyse_orbitals()

        # self.energy_r2_n_min = min(np.min(len(self.e_n)), np.min(len(self.eigvals)))
        print('R2 score of energies:', r2_score(self.e_n[:min(np.min(len(self.e_n)), np.min(len(self.eigvals)))], 
                                                self.eigvals[:min(np.min(len(self.e_n)), np.min(len(self.eigvals)))]))

        # self.plot_Ham_TBM()

        # GW calculation
        self.gw = None
        self.gw_result = None
        # TODO: finish GW Hamiltonian

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
        
    def build_Ham_TBM(self):
        # c_ni = <n|i>, where |n> is eigenvector and |i> is the AO
        # |n> = sum_i c_ni |i>, which is othornomal Bloch basis
        # |i> is not orthonormal. So we need to transform this to a general eigenvalue problem
        self.H_ab = np.einsum("an, nb, n -> ab", self.n_overlap_a.conj().transpose(), self.n_overlap_a, self.e_n)
        self.S_ab = np.einsum("an, nb -> ab", self.n_overlap_a.conj().transpose(), self.n_overlap_a)
        # Check if self.S_ab is positive definite
        if np.all(np.linalg.eigvals(self.S_ab) > 0):
            print('Overlap matrix is positive definite')
            self.S_ab_posdef = self.S_ab
            pass
        else:
            print('Overlap matrix is not positive definite, add a small diagonal term %s'%self.pos_def_shift_for_S_ab)
            self.S_ab_posdef = self.S_ab + np.eye(self.S_ab.shape[0]) * self.pos_def_shift_for_S_ab

        self.eigvals, self.eigvecs = sp.linalg.eigh(self.H_ab, self.S_ab_posdef)

    def partition_Ham_TBM_for_Aij(self, atom_i=0, atom_j=0):
        if self.mol.get_distance(atom_i, atom_j) > self.near_distance_thres:
            return None, None
        
        key = [0, 0, 0, atom_i, atom_j] # [Rx, Ry, Rz, atom_i, atom_j] in hamiltonian.h5
        # partition the Hamiltonian matrix by atom_i and atom_j
        atom_i_index_min = sum(self.norbitals[:atom_i])
        atom_i_index_max = sum(self.norbitals[:atom_i+1])
        atom_j_index_min = sum(self.norbitals[:atom_j])
        atom_j_index_max = sum(self.norbitals[:atom_j+1])
        if self.H_ab is None:
            print('Hamiltonian matrix is not built yet')
            print('Building Hamiltonian matrix...')
            self.build_Ham_TBM()
        H_ia_jb = self.H_ab[atom_i_index_min:atom_i_index_max, atom_j_index_min:atom_j_index_max]

        # visualize the Hamiltonian matrix (verify the partition)
        # plt.imshow(np.real(H_ia_jb), cmap='RdBu_r', vmin=-np.real(abs(self.H_ab)).max()*0.6, vmax=np.real(abs(self.H_ab)).max()*0.6)

        return key, H_ia_jb
    
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
    
    def plot_Ham_TBM(self, path='./'):
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
                        y_label.append(self.mol[i].symbol+r'$_{%s}$'%(i+1) + ', l=%s'%j)
                        x_label.append(self.mol[i].symbol+r'$_{%s}$'%(i+1) + ', l=%s'%j)
                    pass

                    plt.imshow(np.real(self.H_ab), cmap='RdBu_r', vmin=-np.real(abs(self.H_ab)).max()*0.6, vmax=np.real(abs(self.H_ab)).max()*0.6)
        plt.xticks(range(len(x_label)), x_label, rotation=-90)
        # plt.gcf().autofmt_xdate()
        # plt.gcf().autofmt_ydate()
        plt.gca().xaxis.set_ticks_position('top')
        plt.gca().xaxis.set_label_position('top')
        plt.yticks(range(len(y_label)), y_label)
        plt.title('Hamiltonian matrix, Fitting '+ r"$R^2$=%.2f"%r2_score(self.e_n[:min(np.min(len(self.e_n)), np.min(len(self.eigvals)))], 
                                                self.eigvals[:min(np.min(len(self.e_n)), np.min(len(self.eigvals)))]))
        cbar = plt.colorbar()
        cbar.set_label(r'$|H_{\alpha, \beta}|$ (eV)', labelpad=-20, y=1.1, rotation=0)
        # save figure
        # Adjust layout
        plt.tight_layout()
        plt.savefig(path+'Hamiltonian_matrix.png')
        plt.show()

    def calculate_GW_energies(self):
        """
        Calculate the GW energies
        TODO: debug this
        """
        start_time_gw = time.time()
        #if '%s_fulldiag.gpw'%self.symbol exists, use it
        #else calculate it
        if '%s_fulldiag.gpw'%self.symbol in os.listdir():
            print('Use existing full diagonalized Hamiltonian')
        else:
            print('Calculate full diagonalized Hamiltonian')
            self.calc.diagonalize_full_hamiltonian(nbands=self.nb*20)
            self.calc.write('%s_fulldiag.gpw'%self.symbol, 'all')

        self.gw = G0W0(calc='%s_fulldiag.gpw'%self.symbol,
                  ecut=self.ecut_gw,
                  filename=f'{self.symbol}_g0w0_{self.ecut_gw}',
                #   truncation='0D',
                #   ppa=True,
                  bands=(0,8),
                  nbands=self.nb*10)
        self.gw_result = self.gw.calculate()


        end_time_gw = time.time()
        print(f'GW step completed in {end_time_gw - start_time_gw:.2f} seconds')
    
    def gpaw2deephe3(self, filename='./deeph3_raw_data'):
        """
        Convert the GPAW output to deehe3 input

        output:
        1. element.dat: atomic number of each element in the system [H=1, He=2, ...]
        2. hamiltonian.h5: 
            - [Rx, Ry, Rz, atom_i, atom_j] -> H_ia_jb (TBM Hamiltonian)
            - Rx, Ry, Rz: near unit cell, [0,0,0] fro molecule
        3. info.json: {"fermi_level": float, "isspinful":False}
        4. lat.dat: lattice vectors (3x3)
        5. orbital_type.dat [0, 0, ..., 1, 1, ..., 2, 2, ...]
            - 0: s, 1: p, 2: d, 3: f
            - example: 0 0 0 1 1 2 2 -> s3p2d2
        6. R_list.dat: list of R vectors (It seems not neccesary)
        7. site_positions.dat: atomic positions (natoms x 3)                
        """

        # 0. Check if the file exists
        self.path_g2d = filename + '/' + self.symbol + '/'
        # path = filename + '/' + self.symbol + '/'
        if not os.path.exists(self.path_g2d):
            os.makedirs(self.path_g2d)

        # 1. element.dat
        np.savetxt(self.path_g2d + 'element.dat', self.mol.get_atomic_numbers().astype(int), fmt='%d')

        # 2. hamiltonian.h5
        # write the Hamiltonian matrix to an h5 file
        # Todo: only save Hamiltonian matrix for nearest neighbor
        h_cnt = 0
        with h5.File(self.path_g2d + 'hamiltonian.h5', 'w') as f:
            for i in range(self.natom):
                for j in range(self.natom):
                    key, H_ia_jb = self.partition_Ham_TBM_for_Aij(i, j)
                    if key == None:
                        # skip if the partition is empty (no nearest neighbor)
                        continue
                    f.create_dataset(str(key), data=H_ia_jb)
                    h_cnt += 1
        print(f'{h_cnt} / {self.natom**2} sub-Hamiltonian matrices are saved')

        # 3. info.json
        info = {"fermi_level": self.fermi_energy if self.fermi_energy != float('inf') else 'inf', "isspinful":False}
        with open(self.path_g2d + 'info.json', 'w') as f:
            json.dump(info, f, indent=4)

        # 4. lat.dat (Molecule system)
        np.savetxt(self.path_g2d + 'lat.dat', mol.mol.cell[:], fmt='%.16f')

        # 5. orbital_type.dat
        with open(self.path_g2d + 'orbital_type.dat', 'w') as f:
            for i in range(self.natom):
                atom_orb = []
                for j in range(len(self.orbitals[i])):
                    num_orb = len(self.orbitals[i][j])
                    if num_orb % (j*2+1) != 0:
                        raise ValueError('Fractional orbital number is not allowed, atom%s, l=%s'%(i, j))
                    atom_orb = atom_orb + [j]* (num_orb // (j*2+1))
                f.write(' '.join(map(str, atom_orb)) + '\n')

        # 6. R_list.dat (not neccessary)
        # Not implemented yet

        # 7. site_positions.dat
        np.savetxt(self.path_g2d + 'site_positions.dat', mol.mol.positions.T, fmt='%.16f')

        pass


if __name__ == "__main__":
    # Generate g2 dataset
    # for name in g2.names[54:]:
    #     # print(name)
    #     mol = Molecule_TBM(name)
    #     mol.gpaw2deephe3()
    #     mol.plot_Ham_TBM(mol.path_g2d)

    mol = Molecule_TBM('C6H6')
    mol.plot_Ham_TBM()
    mol.gpaw2deephe3()
    # mol.calculate_GW_energies()

    # Make a demo in group meeting
    # TODO: finish this
