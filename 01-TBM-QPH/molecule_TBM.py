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
from mpi4py import MPI



def get_angular_projectors(setup, angular, type='bound', xyz_order = True):
    """
    Determine the projector indices which have specified angula
    quantum number.

    angular can be s, p, d, f, or a list of these.
    If type is 'bound', only bound state projectors are considered, otherwise
    all projectors are included.

    The m order of the projector is y, z, x if xyz_order is False (default by GPAW)

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
            # todo:figure out the order of l=2 (d-orbital)
            if 'spdf'[setup.l_j[j]] == 'p' and xyz_order:
                pm = list(range(i, i + m))
                projectors.extend([pm[2], pm[0], pm[1]])
            else:
                # follow y z x by default
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
    self.H_ab_GW = Hamiltonian matrix for GW calculation (comparable to DFT Hamiltonian)
    self.S_ab_GW = Overlap matrix for GW Hamiltonian (comparable to DFT Overlap matrix)
    self.e_n_GW = eigenvalues by GW
    self.eigvals_GW = eigenvalues by GW TBM

    # Useful methods
    self.gpaw2deephe3() to convert the GPAW output to deephe3 input
    self.build_Ham_TBM() (by default) to build the Hamiltonian matrix for TBM
    slef.build_GW_Ham_TBM() to build the Hamiltonian matrix for GW calculation

    self.plot_Ham() to plot the Hamiltonian matrix
    self.plot_ortho() to plot the overlap matrix
    self.calculate_GW_energies() to calculate the GW energies
    
    """    
    def __init__(self, symbol):
        """
        i) Initialize the molecule and calculate the DFT Hamiltonian (GPAW)
        ii) Build the tight-binding Hamiltonian
        iii) Analyse the orbitals
        """
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        print('Rank:', self.rank)
        print('Size:', self.size)


        # -> Parameters
        self.ecut = 100
        self.vacuum = 5
        self.basis = 'dzp'
        self.orbtype = 'spdf'
        self.Hartree2eV = 27.21138602
        self.nbands_gpaw = '600%'
        self.pos_def_shift_for_S_ab = 1E-9 # Force S_ab to be positive definite by add a small diagonal term
        self.rotation = [] # rotation parameter
        self.ecut_gw = 20
        self.near_distance_thres = 20 # [Angstrom] for nearest neighbor
        self.parallel = {'band':self.size}
        self.gpaw_path = 'gpaw_gw'

        #->  Build molecule
        self.symbol = symbol
        self.mol = molecule(symbol, vacuum=self.vacuum)

        self.mol.rotate(0, 'z')

        self.calc = GPAW(mode=PW(ecut=self.ecut, force_complex_dtype=True), 
                         basis=self.basis,
                         nbands=self.nbands_gpaw,
                         parallel=self.parallel,)
        self.mol.calc = self.calc

        # -> DFT calculation
        self.mol.get_potential_energy()
        self.wfs = self.calc.wfs
        self.nb = self.wfs.bd.nbands # number of eigenstates in DFT
        self.nk = len(self.wfs.kd.weight_k)
        self.nvalence_band = self.calc.get_number_of_electrons() // 2
        self.fermi_energy = self.calc.get_fermi_level()
        self.natom = len(self.wfs.setups)

        # -> Collect orbital LDOS
        self.setup = [setup for setup in self.wfs.setups]
        self.weights = []
        self.orbitals = {} # {atom1:{l=0:n1, l=1:n2}, atom2:{}...}
        self.norbitals = [] # [norb_atom1, norb_atom2, ...]
        self.e_n = None

        self.collect_c_ni()
        self.n_overlap_a = np.concatenate(self.weights, axis=1)
        self.delta_nm = None

        # -> Build DFT Hamiltonian matrix
        self.H_ab = None
        self.S_ab = None
        self.eigvals = None
        self.eigvecs = None
        self.build_Ham_TBM() # by default, we calculate the DFT Hamiltonian
        self.analyse_orbitals()

        # print('DFT energies:', self.e_n)

        # -> GW calculation related
        self.gw = None
        self.gw_result = None
        self.e_n_GW = None
        self.H_ab_GW = None
        self.S_ab_GW = None

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
        # if np.all(np.real(np.linalg.eigvals(self.S_ab)) > 0):
        # if None:
        #     parprint('Overlap matrix is positive definite')
        #     parprint('The min eigenvalue of S_ab:', np.linalg.eigvals(self.S_ab).min())
        #     self.S_ab_posdef = self.S_ab
        #     pass
        # force a small shift to make S_ab positive definite
        parprint('A shift %s is applied to force S matrix postive definite'%self.pos_def_shift_for_S_ab)
        parprint('The min eigenvalue of S_ab:', np.real(np.linalg.eigvals(self.S_ab)).min())
        self.S_ab_posdef = self.S_ab + np.eye(self.S_ab.shape[0]) * self.pos_def_shift_for_S_ab

        self.eigvals, self.eigvecs = sp.linalg.eigh(self.H_ab, self.S_ab_posdef)
        parprint('R2 score of energies:', r2_score(self.e_n[:min(np.min(len(self.e_n)), np.min(len(self.eigvals)))], 
                                                self.eigvals[:min(np.min(len(self.e_n)), np.min(len(self.eigvals)))]))


    def build_GW_Ham_TBM(self):
        if self.gw is None:
            parprint('GW calculation is not done yet')
            parprint('Calculate GW energies')
            self.calculate_GW_energies()
            parprint('GW calculation is done')
            parprint('QP energies:', self.gw_result['qp'])
            # assert (self.gw_result['qp'].squeeze() == self.e_n_GW).all()
        
        if self.norbitals == []:
            raise ValueError('Orbitals are not collected yet')
        
        self.H_ab_GW = np.einsum("an, nb, n -> ab", self.n_overlap_a.conj().transpose(), self.n_overlap_a, self.e_n_GW)
        self.S_ab_GW = np.einsum("an, nb -> ab", self.n_overlap_a.conj().transpose(), self.n_overlap_a)
        
        # force a small shift to make S_ab positive definite               
        parprint('A shift %s is applied to force S matrix postive definite'%self.pos_def_shift_for_S_ab)
        parprint('The min eigenvalue of S_ab_GW:', np.real(np.linalg.eigvals(self.S_ab_GW)).min())
        self.S_ab_GW_posdef = self.S_ab_GW + np.eye(self.S_ab_GW.shape[0]) * self.pos_def_shift_for_S_ab

        self.eigvals_GW, self.eigvecs_GW = sp.linalg.eigh(self.H_ab_GW, self.S_ab_GW_posdef)
        parprint('R2 score of energies:', r2_score(self.e_n_GW[:min(np.min(len(self.e_n_GW)), np.min(len(self.eigvals_GW)))], 
                                                self.eigvals_GW[:min(np.min(len(self.e_n_GW)), np.min(len(self.eigvals_GW)))]))

    def partition_Ham_TBM_for_Aij(self, atom_i=0, atom_j=0, Ham_type='DFT'):
        if self.mol.get_distance(atom_i, atom_j) > self.near_distance_thres:
            return None, None
        
        if Ham_type == 'DFT':
            assert self.H_ab is not None
                # parprint('Hamiltonian matrix is not built yet')
                # parprint('Building Hamiltonian matrix...')
                # self.build_Ham_TBM()
            Ham_ab = self.H_ab
        elif Ham_type == 'GW':
            assert self.H_ab_GW is not None
                # parprint('GW Hamiltonian matrix is not built yet')
                # parprint('Building GW Hamiltonian matrix...')
                # self.build_GW_Ham_TBM()
            Ham_ab = self.H_ab_GW

        # It seems atom_i in e3nn starts from 1
        key = [0, 0, 0, atom_i+1, atom_j+1] # [Rx, Ry, Rz, atom_i, atom_j] in hamiltonian.h5 
        # partition the Hamiltonian matrix by atom_i and atom_j
        atom_i_index_min = sum(self.norbitals[:atom_i])
        atom_i_index_max = sum(self.norbitals[:atom_i+1])
        atom_j_index_min = sum(self.norbitals[:atom_j])
        atom_j_index_max = sum(self.norbitals[:atom_j+1])
        # if self.H_ab is None:
        #     parprint('Hamiltonian matrix is not built yet')
        #     parprint('Building Hamiltonian matrix...')
        #     self.build_Ham_TBM()
        H_ia_jb = Ham_ab[atom_i_index_min:atom_i_index_max, atom_j_index_min:atom_j_index_max]

        # visualize the Hamiltonian matrix (verify the partition)
        # plt.imshow(np.real(H_ia_jb), cmap='RdBu_r', vmin=-np.real(abs(self.H_ab)).max()*0.6, vmax=np.real(abs(self.H_ab)).max()*0.6)

        return key, H_ia_jb
    
    def plot_delta_nm(self, type='ab'):
        """
        type = 'nm' or 'ab'

        "nm":
        Sum_a <n|a><a|m> = delta_nm

        "ab":
        Sum_n <a|n><n|b> = delta_ab

        "na::
        <n|a>
        """
        assert type in ['nm', 'ab', 'na']

        if self.rank == 0:
            plt.figure(figsize=(24, 12))
            if type == 'nm':
                self.delta_nm = np.einsum("na, ma -> nm", self.n_overlap_a, self.n_overlap_a.conj())
                vmin = -np.real(abs(self.delta_nm)).max()*1.0
                vmax = np.real(abs(self.delta_nm)).max()*1.0
                plt.xlabel('n')
                plt.ylabel('m')
                plt.title(r'<n|m>')
                plt.imshow(np.abs(self.delta_nm), cmap='RdBu_r',vmin=vmin, vmax=vmax)
                plt.colorbar(shrink=0.7)
                plt.show()
            elif type == 'ab':
                self.delta_ab = np.einsum("na, nb -> ab", self.n_overlap_a, self.n_overlap_a.conj())
                vmin = -np.real(abs(self.delta_ab)).max()*1.0
                vmax = np.real(abs(self.delta_ab)).max()*1.0
                plt.xlabel('a')
                plt.ylabel('b')
                plt.title(r'<a|b>')
                plt.imshow(np.abs(self.delta_ab), cmap='RdBu_r',vmin=vmin, vmax=vmax)
                plt.colorbar(shrink=0.7)
                plt.show()
            elif type == 'na':
                vmin = -np.real(abs(self.n_overlap_a)).max()*1.0
                vmax = np.real(abs(self.n_overlap_a)).max()*1.0
                plt.xlabel('Orbital Index - a')
                plt.xticks(range(self.n_overlap_a.shape[1]))
                plt.ylabel('Energy index - n')
                plt.yticks(range(self.n_overlap_a.shape[0]))
                plt.title(r'<n|a>')
                plt.imshow(np.abs(self.n_overlap_a), cmap='RdBu_r',vmin=vmin, vmax=vmax)
                plt.colorbar(shrink=0.7)
                plt.show()

    
    def plot_Ham_TBM(self, path='./', Ham_type='DFT'):
        """
        Characterize the Hamiltonian matrix by atomic orbitals
        Ham_type: 'DFT' or 'GW'
        """
        assert Ham_type in ['DFT', 'GW']
        if self.rank == 0: # only master node plot the figure
            if Ham_type == 'DFT':
                assert self.H_ab is not None
                r2_dft = r2_score(self.e_n[:min(np.min(len(self.e_n)), np.min(len(self.eigvals)))], 
                                                    self.eigvals[:min(np.min(len(self.e_n)), np.min(len(self.eigvals)))])
                vmin = -np.real(abs(self.H_ab)).max()*1.0
                vmax = np.real(abs(self.H_ab)).max()*1.0
                        
            elif Ham_type == 'GW':
                assert self.H_ab is not None
                assert self.H_ab_GW is not None
                r2_dft = r2_score(self.e_n[:min(np.min(len(self.e_n)), np.min(len(self.eigvals)))], 
                                        self.eigvals[:min(np.min(len(self.e_n)), np.min(len(self.eigvals)))])
                r2_gw = r2_score(self.e_n_GW[:min(np.min(len(self.e_n_GW)), np.min(len(self.eigvals_GW)))], 
                                        self.eigvals_GW[:min(np.min(len(self.e_n_GW)), np.min(len(self.eigvals_GW)))])
                vmin = -np.real(abs(self.H_ab_GW)).max()*1.0
                vmax = np.real(abs(self.H_ab_GW)).max()*1.0

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

            if Ham_type == 'DFT': # only plot DFT Hamiltonian
                plt.imshow(np.real(self.H_ab), cmap='RdBu_r', vmin=vmin, vmax=vmax)
                plt.xticks(range(len(x_label)), x_label, rotation=-90)
                # plt.gcf().autofmt_xdate()
                # plt.gcf().autofmt_ydate()
                plt.gca().xaxis.set_ticks_position('top')
                plt.gca().xaxis.set_label_position('top')
                plt.yticks(range(len(y_label)), y_label)
                plt.title(Ham_type + ' Hamiltonian matrix, Fitting '+ r"$R^2$=%.2f"%r2_dft)
                cbar = plt.colorbar()
                cbar.set_label(r'$|H_{\alpha, \beta}|$ (eV)', labelpad=-20, y=1.1, rotation=0)
                # save figure
                # Adjust layout
                plt.tight_layout()
                plt.savefig(path + Ham_type +'_Hamiltonian_matrix.png')
                plt.show()
            elif Ham_type == 'GW': # plot both DFT and GW Hamiltonian
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                im0 = ax[0].imshow(np.real(self.H_ab), cmap='RdBu_r', vmin=vmin, vmax=vmax)
                ax[0].set_xticks(range(len(x_label)))
                ax[0].set_xticklabels(x_label, rotation=-90)
                ax[0].xaxis.set_ticks_position('top')
                ax[0].xaxis.set_label_position('top')
                ax[0].set_yticks(range(len(y_label)))
                ax[0].set_yticklabels(y_label)
                ax[0].set_title('DFT Hamiltonian matrix, Fitting '+ r"$R^2$=%.2f"%r2_dft)
                # cbar = ax[0].figure.colorbar(im0)
                # cbar.set_label(r'$|H_{\alpha, \beta}|$ (eV)', labelpad=-20, y=1.1, rotation=0)
                fig.colorbar(im0, ax=ax[0], shrink=0.7)

                im1 = ax[1].imshow(np.real(self.H_ab_GW), cmap='RdBu_r', vmin=vmin, vmax=vmax)
                ax[1].imshow(np.real(self.H_ab_GW), cmap='RdBu_r', vmin=vmin, vmax=vmax)
                ax[1].set_xticks(range(len(x_label)))
                ax[1].set_xticklabels(x_label, rotation=-90)
                ax[1].xaxis.set_ticks_position('top')
                ax[1].xaxis.set_label_position('top')
                ax[1].set_yticks(range(len(y_label)))
                ax[1].set_yticklabels(y_label)
                ax[1].set_title('GW Hamiltonian matrix, Fitting '+ r"$R^2$=%.2f"%r2_gw)
                fig.colorbar(im1, ax=ax[1], shrink=0.7)
                # cbar = ax[1].figure.colorbar(im1)
                # cbar.set_label(r'$|H_{\alpha, \beta}|$ (eV)', labelpad=-20, y=1.1, rotation=0)
                # save figure
                # Adjust layout
                plt.tight_layout()
                plt.savefig(path + Ham_type +'_Hamiltonian_matrix.png')
                plt.show()
        else:
            pass


    def calculate_GW_energies(self):
        """
        Calculate the GW energies
        TODO: debug this
        """

        # put calculated result to gpaw_gw dir
        gpaw_path = self.gpaw_path
        if not os.path.exists(gpaw_path) and self.rank == 0:
            os.makedirs(gpaw_path)
        
        start_time_gw = time.time()
        #if '%s_fulldiag.gpw'%self.symbol exists, use it
        #else calculate it

        # sync the calculation
        self.comm.Barrier()

        if gpaw_path+'/%s_fulldiag.gpw'%self.symbol in os.listdir():
            parprint('Use existing full diagonalized Hamiltonian')
        else:
            parprint('Calculate full diagonalized Hamiltonian')
            self.calc.diagonalize_full_hamiltonian(nbands=self.nb*20)
            self.calc.write(gpaw_path+'/%s_fulldiag.gpw'%self.symbol, 'all')

        self.gw = G0W0(calc=gpaw_path+'/%s_fulldiag.gpw'%self.symbol,
                  ecut=self.ecut_gw,
                  filename=gpaw_path+'/'+f'{self.symbol}_g0w0_{self.ecut_gw}',
                #   truncation='0D',
                #   ppa=True,
                  bands=(0, self.nb), # make it consistent with the number of bands in DFT
                  nbands=self.nb*10)
        self.gw_result = self.gw.calculate()
        self.e_n_GW = self.gw_result['qp'][0,0,:] # todo (all spin) here is only spin=0

        end_time_gw = time.time()
        parprint(f'GW step completed in {end_time_gw - start_time_gw:.2f} seconds')
    
    def gpaw2deephe3(self, filename='./deeph3_raw_data', Ham_type='DFT'):
        """
        Convert the GPAW output to deehe3 input

        output:
        1. element.dat: atomic number of each element in the system [H=1, He=2, ...]
        2. hamiltonian.h5: 
            - [Rx, Ry, Rz, atom_i, atom_j] -> H_ia_jb (TBM Hamiltonian)
            - Rx, Ry, Rz: near unit cell, [0,0,0] fro molecule
        3. info.json: {"fermi_level": float, "isspinful":False}
        4. lat.dat: lattice vectors (3x3)
        5. orbital_types.dat [0, 0, ..., 1, 1, ..., 2, 2, ...]
            - 0: s, 1: p, 2: d, 3: f
            - example: 0 0 0 1 1 2 2 -> s3p2d2
        6. R_list.dat: list of R vectors (It seems not neccesary)
        7. site_positions.dat: atomic positions (natoms x 3)                
        """


        # Tight-binding model for Hamiltonian is built by default, no need to check that.
        if Ham_type == 'GW':
            self.build_GW_Ham_TBM()

        if self.rank == 0:

            # 0. Check if the file exists
            self.path_g2d = filename + '/' + self.symbol + '/'
            # path = filename + '/' + self.symbol + '/'
            if not os.path.exists(self.path_g2d):
                os.makedirs(self.path_g2d)

            # 1. element.dat
            np.savetxt(self.path_g2d + 'element.dat', self.mol.get_atomic_numbers().astype(int), fmt='%d')

            # 2. hamiltonian.h5
            # Build TBM Hamiltonian (GW/BSE)
            #  - calculate DFT or GW
            #  - projection to atomic orbitals
            # Write the Hamiltonian matrix to an h5 file
            #  - partition the Hamiltonian matrix by atom_i and atom_j
            h_cnt = 0
            with h5.File(self.path_g2d + 'hamiltonians.h5', 'w') as f:
                for i in range(self.natom):
                    for j in range(self.natom):
                        key, H_ia_jb = self.partition_Ham_TBM_for_Aij(i, j, Ham_type=Ham_type)
                        if key == None:
                            # skip if the partition is empty (no nearest neighbor)
                            continue
                        f.create_dataset(str(key), data=H_ia_jb)
                        h_cnt += 1
            parprint(f'{h_cnt} / {self.natom**2} sub-Hamiltonian matrices are saved')

            # 3. info.json
            info = {"fermi_level": self.fermi_energy if self.fermi_energy != float('inf') else self.e_n[-1]+0.001, "isspinful":False}
            with open(self.path_g2d + 'info.json', 'w') as f:
                json.dump(info, f, indent=4)

            # 4. lat.dat (Molecule system)
            np.savetxt(self.path_g2d + 'lat.dat', mol.mol.cell[:], fmt='%.16f')

            # 5. orbital_type.dat
            with open(self.path_g2d + 'orbital_types.dat', 'w') as f:
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

            # 8. Ham_plot
            self.plot_Ham_TBM(path=self.path_g2d, Ham_type=Ham_type)

        else:
            pass

if __name__ == "__main__":
    mol = Molecule_TBM('CH3CH2OH')
    mol.build_Ham_TBM()
    mol.plot_Ham_TBM(Ham_type='DFT')
    # mol.build_GW_Ham_TBM()
    # mol.plot_Ham_TBM(Ham_type='GW')
    # mol.gpaw2deephe3(filename='./deeph3_raw_data_GW', Ham_type='GW')
    
    # Generate g2 dataset
    # g2.names[:]
    # CH_dataset = ['C2H2','C2H3', 'C2H4','C2H5','C2H6','C3H8',
    # 'C3H9C','C6H6','CH','CH4']
    # CH_dataset = [ 'H2', 'C', 'C3H7', 'CH3',  'H',  'CCH', 'C5H8', ]

    # for idx, name in enumerate(CH_dataset[5:]):
    #     mol = Molecule_TBM(name)
    #     mol.gpaw2deephe3(filename='./deeph3_raw_data_GW', Ham_type='GW')
