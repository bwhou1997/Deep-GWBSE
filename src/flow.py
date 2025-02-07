from __future__ import print_function

from os.path import join as pjoin
from from_bgwpy.config import flavors
from from_bgwpy.config import is_dft_flavor_espresso, check_dft_flavor
from from_bgwpy.external import Structure
from from_bgwpy.core import Workflow
from from_bgwpy.BGW import EpsilonTask, SigmaTask
from from_bgwpy.QE import QeScfTask, QeBgwFlow, Qe2BgwTask, QeWfnTask
from from_bgwpy.DFT import WfnBgwFlow
from ase import Atoms
import ase.io
import subprocess
from fptask import AobasisTask, HPROTask, PseudoBandTask, QeBgwFlow_NNS, EpsilonTask_NNS, SigmaTask_NNS, nns_helper_epsilon, ParaBandTask

from config import fp_config
import re



"""This file is modified from BGWpy"""
class DFT_GW_HPRO_Flow(Workflow):
    """
    A one-shot GW workflow made of the following tasks:
        - DFT charge density, wavefunctions and eigenvalues
        - Dielectric Matrix (Epsilon and Epsilon^-1)
        - Self-energy (Sigma)
    """

    def __init__(self, **kwargs):
        """
        Keyword arguments

        General:
        -----------------
        dirname : str
            Directory in which the files are written and the code is executed.
            Will be created if needed.
        stru_file : str
            structure file of crystal
        prefix : str
            Prefix required by QE as a rootname.

        QE:
        -----------------
        dft_flavor : 'espresso' 
            Choice of DFT code for density and wavefunctions calculations.
        pseudo_dir : str
            Directory in which pseudopotential files are found.
        pseudos : list, str
            Pseudopotential files.
        ngkpt : list(3), float
            K-points grid. Number of k-points along each primitive vector
            of the reciprocal lattice.
        kshift : list(3), float, optional
            Relative shift of the k-points grid along each direction,
            as a fraction of the smallest division along that direction.
        qshift : list(3), float
            Q-point used to treat the Gamma point.
        nbnd : int
            Number of bands to be computed.
        ecutwfc : float
            Energy cutoff for the wavefunctions
        ecuteps : float
            Energy cutoff for the dielectric function.
        ibnd_min : int
            Minimum band index for GW corrections.
        ibnd_max : int
            Maximum band index for GW corrections.

        SIESTA:
        -----------------
        basis_set_siesta : str
            Basis precision set (single or double Zeta)
        mesh_cutoff_siesta : float (Ry)
            xxx
        dm_tolerance_siesta : float           
            Self consistent calculation tolerance
        
        Optional:
        -----------------        
        truncation_flag : str, optional
            Which truncation flag to use in BerkeleyGW, e.g. "cell_slab_truncation".
        sigma_kpts : list of list(3), optional
            K-points to evaluate self-energy operator. Defaults to all
            k-points defined by the Monkhorst-Pack grid ngkpt.
        epsilon_extra_lines : list, optional
            Any other lines that should appear in the epsilon input file.
        epsilon_extra_variables : dict, optional
            Any other variables that should be declared in the epsilon input file.
        sigma_extra_lines : list, optional
            Any other lines that should appear in the sigma input file.
        sigma_extra_variables : dict, optional
            Any other variables that should be declared in the sigma input file.

        pseudobands : bool (default is True)
            do pseudobands

        max_scf_iter_siesta : int
            Max SCF Iteration steps            
        """
        # add "structure" to kwargs (historic reason)
        kwargs.update({'structure':Structure.from_file(kwargs['stru_file'])})

        super(DFT_GW_HPRO_Flow, self).__init__(**kwargs)

        kwargs.pop('dirname', None)

        self.structure = kwargs['structure']
        self.atoms = ase.io.read(kwargs['stru_file'])

        self.ngkpt = kwargs.pop('ngkpt')
        self.kshift = kwargs.pop('kshift', [.0,.0,.0])
        self.qshift = kwargs.pop('qshift', [.0,.0,.0])

        nband_aliases = ('nbnd', 'nband')
        for key in nband_aliases:
            if key in kwargs:
                self.nbnd = kwargs.pop(key)
                break
        else:
            raise Exception(
            'Number of bands must be specified with one of these keywords: {}.'
            .format(nband_aliases))

        self.dft_flavor = check_dft_flavor(kwargs.get('dft_flavor', flavors['dft_flavor']))

        # ==== Check Pseudobands ==== #
        pseudos_z_valence = check_pseudo(pseudo_dir=kwargs['pseudo_dir'], pseudos=kwargs['pseudos'])
        print(pseudos_z_valence)
        self.n_z_valence = 0
        for atom_ele in self.atoms.get_chemical_symbols():
            pp = '.'.join([atom_ele] + (kwargs['pseudos'][0].split('.'))[1:])
            if pseudos_z_valence.get(pp, None) != None:
                self.n_z_valence += pseudos_z_valence[pp]
            else:
                print(f'warning: upf is not found for {pp}')
        if True: # consider SOC in the future
            self.n_z_valence = int(self.n_z_valence // 2)
            self.n_z_valence = None if self.n_z_valence == 0 else self.n_z_valence

        # update band_index_min & band_index_max
        assert self.n_z_valence != None
        if kwargs.get('nvbnd_sigma',None):
            kwargs.update({'ibnd_min': max(1, self.n_z_valence - kwargs.get('nvbnd_sigma',2))})
            kwargs.update({'ibnd_max': self.n_z_valence + kwargs.get('ncbnd_sigma',2)})

        # ==== DFT calculations ==== #

        # Quantum Espresso flavor
        assert is_dft_flavor_espresso(self.dft_flavor), "Only Quantum Espresso is supported for DFT calculations."
        fnames = self.make_dft_tasks_espresso(**kwargs)
        kwargs.update(fnames)
        
        # ==== Aobasis(SIESTA) ==== #
        self.aobasis_task = AobasisTask(
             dirname = pjoin(self.dirname, '07-aobasis'),
             **kwargs)
        self.add_task(self.aobasis_task)
        kwargs.update(dict(aobasis_dirname=self.aobasis_task.dirname))

        # ==== GW calculations ==== #

        # Set some common variables for Epsilon and Sigma
        self.epsilon_extra_lines = kwargs.pop('epsilon_extra_lines', [])
        self.epsilon_extra_variables = kwargs.pop('epsilon_extra_variables',{})
        
        self.sigma_extra_lines = kwargs.pop('sigma_extra_lines', [])
        self.sigma_extra_variables = kwargs.pop('sigma_extra_variables', {})
        
        # Dielectric matrix computation and inversion (epsilon)
        self.epsilontask = EpsilonTask(
            dirname = pjoin(self.dirname, '11-epsilon'),
            ngkpt = self.ngkpt,
            qshift = self.qshift,
            extra_lines = self.epsilon_extra_lines,
            extra_variables = self.epsilon_extra_variables,
            **kwargs)

        self.nns_helper_epsilon_task = nns_helper_epsilon(dirname=pjoin(self.dirname, '12-epsilon-nns'), **kwargs)


        self.epsilontask_nns = EpsilonTask_NNS(
            dirname = pjoin(self.dirname, '12-epsilon-nns'),
            ngkpt = self.ngkpt,
            qshift = self.qshift,
            extra_lines = self.epsilon_extra_lines,
            extra_variables = self.epsilon_extra_variables,
            **kwargs)
        kwargs.update(dict(eps0_nns_dir=pjoin(self.dirname, '12-epsilon-nns')))

        # Self-energy calculation (sigma)
        self.sigmatask = SigmaTask_NNS(
            dirname = pjoin(self.dirname, '13-sigma'),
            ngkpt = self.ngkpt,
            extra_lines = self.sigma_extra_lines,
            extra_variables = self.sigma_extra_variables,
            eps0mat_fname = self.epsilontask.eps0mat_fname,
            epsmat_fname = self.epsilontask.epsmat_fname,
            **kwargs)
        
        # Add tasks to the workflow
        self.add_tasks([self.epsilontask, 
                        self.nns_helper_epsilon_task,
                        self.epsilontask_nns ,
                        self.sigmatask], merge=False)

        self.truncation_flag = kwargs.get('truncation_flag')
        self.sigma_kpts = kwargs.get('sigma_kpts')

        # ==== SIESTA/HPRO ==========
        self.hpro_task = HPROTask(
            dirname = pjoin(self.dirname, '16-reconstruction'),
             **kwargs)
        self.add_task(self.hpro_task)


    @property
    def has_kshift(self):
        return any([i!=0 for i in self.kshift])

    @property
    def sigma_kpts(self):
        return self.sigmatask.input.kpts

    @sigma_kpts.setter
    def sigma_kpts(self, value):
        if value:
            self.sigmatask.input.kpts = value

    _truncation_flag = ''
    @property
    def truncation_flag(self):
        return self._truncation_flag

    @truncation_flag.setter
    def truncation_flag(self, value):

        for task in (self.epsilontask, self.sigmatask):

            # Remove old value
            if self._truncation_flag in task.input.keywords:
                i = task.input.keywords.index(self._truncation_flag)
                del task.input.keywords[i]

            # Add new value
            if value:
                task.input.keywords.append(value)

        self._truncation_flag = value

    def make_dft_tasks_espresso(self, **kwargs):
        """
        Initialize all DFT tasks using Quantum Espresso.
        Return a dictionary of file names.
        """

        if 'charge_density_fname' in kwargs:
            if 'data_file_fname' not in kwargs:
                raise Exception("Error, when providing charge_density_fname, data_file_fname is required.")

        else:
            self.scftask = QeScfTask(
                dirname = pjoin(self.dirname, '01-density'),
                ngkpt = self.ngkpt,
                kshift = self.kshift,
                **kwargs)

            self.add_task(self.scftask)

            # Add a scf2bgw task for scf (HPRO)
            self.scf2bgwtask = Qe2BgwTask(
                dirname = self.scftask.dirname,
                ngkpt = self.ngkpt,
                kshift = self.kshift,
                rhog_flag = True,
                **kwargs)
            self.add_task(self.scf2bgwtask, merge=False)
                
            kwargs.update(
                charge_density_fname = self.scftask.charge_density_fname,
                data_file_fname = self.scftask.data_file_fname,
                spin_polarization_fname = self.scftask.spin_polarization_fname)
            
        # Wavefunction tasks for Epsilon
        self.wfntask_ksh = QeBgwFlow(
            dirname = pjoin(self.dirname, '02-wfn'),
            ngkpt = self.ngkpt,
            kshift = self.kshift,
            nbnd = self.nbnd,
            rhog_flag = True,
            paraband_nproc = True,
            **kwargs)

        if kwargs.get('paraband'):
            assert kwargs.get('pseudobands') # pseudobands are required when parabands is used

        if kwargs.get('paraband'):
            self.paraband_task = ParaBandTask(
                dirname= pjoin(self.dirname, '02-wfn'),
                **kwargs)

        else:
            if kwargs.get('pseudobands', True):
                self.paraband_task = PseudoBandTask(
                    dirname= pjoin(self.dirname, '02-wfn'),
                    wfn2hdfonly = True,
                    **kwargs)

        self.wfntask_qsh = QeBgwFlow(
            dirname = pjoin(self.dirname, '03-wfnq'),
            ngkpt = self.ngkpt,
            kshift = self.kshift,
            qshift = self.qshift,
            # nbnd = None,
            nbnd = self.n_z_valence + 4,
            **kwargs)

        if kwargs.get('pseudobands', True):
            self.pseudoband_q = PseudoBandTask(
                dirname= pjoin(self.dirname, '03-wfnq'),
                wfnq_dir = pjoin(self.dirname, '03-wfnq'),
                wfnk_dir = pjoin(self.dirname, '02-wfn'),
                wfn2hdfonly = False,
                **kwargs)

        if kwargs.get('pseudobands', True):
            self.add_tasks([self.wfntask_ksh, self.paraband_task, self.wfntask_qsh, self.pseudoband_q])
        else:
            self.add_tasks([self.wfntask_ksh, self.wfntask_qsh])

        # TODO: NNS

        self.wfntask_q_nns = QeBgwFlow_NNS(
            dirname = pjoin(self.dirname, '06-wfnq-nns'),
            ngkpt = self.ngkpt,
            kshift = self.kshift,
            # qshift = self.qshift,
            # nbnd = None,
            nbnd = self.n_z_valence + 4,
            wfn_fname_nns_input = self.wfntask_ksh.wfn_fname,
            **kwargs)
        self.add_task(self.wfntask_q_nns)

        # Unshifted wavefunction tasks for Sigma
        # only if not already computed for Epsilon.
        if self.has_kshift:

            self.wfntask_ush = QeBgwFlow(
                dirname = pjoin(self.dirname, '04-wfn_co'),
                ngkpt = self.ngkpt,
                nbnd = self.nbnd,
                rhog_flag = True,
                **kwargs)

            self.add_task(self.wfntask_ush)

        else:
            self.wfntask_ush = self.wfntask_ksh

        fnames = dict(VSC_fname = self.scftask.dirname+'/VSC',
                      wfn_fname = self.wfntask_ksh.wfn_fname,
                      wfnq_fname = self.wfntask_qsh.wfn_fname,
                      wfn_co_fname = self.wfntask_ush.wfn_fname,
                      rho_fname = self.wfntask_ush.rho_fname,
                      vxc_dat_fname = self.wfntask_ush.vxc_dat_fname,
                      wfn_nns_dir=self.wfntask_q_nns.dirname,
                      wfn_nns_fname=self.wfntask_q_nns.wfn_fname)

        return fnames
    
    def summary(self, verbose):
        pass


def check_pseudo(pseudo_dir='./from_oncvpsp/', pseudos=['S.upf','H.upf']):
    pattern = 'z_valence'
    pseudos_z_valence = {}
    for pseudo in pseudos:
        result = subprocess.run(['grep', pattern, pseudo_dir+pseudo], capture_output=True, text=True)
        # print(result.stdout)
        match = re.search(r'[-+]?\d*\.?\d+', result.stdout)
        if match:
            number = float(match.group())  # Convert to float if needed
            # print(pseudo, 'z_valence:', number)
            pseudos_z_valence.update({pseudo:number})
        else:
            print('Warning: pseudo file is not found:', pseudo)
    # print(pseudos_z_valence)
    return pseudos_z_valence


if __name__ == "__main__":
    pass

    flow = DFT_GW_HPRO_Flow(
        mpirun='ibrun',
        nproc_flag = '-n',
        nproc=2240,
        nproc_per_node_flag='',
        nproc_per_node='',
        Siesta = '/work2/08237/bwhou/frontera/software/Anaconda/envs/siesta/bin/siesta',
        hpro = '/scratch1/08237/bwhou/12-deepGWBSE/Deep-GWBSE/HPRO/src/calc.py',
        PWFLAGS='-nk 16',
        PW='pw.x',
        dirname='flow',
        stru_file = './fp-input/mat-3/stru.cif',
        ecuteps = 15.0,
        ncbnd_sigma = 4,
        nvbnd_sigma = 5, # TODO: band check degeneracy sees not right
        ngkpt = [12,12, 1],
        qshift = [.001,.0,.0],
        nbnd = 100,
        ecutwfc = 75,
        prefix = 'MoS2',
        pseudo_dir = './from_oncvpsp/',
        # pseudos = ['Si.upf','H.upf'],
        pseudos = ['Mo.upf','S.upf'],
        basis_set_siesta = 'DZP',
        mesh_cutoff_siesta = 320,
        dm_tolerance_siesta = 1e-6, 
        max_scf_iter_siesta = 300,
        epsilon_extra_lines=['restart','degeneracy_check_override','dont_check_norms'],
        sigma_extra_lines=['degeneracy_check_override', 'dont_check_norms'],
        pseudobands = True, # assert ture if parabands is ture
        N_P_cond = 50,
        N_S_cond = 30,
        N_xi_cond = 10,
        paraband = True,
        nparaband = 3000,
    )

    flow.write()

    # check_pseudo()

