from os.path import join as pjoin
import ase.io
from ase.calculators.siesta import Siesta
from from_bgwpy.core import MPITask, IOTask
import os
from from_bgwpy.QE import QeScfTask, QeWfnTask, Qe2BgwTask
from from_bgwpy.DFT import WfnBgwFlow
from from_bgwpy.BGW import EpsilonTask
from from_bgwpy.QE.pseudobands_str import pseudoband_py
import os
import json

# with open('./from_bgwpy/QE/pseudobands.py','r') as file:
#     pseudoband_py = file.read()

class DeepTask(MPITask, IOTask):
    _TAG_JOB_COMPLETED = 'TOTAL'
    pass


class AobasisTask(DeepTask):
    """
        Arguments
        ---------
        dirname : str
            Directory in which the files are written and the code is executed.
            Will be created if needed.

        Keyword arguments (see DFT_GW_HPRO_FLOW())
        -----------------
        (All mandatory unless specified otherwise)

        Properties
        ----------
    """
    def __init__(self, dirname, **kwargs):
        super(AobasisTask, self).__init__(dirname, **kwargs)

        self.dirname = dirname
        self.atoms = ase.io.read(kwargs['stru_file'])
        self.symbols = str(self.atoms.symbols)
        self.prefix = self.symbols if 'prefix' not in kwargs else kwargs['prefix']

        mpirun_flag = kwargs.get('mpirun', 'mpirun')
        siesta_flag = kwargs.get('Siesta','siesta')
        nproc_flag = kwargs.get('nproc_flag', ' -n ')

        self.calc = Siesta(directory=self.dirname,
                           label=self.prefix,
                           xc='PBE',
                           mesh_cutoff = kwargs['mesh_cutoff_siesta'],
                        #    basis_set=kwargs['basis_set_siesta'],
                           pseudo_path=os.path.relpath(kwargs['pseudo_dir'], self.dirname),
                        #    pseudo_qualifier = 'psf',
                           fdf_arguments={'MaxSCFIterations': kwargs['max_scf_iter_siesta'],
                                          'DM.MixingWeight':kwargs['dm_tolerance_siesta']})

        self.atoms.calc = self.calc
        self.runscript.fname = "aobasis.run"
        self.runscript.append(mpirun_flag+' '+nproc_flag+' 1 '+f'{siesta_flag} < {self.prefix}.fdf &> siesta.out')

    def write(self):
        # print('write siesta:', self.calc.getpath())
        super(AobasisTask, self).write()
        self.calc.write_input(self.atoms,'density')
    


class HPROTask(DeepTask):
    def __init__(self, dirname, **kwargs):
        super(HPROTask, self).__init__(dirname, **kwargs)
        self.dirname = dirname
        # self.link_test()
        mpirun_flag = kwargs.get('mpirun', 'mpirun')
        nproc_flag = kwargs.get('nproc_flag', ' -n ')

        self.PW2AO_kwargs = {
                'Warning': "you might modify fptask.py to change path if you change folder name of previous step",
                'lcao_interface':'siesta',
                'lcaodata_root':'../05-aobasis',  # This might introduce errors when we change name of 05-aobasis
                'hrdata_interface':'qe-bgw',
                'vscdir':'../01-density/VSC',
                'upfdir':f"{os.path.relpath(kwargs['pseudo_dir'], self.dirname)}",
                'ecutwfn':kwargs.get('ecutwfn_hpro', 30),
                'outdir':f"{self.dirname}/aohamiltonian"}
        self.runscript.fname = 'hpro.run'
        self.runscript.append(mpirun_flag+' '+nproc_flag+' 1 '+f"python {kwargs['hpro']} > hpro.out")

    def write(self):
        super(HPROTask, self).write()
        with open(self.dirname+'/calc.json', 'w') as file:
            json.dump(self.PW2AO_kwargs, file, indent=4)
        # with open(self.dirname+'/calc.py', 'w') as file:
        #     file.write(pseudoband_py)

class PseudoBandTask(DeepTask):
    def __init__(self, dirname, **kwargs):
        super(PseudoBandTask, self).__init__(dirname, **kwargs)
        self.dirname = dirname
        self.wfn2hdfonly = kwargs.get('wfn2hdfonly', None)
        # wfn2hdf5
        mpirun_flag = kwargs.get('mpirun', 'mpirun')
        nproc_flag = kwargs.get('nproc_flag', ' -n ')
        self.runscript.fname = 'pseudo.sh'
        self.runscript.append(mpirun_flag+' '+nproc_flag+' 1 '+'wfn2hdf.x BIN wfn.cplx wfn.h5 &> wfn2hdf.out')

        # pseudonize
        if not kwargs.get('wfn2hdfonly'):
            cur_pat = self.dirname
            self.wfnq_fname = os.path.relpath(kwargs['wfnq_dir'] + '/wfn.h5', cur_pat)
            self.wfnk_fname = os.path.relpath(kwargs['wfnk_dir'] + '/wfn.h5', cur_pat)

            self.wfnq_fname_out = os.path.relpath(kwargs['wfnq_dir'] + '/wfn_q.h5', cur_pat)
            self.wfnk_fname_out = os.path.relpath(kwargs['wfnq_dir'] + '/wfn_k.h5', cur_pat)

            self.wfnq_fname_out_h5 = os.path.relpath(kwargs['wfnq_dir'] + '/wfn.cplx', cur_pat)
            self.wfnk_fname_out_h5 = os.path.relpath(kwargs['wfnk_dir'] + '/wfn.cplx', cur_pat)

            # TODO: add pseudobands setting; wfnq?
            self.runscript.append(mpirun_flag+' '+nproc_flag+' 1 '+f'python pseudobands.py --fname_in {self.wfnk_fname} --fname_in_q {self.wfnq_fname} --fname_out {self.wfnk_fname_out} --fname_out_q {self.wfnq_fname_out} --N_P_cond {kwargs.get("N_P_cond", 100)} --N_S_cond {kwargs.get("N_S_cond", 10)} --N_xi_cond {kwargs.get("N_xi_cond", 5)}  &> pseudo.out')
            # self.runscript.append(mpirun_flag+' '+nproc_flag+' 1 '+f'hdf2wfn.x BIN {self.wfnq_fname_out} {self.wfnq_fname_out_h5} &> wfn2hdf.out') # we don't do anything to wfnq
            self.runscript.append(mpirun_flag+' '+nproc_flag+' 1 '+f'hdf2wfn.x BIN {self.wfnk_fname_out} {self.wfnk_fname_out_h5} &> wfn2hdf.out')

            # self.runscript.append(mpirun_flag+' '+nproc_flag+' 1 '+'python pseudobands.py --fname_in WFN.h5 --fname_in_q WFNq.h5 --fname_out WFN_SPB.h5 --fname_out_q WFN_SPB_q.h5 --N_P_val 10 --N_P_cond 10 --N_S_val 10 --N_S_cond 150 --N_xi_val 2 --N_xi_cond 2')
    
    def write(self,): 
        super().write()

        if not self.wfn2hdfonly:        
            with open(self.dirname+'/pseudobands.py', 'w') as file:
                file.write(pseudoband_py)
        pass

if __name__ == "__main__":
    aobasistask = AobasisTask(dirname='./aobasis', 
                              stru_file='./fp-input/mat-1/stru.cif',
                              mesh_cutoff_siesta=300,                      
                              basis_set_siesta='DZP',
                              pseudo_dir='./scratch1/08237/bwhou/12-deepGWBSE/Deep-GWBSE/src/pseudo',
                              max_scf_iter_siesta=100,
                              dm_tolerance_siesta=1e-6)
    aobasistask.write()
