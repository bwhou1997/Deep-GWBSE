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

with open('./from_bgwpy/QE/pseudobands.py','r') as file:
    pseudoband_py = file.read()

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

        self.calc = Siesta(directory=self.dirname,
                           label=self.prefix,
                           xc='PBE',
                           mesh_cutoff = kwargs['mesh_cutoff_siesta'],
                        #    basis_set=kwargs['basis_set_siesta'],
                           pseudo_path=kwargs['pseudo_dir']+'/pseudo_siesta',
                        #    pseudo_qualifier = 'psf',
                           fdf_arguments={'MaxSCFIterations': kwargs['max_scf_iter_siesta'],
                                          'DM.MixingWeight':kwargs['dm_tolerance_siesta']})

        self.atoms.calc = self.calc

    def write(self):
        # print('write siesta:', self.calc.getpath())
        super(AobasisTask, self).write()
        self.calc.write_input(self.atoms,'density')
    


class HPROTask(DeepTask):
    def __init__(self, dirname, **kwargs):
        super(HPROTask, self).__init__(dirname, **kwargs)
        self.dirnamt = dirname
        self.link_test()

    def write(self):
        super(HPROTask, self).write()
        # os.mkdir(self.dirname)
    
    def link_test(self):
        # with self.exec_from_dirname():
        a = './a'
        b = './b'
        # print(a, b)
        self.update_link(a,b)


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
            self.wfnq_fname  = os.path.relpath(kwargs['wfnq_dir'] + '/wfn.h5', cur_pat)
            self.wfnk_fname  = os.path.relpath(kwargs['wfnk_dir'] + '/wfn.h5', cur_pat)
            self.wfnq_fname_out  = os.path.relpath(kwargs['wfnq_dir'] + '/wfn_spb.h5', cur_pat)
            self.wfnk_fname_out  = os.path.relpath(kwargs['wfnk_dir'] + '/wfn_spb.h5', cur_pat)
            self.runscript.append(mpirun_flag+' '+nproc_flag+' 1 '+f'python pseudobands.py --fname_in {self.wfnk_fname} --fname_in_q {self.wfnq_fname} --fname_out {self.wfnk_fname_out} --fname_out_q {self.wfnq_fname_out} $> pseudo.out')
            # self.runscript.append(mpirun_flag+' '+nproc_flag+' 1 '+'wfn2hdf.x BIN wfn.cplx wfn.h5 &> wfn2hdf.out')
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