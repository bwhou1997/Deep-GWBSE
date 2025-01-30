from os.path import join as pjoin
import ase.io
from ase.calculators.siesta import Siesta
from from_bgwpy.core import MPITask, IOTask
import os
from from_bgwpy.QE import QeScfTask, QeWfnTask, Qe2BgwTask
from from_bgwpy.DFT import WfnBgwFlow


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


if __name__ == "__main__":
    aobasistask = AobasisTask(dirname='./aobasis', 
                              stru_file='./fp-input/mat-1/stru.cif',
                              mesh_cutoff_siesta=300,                      
                              basis_set_siesta='DZP',
                              pseudo_dir='./scratch1/08237/bwhou/12-deepGWBSE/Deep-GWBSE/src/pseudo',
                              max_scf_iter_siesta=100,
                              dm_tolerance_siesta=1e-6)
    aobasistask.write()