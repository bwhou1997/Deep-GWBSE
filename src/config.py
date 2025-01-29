import json as js
import os
import ase.io
from ase import Atoms
import tqdm

class single_mat:
    """
    information of each material directory
    mat-1/
    └── stru.cif

    """
    def __init__(self, root='./fp-input/mat-2'):
        # make root to absolute path
        self.root = os.path.abspath(root)
        self.prefix = root.split('/')[-1]
        self.atoms : Atoms = ase.io.read(self.root + '/stru.cif')

        self.natoms = len(self.atoms)
        self.elements_order = self.atoms.get_chemical_symbols() # ['Si', 'H', 'H', 'H', 'H']
        self.unique_elements_order = [] # ['Si', 'H']
        for ele in self.elements_order:
            if ele not in self.unique_elements_order:
                self.unique_elements_order.append(ele)
        self.nelements = len(set(self.elements_order))

class pseudo:
    """
    pseudo/
    ├── pseudo_qe/
    |   ├── ele1.upf
    |   └── ...
    ├── pseudo_siesta/
    |   ├── ele1.psf/psml
    |   └── ...
    """
    def __init__(self):
        self.pseudo_qe = os.listdir(self.root + '/pseudo_qe')
        self.pseudo_siesta = os.listdir(self.root + '/pseudo_siesta')

class fp_config:
    """
    Read fpconfig.json and analyze the fp-input directory

    fp-input/
    ├── fpconfig.json
    ├── mat-1/
    ├── mat-2/
    └── ...
    ```
    key attributes:
        mats = [mat1, mat2, ...], mat1 is a single_mat object
        kwargs: dict {'fp_config':xx, ...}
    """
    def __init__(self):
        self.fpconfig_read = False
        self.dir_read = False
    
    def read_fpconfig(self, root='./fp-input'):
        print(f'read fp config...')
        self.root = os.path.abspath(root)
        self.config_fname = self.root + '/fpconfig.json'
        with open(self.config_fname, 'r') as file:
            self.kwargs = js.load(file)
            
        # key fp configuration
        # registered name
        self.ibnd_min = self.kwargs['ibnd_min']
        self.ibnd_max = self.kwargs['ibnd_max']
        self.ecuteps = self.kwargs['ecuteps']
        self.ngkpt = self.kwargs['ngkpt']
        self.qshift = self.kwargs['qshift']
        self.nbnd = self.kwargs['nbnd'] # nscf
        self.ecutwfc = self.kwargs['ecutwfc']
        self.basis_set_siesta = self.kwargs['basis_set_siesta']
        self.mesh_cutoff_siesta = self.kwargs['mesh_cutoff_siesta']
        self.dm_tolerance_siesta = self.kwargs['dm_tolerance_siesta']
        self.max_scf_iter_siesta = self.kwargs['max_scf_iter_siesta']
        # analyze fp-input

        self.mats = [] # {'prefix/stru.cif':[ele1.upf, ele2.upf...]}
        total = len(os.listdir(self.root)) - 1 # exclude config file
        for root, dirs, files in tqdm.tqdm(os.walk(self.root), desc='read fp-input', total=total):
            if 'stru.cif' in files:
                self.mats.append(single_mat(root))

        self.fpconfig_read = True

    def summary_mats(self,):
        assert self.fpconfig_read, 'fpconfig.json not read'
        print('\n==========> summary <==========')
        print('===> mats <===')
        unique_elements_mats = set()
        average_atoms_mats = 0
        average_unique_elements_mats = 0
        for mat in self.mats:
            unique_elements_mats = unique_elements_mats.union(set(mat.elements_order))
            average_atoms_mats += mat.natoms
            average_unique_elements_mats += mat.nelements
        average_atoms_mats /= len(self.mats)
        average_unique_elements_mats /= len(self.mats)
        print(f'  materials: {len(self.mats)}')
        print(f'  unique elements: {len(unique_elements_mats)}')
        print('  ',unique_elements_mats)
        print(f'  average atoms : {average_atoms_mats :.2f}')
        print(f'  average elements: {average_unique_elements_mats :.2f}')
        print('===> mats <===')

        print('===> fp config <===')
        print(js.dumps(self.kwargs, indent=1, ensure_ascii=False))
        print('===> fp config <===')
        print('==========> summary <==========\n')

    def generate_fpconfig_default(self, output_path='./fpconfig_default.json'):        
        # Create a default structure using the current class attributes
        default_config = {
            'ibnd_min': 1,
            'ibnd_max': 8,
            'ecuteps': 30.0,
            'ngkpt': [4, 4, 4],  # Example data
            'qshift': [0.0, 0.0, 0.001],  # Example data
            'nbnd': 400,
            'ecutwfc': 60.0,
            'basis_set_siesta' : 'DZP',
            'mesh_cutoff_siesta' : 320,
            'dm_tolerance_siesta' : 1e-6, 
            'max_scf_iter_siesta' : 300
        }

        with open(output_path, 'w') as file:
            js.dump(default_config, file, indent=2, ensure_ascii=False, separators=(',', ': '))
        
        print(f"Generated and dumped default fpconfig to {output_path}")

if __name__ == "__main__":
    config = fp_config()
    config.generate_fpconfig_default('./fpconfig_default.json')
    config.read_fpconfig('./fp-input/')
    config.summary_mats()
    # sm = single_mat()