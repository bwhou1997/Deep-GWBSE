import ase
from ase import Atoms
from ase.io.espresso import read_espresso_in, read_espresso_out
from ase.visualize.ngl import NGLDisplay
import matplotlib.pyplot as plt
import copy
import numpy as np
from os.path import join as pjoin
import os

suffix = 'AB-661'
md_input_fname = './flow-hBN-md/flow-hBN-AB-661/01-density/scf.in'
md_output_fname = './flow-hBN-md/flow-hBN-AB-661/01-density/md.out'
stru_dir = './fp-input-AB-661/'


# only md atomic positions
# md = read_espresso_out('./flow-hBN/01-density/md.out')
structure = read_espresso_in(md_input_fname)
# NGLDisplay([structure])
natom = len(structure)
-3
with open(md_output_fname,'r') as f:
    lines = f.readlines()

temperature = []
structures = []

for i, line in enumerate(lines):
    if "ATOMIC_" in line:
        assert "crystal" in line, "Only crystal coordinates are supported"
        temp_structure = copy.deepcopy(structure)
        pos_frac = np.array([list(map(float, x.split()[1:])) for x in lines[i+1:i+1+natom]])
        pos_cart = pos_frac @ structure.cell.array
        temp_structure.set_positions(pos_cart)
        structures.append(temp_structure)

    if "temperature           =" in line:
        temperature.append(float(line.split()[2]))
        # print(line.split()[1])
    
# create stru list directory
print(f"Creating directory {stru_dir}")
os.makedirs(stru_dir, exist_ok=True)
for i, stru in enumerate(structures):
    dir_name = f'mat-{i+1:03d}-{temperature[i]:.2f}-{suffix}'
    os.makedirs(pjoin(stru_dir, dir_name), exist_ok=True)
    stru.write(pjoin(stru_dir, dir_name, 'stru.cif'), format='cif')
