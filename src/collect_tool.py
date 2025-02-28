#!/usr/bin/env python
import os
import subprocess
from os.path import join as pjoin
import numpy as np
import json
import ase
from ase import Atoms
from ase.io.espresso import read_espresso_in, read_espresso_out
from ase.visualize.ngl import NGLDisplay
import matplotlib.pyplot as plt
import copy
import numpy as np
from os.path import join as pjoin
import os
import subprocess
from tqdm import tqdm

suffix = 'AB-661'
md_input_fname = './flow-hBN-md/flow-hBN-AB-661/01-density/scf.in'
md_output_fname = './flow-hBN-md/flow-hBN-AB-661/01-density/md.out'
stru_dir = './fp-input-AB-661/'


def collect_from_md(md_input_fname = './flow-hBN-md/flow-hBN-AB-661/01-density/scf.in',
                    md_output_fname = './flow-hBN-md/flow-hBN-AB-661/01-density/md.out',
                    suffix = ''):

    # only md atomic positions
    # md = read_espresso_out('./flow-hBN/01-density/md.out')
    stru_dir = './collected-md-stru/'
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

def collect_from_flows_2_deep(deeph_flows='./'):

    assert os.path.exists(deeph_flows), f"Directory {deeph_flows} does not exist"

    roots = []
    for root, dirs, files in os.walk(deeph_flows):
        if "hamiltonians.h5" in files:
            roots.append(root)
            print(root)

    os.makedirs("dataset", exist_ok=True)
    for i, root in tqdm(enumerate(roots), total=len(roots)):
        subprocess.run(["cp", "-r", root, f"dataset/ham-{i:03d}"])


def metal_seek(flows='./flows-bwhou'):
    # walk through the directory
    roots = []
    for root, dirs, files in os.walk(flows):
        if "stru.cif" in files:
            roots.append((root, root.split('/')[-1]))

    summary = {"metal":[], "semiconductor":[], "unknown":[]}
    for root, mat_id in roots:

        scf_out = pjoin(root, "01-density",'scf.out') 
        scf_in = pjoin(root, "01-density",'scf.in')
        bands_dat = pjoin(root, "05-band",'bands.dat.gnu')

        # grep "Fermi" of scf_out
        print(f"Material: {mat_id}")
        result = subprocess.run(f"grep 'Fermi' {scf_out}", capture_output=True, shell=True)
        if result.stdout == b'':
            print(f"Fermi level not found in {scf_out}")
            summary['unknown'].append(mat_id)
            continue
        Fermi_level = float(result.stdout.split()[-2])
        print(f"  Fermi level: {Fermi_level:.2f} eV")

        ele_res = subprocess.run(f"grep 'number of electrons' {scf_out}", capture_output=True, shell=True)
        if ele_res.stdout == b'':
            print(f"Number of electrons not found in {scf_out}")
            continue
        num_electrons = float(ele_res.stdout.split()[-1])
        print(f"  Number of electrons: {num_electrons:.2f}")
        # read the bands.dat.gnu file

        spin_orb_res = subprocess.run(f"grep 'lspinorb' {scf_in}", capture_output=True, shell=True)
        if spin_orb_res.stdout == b'':
            soc = False
            nvalence = num_electrons / 2 
        else:
            soc = True
            nvalence = num_electrons
        print(f"  Spin-orbit coupling: {'on' if soc else 'off'}")
        print(f"  Number of valence electrons: {nvalence:.2f}")

        with open(bands_dat, 'r') as f:
            lines = f.readlines()
            nk = lines.index('\n') # number of k-points
            assert len(lines) % (nk + 1) == 0, "bands.dat.gnu file is not correctly formatted"
            nb = len(lines) // (nk + 1) # +1 is from space line
            # extract the last number
            f = np.loadtxt(bands_dat)[:,1].reshape(nb, nk)

        if nvalence % 1 != 0: # half-occupation
            vbm_index = int(np.ceil(nvalence)) - 1
            cbm_index = vbm_index
        else:
            vbm_index = int(nvalence) - 1
            cbm_index = vbm_index + 1

        vbm = np.max(f[vbm_index, :])
        cbm = np.min(f[cbm_index, :])

        print(f"  VBM index (start with 0): {vbm_index}, VBM energy: {vbm:.2f} eV")
        print(f"  CBM index (start with 0): {cbm_index}, CBM energy: {cbm:.2f} eV")

        if cbm > Fermi_level and Fermi_level > vbm:
            print(f"  {mat_id} is a semiconductor")
            summary['semiconductor'].append(mat_id)
        else:
            print(f"  {mat_id} is a metal")
            summary['metal'].append(mat_id)

        # break

    print("Summary:")
    print("  Metals:", len(summary['metal']))
    print("  Semiconductors:", len(summary['semiconductor']))
    print("  Unknown:", len(summary['unknown']))
    print("  saved to metal_seek.json")
    with open("metal_seek.json", "w") as f:
        json.dump(summary, f, indent=4)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Collect Tools')

    parser = argparse.ArgumentParser(
        description="""Collect Tools:
    A collection tools for different modes.
    """,
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('mode', choices=['md', 'deeph', 'metalseek'], help="""\
    md: collect structures from MD output.
    deeph: collect DFT-Ham from DFT/SIESTA/HPRO flows.
    metalseek: determine metallicity from DFT flows
    """)
    # parser.add_argument('mode', choices=['md', 'deeph', 'metal'], help='md: collect structures from MD output. \ndeeph: collect DFT-Ham from DFT/SIESTA/HPRO flows. \nmetal:')
    parser.add_argument('--md_input', type=str, help='md: input file name')
    parser.add_argument('--md_output', type=str, help='md: output file name')
    parser.add_argument('--md_suffix', type=str, default='', help='md: suffix for MD files')
    parser.add_argument('--flows', type=str, default='./flows', help='deeph/metalseek: directory containing DFT/SIESTA/HPRO flows')

    args = parser.parse_args()

    if args.mode == 'md':
        if not args.md_input or not args.md_output:
            parser.error('--md_input_fname and --md_output_fname are required in "md" mode')
        if not args.md_suffix:
            args.md_suffix = ''
        collect_from_md(args.md_input, args.md_output, args.md_suffix)
    elif args.mode == 'deeph':
        collect_from_flows_2_deep(args.deeph_flows)
    
    elif args.mode == 'metalseek':
        if not args.flows:
            parser.error('--flows is required in "metalseek" mode')
        metal_seek(args.flows)
