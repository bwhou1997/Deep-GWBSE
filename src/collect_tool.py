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

        if not os.path.exists(scf_out) or not os.path.exists(scf_in) or not os.path.exists(bands_dat):
            print(f"Missing files in {root}, skipping...")
            summary['unknown'].append(mat_id)
            continue

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

    print("Summary:")
    print("  Metals:", len(summary['metal']))
    print("  Semiconductors:", len(summary['semiconductor']))
    print("  Unknown:", len(summary['unknown']))
    print("  saved to metal_seek.json")
    with open("metal_seek.json", "w") as f:
        json.dump(summary, f, indent=4)

import os 
import json
import subprocess

def jobdone(task_dir: str) -> bool:
    task = os.path.basename(task_dir)
    if task == '01-density':
        result = subprocess.run(['grep', 'DONE', os.path.join(task_dir, 'scf.out')], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    elif task == "02-wfn":
        res1 = subprocess.run(['grep', 'DONE', os.path.join(task_dir, 'wfn.out')], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        res2 = subprocess.run(['grep', 'TOTAL', os.path.join(task_dir, 'parabands.out')], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        res3 = subprocess.run(['grep', 'DONE', os.path.join(task_dir, 'wfn.pp.out')], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return res1.returncode == 0 and res2.returncode == 0 and res3.returncode == 0
    elif task == "03-wfnq":
        res1 = subprocess.run(['grep', 'DONE', os.path.join(task_dir, 'wfn.out')], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        res2 = subprocess.run(['grep', 'DONE', os.path.join(task_dir, 'wfn.pp.out')], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        res3 = subprocess.run(['grep', 'alpha', os.path.join(task_dir, 'pseudo.out')], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return res1.returncode == 0 and res2.returncode == 0 and res3.returncode == 0
    elif task == '05-band':
        res1 = subprocess.run(['grep', 'DONE', os.path.join(task_dir, 'wfn.out')], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        res2 = subprocess.run(['grep', 'DONE', os.path.join(task_dir, 'wfn.pp.out')], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        res3 = subprocess.run(['grep', 'DONE', os.path.join(task_dir, 'bands.out')], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return res1.returncode == 0 and res2.returncode == 0 and res3.returncode == 0
    elif task == '06-wfnq-nns':
        res1 = subprocess.run(['grep', 'DONE', os.path.join(task_dir, 'wfn.out')], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        res2 = subprocess.run(['grep', 'DONE', os.path.join(task_dir, 'wfn.pp.out')], stdout=subprocess.PIPE, stderr=subprocess.PIPE)     
        return res1.returncode == 0 and res2.returncode == 0   
    elif task == "11-epsilon":
        result = subprocess.run(['grep', 'Job Done', os.path.join(task_dir, 'epsilon.out')], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    elif task == "12-epsilon-nns":
        result = subprocess.run(['grep', 'Job Done', os.path.join(task_dir, 'epsilon.out')], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0        
    elif task == "13-sigma":
        result = subprocess.run(['grep', 'Job Done', os.path.join(task_dir, 'sigma.out')], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    elif task == "14-inteqp":
        # grep 'Job Done' inteqp.log
        result = subprocess.run(['grep', 'Job Done', os.path.join(task_dir, 'inteqp.log')], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    elif task == "17-wfn_fi":
        res1 = subprocess.run(['grep', 'DONE', os.path.join(task_dir, 'wfn.out')], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        res2 = subprocess.run(['grep', 'DONE', os.path.join(task_dir, 'wfn.pp.out')], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return res1.returncode == 0 and res2.returncode == 0
    elif task == "18-kernel":
        # grep 'TOTAL' kernel.out
        result = subprocess.run(['grep', 'TOTAL', os.path.join(task_dir, 'kernel.out')], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    elif task == '19-absorption':
        # grep 'TOTAL' absorption.out
        result = subprocess.run(['grep', 'TOTAL', os.path.join(task_dir, 'absorption.out')], stdout=subprocess.PIPE, stderr=subprocess.PIPE)    
        return result.returncode == 0
    else:
        return 'unknown task'


def check_flows_status(flows: str = './flows-semi', dump: bool = True):
    print(f"Checking flows status in {flows}...")
    flows_status = {}
    
    for root, dirs, _ in os.walk(flows):
        if '01-density' not in dirs:
            if "02-wfn" not in dirs:
                # gw augmentation
                if "17-wfn_fi" not in dirs:
                    # bse augmentation
                    continue
        
        flow_status = {"Yes": [], "No": [], "Unknown Job": []}
        for dir in filter(lambda d: d != 'pp', dirs):
            job_status = jobdone(os.path.join(root, dir))
            if job_status is True:
                flow_status["Yes"].append(dir)
            elif job_status is False:
                flow_status["No"].append(dir)
            else:
                flow_status["Unknown Job"].append(dir)
        
        # Sort once at the end for efficiency
        for key in flow_status:
            if flow_status[key]:  
                flow_status[key].sort()
        
        flows_status[root] = {k: ",".join(v) for k, v in flow_status.items()}
    
    if dump:
        output_file = f"{os.path.basename(flows)}_status.json"
        print(output_file, flows)
        with open(output_file, 'w') as f:
            json.dump(flows_status, f, indent=4, separators=(',', ': '))
        print(f"Flows status saved to {output_file}")
    
    return flows_status

def generate_sbatch_jobs(fname='./run_aug.sh', nsbatch=3, hours=4, cluster='perlmutter', nodes=4):
    """
    Parses a script file to extract tasks and generates multiple SBATCH job scripts.

    Parameters:
        fname (str): Path to the input script file.
        nsbatch (int): Number of batch jobs to create.
        hours (int): Time in hours for each job.
        cluster (str): Cluster name (only 'perlmutter' is supported).
        nodes (int): Number of nodes for the job.
    """
    
    def set_sbatch(cluster, nodes, hours):
        assert cluster in ['perlmutter'], "Only perlmutter is supported"
        return f"#!/bin/bash\n#SBATCH -N {nodes}\n#SBATCH -C cpu\n#SBATCH -q regular\n#SBATCH -t {hours}:00:00\n"
    
    with open(fname, 'r') as f:
        lines = f.readlines()
    
    tasks = []
    start = None
    stack = []
    for i, line in enumerate(lines):
        if "cd" in line and "cd .." not in line:
            stack.append('cd')
            if start is None:
                start = i
        if "cd .." in line and start is not None:
            stack.pop()
            if not stack:
                tasks.append(''.join(lines[start:i+1]))
                start = None
            
    
    tasks_per_job = int(np.ceil(len(tasks) / nsbatch))
    
    for i in range(nsbatch):
        start = i * tasks_per_job
        end = min((i + 1) * tasks_per_job, len(tasks))
        job_tasks = tasks[start:end]
        
        prefix = os.path.splitext(fname)[0]
        job_file = prefix+f'_sub_{i+1}.sh'
        with open(job_file, 'w') as f:
            f.write(set_sbatch(cluster, nodes, hours))
            f.write("\n")
            for task in job_tasks:
                f.write(task)
                f.write("\n")
    
    print(f"Generated {nsbatch} job scripts.")

def compact_data(folder: str = '.', unwanted: dict = None):
    unwanted_files = unwanted.get('unwanted_files', [])
    for file in unwanted_files:
        file_path = pjoin(folder, file)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed {file_path}")
        else:
            print(f"{file_path} does not exist")
    return 0

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Collect Tools')

    parser = argparse.ArgumentParser(
        description="""Collect Tools:
    A collection tools for different modes.
    """,
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('mode', choices=['md', 'deeph', 'metalseek', 'st', 'sub','compact'], help="""\
    md: collect structures from MD output.
    deeph: collect DFT-Ham from DFT/SIESTA/HPRO flows.
    metalseek: determine metallicity from DFT flows
    st: check the status of the flows
    compact: compact data. delete the unwanted files to save space
    """)
    # parser.add_argument('mode', choices=['md', 'deeph', 'metal'], help='md: collect structures from MD output. \ndeeph: collect DFT-Ham from DFT/SIESTA/HPRO flows. \nmetal:')
    parser.add_argument('-md_input', type=str, help='md: input file name')
    parser.add_argument('-md_output', type=str, help='md: output file name')
    parser.add_argument('-md_suffix', type=str, default='', help='md: suffix for MD files')
    parser.add_argument('-flows', type=str, help='deeph/metalseek: directory containing DFT/SIESTA/HPRO flows')
    parser.add_argument('-job', type=str, help='sub: sbatch job file name')
    parser.add_argument('-nsbatch', type=int, default=3, help='sub: number of sub-sbatch jobs')
    parser.add_argument('-hours', type=int, default=4, help='sub: hours for each job')
    parser.add_argument('-nodes', type=int, default=4, help='sub: number of nodes for each job')
    parser.add_argument('-folder', type=str, default='.', help='compact: folder to compact')
    parser.add_argument('-unwanted', type=str, default='./unwanted.json', help='compact: json includes unwanted files to delete')

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
    
    elif args.mode == 'st':
        if not args.flows:
            parser.error('--flows is required in "st" mode')
        check_flows_status(args.flows)
    
    elif args.mode == 'sub':
        if not args.job:
            parser.error('--job is required in "sub" mode')
        generate_sbatch_jobs(args.job, args.nsbatch, args.hours, 'perlmutter', args.nodes)

    elif args.mode == 'compact':
        if not args.folder:
            parser.error('--folder is required in "compact" mode')
        if not args.unwanted:
            parser.error('--unwanted is required in "compact" mode')
        with open(args.unwanted, 'r') as f:
            unwanted = json.load(f)
        compact_data(args.folder, unwanted)
        
