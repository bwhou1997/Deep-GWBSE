import os
import stat
import json
from .generate_inputs import generate_qe_ins, generate_wfck2r_in, generate_pw2bgw_in, generate_bands_pp_in, generate_win, generate_pw2wan
from .helpers import get_chemical_formula, get_materials

def populate_dirs(json_path='manager/parameters.json'):
    """
    This function creates the calculation subdirectories in each material.
    """
    with open(json_path, 'r') as f:
        params = json.load(f)
    
    data_active_path = params['paths']['data_active']
    materials = get_materials(data_active_path)
    calculations = params['calculations']
    
    for material in materials:
        for calculation in calculations:
            os.makedirs(os.path.join(data_active_path, material, calculation), exist_ok=True)
    

def make_high_run(json_path='manager/parameters.json'):
    """
    This function makes the high-level run.sh script that cds into data_active, 
    into each material, and then calls its run.sh script.
    """
    with open(json_path, 'r') as f:
        params = json.load(f)
    
    data_active_path = params['paths']['data_active']
    materials = get_materials(data_active_path)
    
    script_path = 'run.sh'
    with open(script_path, 'w') as f:
        f.write(f'#!/bin/bash\n\n')
        f.write(f'#SBATCH --qos=regular\n')
        f.write(f'#SBATCH --time=02:00:00\n')
        f.write(f'#SBATCH --nodes=8\n')
        f.write(f'#SBATCH --ntasks-per-node=64\n')
        f.write(f'#SBATCH -o myjob.o\n')
        f.write(f'#SBATCH -e myjob.e\n')
        f.write(f'#SBATCH -C cpu\n\n')
        f.write(f'export OMP_NUM_THREADS=2\n')
        f.write(f'module load espresso/7.3.1-libxc-6.2.2-cpu\n')
        f.write(f'module load python\n')
        f.write(f'conda activate nersc-python\n\n')
        f.write(f'cd {data_active_path}\n\n')
        for material in materials:
            f.write(f'cd {material}\n')
            f.write(f'echo "{material}"\n')
            f.write(f'bash ./run.sh\n')
            f.write(f'cd ..\n')
            f.write(f'\n')
        f.write(f'cd ..\n')

    # 3. Make the script executable
    st = os.stat(script_path)
    os.chmod(script_path, st.st_mode | stat.S_IEXEC)

def make_middle_runs(json_path='manager/parameters.json'):
    """
    This function populates each material with a run.sh script
    that cds into each calculation and calls its run.sh script.s
    """
    with open(json_path, 'r') as f:
        params = json.load(f)
    
    data_active_path = params['paths']['data_active']
    materials = get_materials(data_active_path)
    calculations = params['calculations']

    os.chdir(f"{data_active_path}")
    for material in materials:
        os.chdir(material)
        script_path = 'run.sh'
        with open(script_path, 'w') as f:
            f.write(f'#!/bin/bash\n\n')
            for calculation in calculations:
                f.write(f'cd {calculation}\n')
                f.write(f'echo "{calculation}"\n')
                f.write(f'bash ./run.sh\n')
                f.write(f'cd ..\n\n')
        st = os.stat(script_path)
        os.chmod(script_path, st.st_mode | stat.S_IEXEC)

        os.chdir('..')
    os.chdir('..')

def make_low_runs(json_path='manager/parameters.json'):
    """
    This function populates each calculation with its run.sh script
    that runs the specific commands for that calculation.
    """
    with open(json_path, 'r') as f:
        params = json.load(f)
    
    data_active_path = params['paths']['data_active']
    materials = get_materials(data_active_path)
    calculations = params['calculations']

    os.chdir(f"{data_active_path}")
    for material in materials:
        os.chdir(material)
        for calculation in calculations:
            os.chdir(calculation)
            formula = get_chemical_formula('../stru.cif')
            abs_json_path = params['paths']['json_path']
            abs_runtime_py_path = params['paths']['runtime.py']
            abs_py_pp_path = params['paths']['py_pp.py']
            abs_step1_path = params['paths']['step1.py']
            abs_mse_out_path = params['paths']['mse_out']

            script_path = 'run.sh'
            with open(script_path, 'w') as f:
                f.write(f'#!/bin/bash\n\n')
                if calculation == '1-scf':
                    f.write(f'PWFLAGS="-npools 16"\n\n')
                    f.write(f'srun pw.x $PWFLAGS -inp scf.in > scf.out\n')
                elif calculation == '2-nscf':
                    f.write(f'mkdir -p {formula}.save\n')
                    f.write(f'mkdir -p rotated.save\n\n')

                    f.write(f'cd {formula}.save\n')
                    f.write(f'ln -sf ../../1-scf/{formula}.save/data* .\n')
                    f.write(f'ln -sf ../../1-scf/{formula}.save/charge* .\n')
                    f.write(f'cd ..\n\n')

                    f.write(f'PWFLAGS="-npools 16"\n\n')

                    f.write(f'python {abs_runtime_py_path} nscf.in {abs_json_path} 1\n')
                    f.write(f'srun pw.x $PWFLAGS -inp nscf.in > nscf.out\n\n')

                    f.write(f'srun -n 1 pw2bgw.x -inp pw2bgw.in > pw2bgw.out\n\n')

                    f.write(f'srun -n 1 wfn2hdf.x BIN WFN wfn.h5 > wfn2hdf.out\n\n')

                    f.write(f'python {abs_runtime_py_path} wfck2r.in {abs_json_path} 5\n')
                    f.write(f'srun -n 1 wfck2r.x < wfck2r.in > wfck2r.out\n\n')

                    f.write(f'rm -rf {formula}.wfc*\n')
                elif calculation == '3-wan':
                    f.write(f'ln -sf ../2-nscf/{formula}.save .\n\n')
                    f.write(f'WANNIER90X="/global/homes/t/taviandj/software/wannier90-3.1.0/wannier90.x"\n')
                    f.write(f'MPIEXEC="srun -n 1"\n\n')
                    f.write(f'python {abs_runtime_py_path} {formula}.win {abs_json_path} 2\n')
                    f.write(f'$MPIEXEC $WANNIER90X -pp {formula}\n')
                    f.write(f'pw2wannier90.x < {formula}.pw2wan > pw2wan.out\n')
                    f.write(f'$MPIEXEC $WANNIER90X {formula}\n')
                elif calculation == '4-bands':
                    f.write(f'mkdir -p {formula}.save\n')
                    f.write(f'cd {formula}.save\n')
                    f.write(f'ln -sf ../../1-scf/{formula}.save/data* .\n')
                    f.write(f'ln -sf ../../1-scf/{formula}.save/charge* .\n')
                    f.write(f'cd ..\n')
                    f.write(f'ln -sf ../3-wan/{formula}_band.kpt .\n\n')
                    f.write(f'PWFLAGS="-npools 16"\n\n')
                    f.write(f'python {abs_runtime_py_path} bands.in {abs_json_path} 4\n')
                    f.write(f'srun pw.x $PWFLAGS -inp bands.in > bands.out\n')
                    f.write(f'srun bands.x $PWFLAGS -inp bands_pp.in > bands_pp.out\n\n')
                    f.write(f'rm -rf {formula}.wfc*\n') 
                elif calculation == '5-python':
                    f.write(f'ln -sf ../3-wan/{formula}_band.dat .\n')
                    f.write(f'ln -sf ../4-bands/{formula}.bands.dat.gnu .\n')
                    f.write(f'ln -sf ../3-wan/{formula}_band.gnu .\n\n')
                    f.write(f'ln -sf ../2-nscf/wfck2r.oct .\n')
                    f.write(f'ln -sf ../3-wan/{formula}_u.mat .\n')
                    f.write(f'ln -sf ../2-nscf/{formula}.save .\n')
                    f.write(f'ln -sf ../2-nscf/rotated.save .\n\n')
                    #                                   <wannier90_bands_file>    <qe_bands_file>   <high_symmetry_wan_kps>   <mse_path>       <json_path>   <material_name>
                    f.write(f'python {abs_py_pp_path} {formula}_band.dat {formula}.bands.dat.gnu {formula}_band.gnu {abs_mse_out_path} {abs_json_path} {material}\n')
                    f.write(f'python {abs_step1_path} {abs_json_path}\n')
            st = os.stat(script_path)
            os.chmod(script_path, st.st_mode | stat.S_IEXEC)

            os.chdir('..')
        os.chdir('..')
    os.chdir('..')


def populate_inputs(json_path='manager/parameters.json'):
    """
    This function populates each calculation with its input files
    """
    with open(json_path, 'r') as f:
        params = json.load(f)
    
    data_active_path = params['paths']['data_active']
    materials = get_materials(data_active_path)
    calculations = params['calculations']

    os.chdir(f"{data_active_path}")
    for material in materials:
        os.chdir(material)
        for calculation in calculations:
            os.chdir(calculation)
            
            mini_cif_path = '../stru.cif'
            abs_json_path = params['paths']['json_path']

            if calculation == '1-scf':
                generate_qe_ins("scf", mini_cif_path, abs_json_path)
            elif calculation == '2-nscf':
                generate_qe_ins("nscf", mini_cif_path, abs_json_path)
                generate_wfck2r_in(mini_cif_path, abs_json_path)
                generate_pw2bgw_in(mini_cif_path, abs_json_path)
            elif calculation == '3-wan':
                generate_win(mini_cif_path, abs_json_path)
                generate_pw2wan(mini_cif_path, abs_json_path)
            elif calculation == '4-bands':
                generate_qe_ins("bands", mini_cif_path, abs_json_path)
                generate_bands_pp_in(mini_cif_path, abs_json_path)
            elif calculation == '5-python':
                # No inputs to generate, just ensure the script is there
                pass
            else:
                raise ValueError(f"Unknown calculation type: {calculation}")

            os.chdir('..')
        os.chdir('..')
    os.chdir('..')

