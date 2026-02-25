import os
import sys
import stat
import subprocess
import time
import json
import shutil
from manager.helpers import get_materials
from manager.generate_flow import populate_dirs, make_high_run, make_middle_runs, make_low_runs, populate_inputs


def clean(json_path='manager/parameters.json'):
    """
    This function generates a Bash script that removes everything in each material 
    directory except the cif file in data_source and deletes data_active entirely.
    
    Then it runs that script and deletes it. BE CAREFUL WITH THIS FUNCTION!
    """
    with open(json_path, 'r') as f:
        params = json.load(f)

    ############## Delete data_active directory ################
    data_active_path = params['paths']['data_active']

    # If it already exists, delete
    if os.path.exists(data_active_path):
        shutil.rmtree(data_active_path)
    ############################################################
    
    data_source_path = params['paths']['data_source']
    materials = get_materials(data_source_path)

    # 2. Write the script to clean everything
    script_path = 'clean_dirs.sh'
    with open(script_path, 'w') as f:
        f.write(f'#!/bin/bash\n\n')
        f.write(f'rm -rf run.sh\n\n')
        f.write(f'cd {data_source_path}\n\n')
        for material in materials:
            f.write(f'cd {material}\n')
            f.write(f'rm -rf 1-scf\n')
            f.write(f'rm -rf 2-nscf\n')
            f.write(f'rm -rf 3-wan\n')
            f.write(f'rm -rf 4-bands\n')
            f.write(f'rm -rf 5-python\n')
            f.write(f'rm -rf run.sh\n')
            f.write(f'cd ..\n')
            f.write(f'\n')
        f.write(f'cd ..\n')

    # 3. Make the script executable
    st = os.stat(script_path)
    os.chmod(script_path, st.st_mode | stat.S_IEXEC)

    # 4. Run the script
    result = subprocess.run(['./' + script_path], capture_output=True, text=True)

    # 5. Clean up the script file
    os.remove(script_path)



if __name__ == "__main__":
    print("Starting build process...")
    
    json_path = '../config/wan_parameters.json'
    
    if len(sys.argv) == 2 and sys.argv[1] == '-c':
        clean(json_path)
        sys.exit(0)

    
    with open(json_path, 'r') as f:
        params = json.load(f)

    data_source_path = params['paths']['data_source']
    data_active_path = params['paths']['data_active']
    shutil.copytree(data_source_path, data_active_path)
    
    populate_dirs(json_path)

    make_high_run(json_path)
    make_middle_runs(json_path)
    make_low_runs(json_path)

    populate_inputs(json_path)




