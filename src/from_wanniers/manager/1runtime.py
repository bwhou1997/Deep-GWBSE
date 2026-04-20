import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import get_chemical_formula, get_num_electrons, get_fermi_energy, EigController

if __name__ == "__main__":
    '''
    This script is run in realtime during the batch just before certain calculations.
    
    This is an executable script for 
        1) updating the nbnd parameter in nscf.in 
        2) updating the first_band and last_band parameters in wfck2r.in
        3) updating and nbnd parameter in bands.in & importing k-points 
            from Wannier90 (to plot comparable bandstructures)
        4) updating the num_bands and exclude_bands parameters in {prefix}.win.
    
    This has to be done in real-time since it relies on the number of electrons coming 
    from scf.out.
    '''
    if len(sys.argv) != 4:
        print("Usage: python 1runtime.py <edit_file> <json_path> <mode: int>")
        sys.exit(1)
    

    edit_file = sys.argv[1]
    json_path = sys.argv[2]
    mode = int(sys.argv[3])

    with open(edit_file, "r") as f:
        lines = f.readlines()
    with open(json_path, "r") as param_file:
        params = json.load(param_file)
    
    seedname = get_chemical_formula("../stru.cif")
    num_electrons = get_num_electrons("../1-scf/scf.out")
    fermi_energy = get_fermi_energy("../1-scf/scf.out")
    nbnd_num_bands = int(num_electrons / 2 + params["globals"]["fixed_conduct_bands"])
    num_wann = params["win"]["num_wann"]

    if mode == 1:  # nscf.in
        # Update the nbnd line
        for i, line in enumerate(lines):
            if "nbnd" in line:
                lines[i] = f"  nbnd        = {nbnd_num_bands}\n"
                break

    elif mode == 5:  # wfck2r.in
        for i, line in enumerate(lines):
            if "first_band" in line: # TODO REMOVE - NUM_WANN//2
                lines[i] = f"  first_band = {int(num_electrons//2 - num_wann//2 + 1)}\n"
            elif "last_band" in line:
                lines[i] = f"  last_band  = {int(num_electrons//2 - num_wann//2 + 2)}\n"

    elif mode == 4:  # bands.in
        for i, line in enumerate(lines):
            if "nbnd" in line:
                lines[i] = f"  nbnd        = {nbnd_num_bands}\n"
                break

        # Import the Wannier90 high symmetry lines from {seedname}_band.kpt
        with open(f"{seedname}_band.kpt", "r") as kpt_file:
            kpt_lines = kpt_file.readlines()

        # Find the K_POINTS line and insert kpt_lines after it
        for i, line in enumerate(lines):
            if "K_POINTS crystal_b" in line:
                # Insert kpt_lines after the K_POINTS line
                lines = lines[:i+1] + kpt_lines + lines[i+1:]
                break

    elif mode == 2:  # {seedname}.win
        lower_cutoff, upper_cutoff = EigController.get_exclude_bands(num_electrons, num_wann)
        for i, line in enumerate(lines):
            # Update the num_bands
            
            # if "num_bands" in line:  TODO DOESN'T WORK FOR SOM REASON!!!
            #     lines[i] = f"num_bands = {nbnd_num_bands}\n"
            
            if "exclude_bands" in line:
                lines[i] = f"exclude_bands = 1-{lower_cutoff},{upper_cutoff}-{nbnd_num_bands}\n"
            if "num_bands" in line:
                lines[i] = f"num_bands = {num_wann}\n"

    
    # OBSOLETE (This is from when I used to use the dis_win_min etc. parameters, 
    #   which had issues, so I use the more basic exclude_bands parameters ^. 
    #   This could be fixed in the future for improvements to performance)
    elif mode == 3:  # {seedname}.win
        controller = EigController(f"{seedname}.eig", num_electrons, nbnd_num_bands, num_wann)

        dis_bot, dis_top, froz_bot, froz_top = controller.get_windows()
        for i, line in enumerate(lines):
            if "dis_win_min" in line:
                lines[i] = f"dis_win_min   = {dis_bot}\n"
            elif "dis_win_max" in line:
                lines[i] = f"dis_win_max   = {dis_top}\n"
            elif "dis_froz_min" in line:
                lines[i] = f"dis_froz_min  = {froz_bot}\n"
            elif "dis_froz_max" in line:
                lines[i] = f"dis_froz_max  = {froz_top}\n"

        eig_data = controller.get_eig_data()
        for i, eig in enumerate(eig_data):
            # Plot vertical dashed line between min and max
            plt.plot([i, i], [min(eig), max(eig)], color='black', linestyle='--', linewidth=0.5)
            # Plot min and max as smaller points
            plt.scatter(i, min(eig), color='black', s=10)
            plt.scatter(i, max(eig), color='black', s=10)

        plt.title("Energy range of all bands")
        plt.xlabel("Band index")
        plt.ylabel("Energy (eV)")
        plt.axhline(fermi_energy, linewidth=0.75, alpha=0.5, color='k', linestyle='--')
        plt.axhline(dis_bot, linewidth=0.75, alpha=0.5, color='teal', label="Disentanglement window")
        plt.axhline(dis_top, linewidth=0.75, alpha=0.5, color='teal')
        plt.axhline(froz_bot, linewidth=0.75, alpha=0.5, color='navy', label="Frozen window")
        plt.axhline(froz_top, linewidth=0.75, alpha=0.5, color='navy')
        plt.legend()
        plt.savefig(f"{seedname}_band_range.png", dpi=600)

    # Write the updated lines back to the file
    with open(edit_file, "w") as f:
        f.writelines(lines)







# exclude_to = int(num_electrons / 2 - num_conduct_bands)

# elif "exclude_bands" in line:
#                 if exclude_to < 1:
#                     # Comment out line
#                     lines[i] = "! exclude_bands = all bands needed\n"
#                 elif exclude_to == 1:
#                     # Set to exclude only the first band
#                     lines[i] = "exclude_bands = 1\n"
#                 else:   
#                     lines[i] = f"exclude_bands = 1-{exclude_to}\n"
#                 break