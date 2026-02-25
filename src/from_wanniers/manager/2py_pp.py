import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import get_num_electrons, get_fermi_energy, EigController, get_evs
import matplotlib.pyplot as plt
from matplotlib import rcParamsDefault
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad


if __name__ == "__main__":
    #################### Parse arguments #########################
    if len(sys.argv) != 7:
        print("Usage: python py_bands.py <wannier90_bands_file> <qe_bands_file> <high_symmetry_wan_kps> <mse_path> <json_path> <material_name>")
        sys.exit(1)

    data_w = np.loadtxt(sys.argv[1])
    data_qe = np.loadtxt(sys.argv[2])
    with open(sys.argv[5], "r") as param_file:
        params = json.load(param_file)
    material_name = sys.argv[6]
    
    seedname = sys.argv[1].split('_')[0]
    num_electrons = get_num_electrons("../1-scf/scf.out")
    fermi_energy = get_fermi_energy("../1-scf/scf.out")
    nbnd_num_bands = int(num_electrons / 2 + params["globals"]["fixed_conduct_bands"])
    num_wann = params["win"]["num_wann"]
    ######################################


    ######################### Wannier90 bands plot #########################
    data_w = data_w.reshape((num_wann, -1, 2))

    for i in range(data_w.shape[0]):
        indices = np.arange(data_w.shape[1])
        if i >= num_wann / 2 - 1 and i < num_wann / 2 + 1: 
            # Green bands for middle two
            plt.plot(indices, 
                data_w[i, :, 1], 
                linewidth=1, 
                alpha=0.5, 
                color='green',
                linestyle='--')
            # Plot upside green triangles at every 10th point
            plt.scatter(
                indices[::10], 
                data_w[i, ::10, 1], 
                marker='^', 
                alpha=0.5,
                facecolors='none', 
                edgecolors='green', 
                s=15, 
                linewidths=0.75, 
                zorder=3
            )
        else: 
            # Red bands
            plt.plot(indices, 
                data_w[i, :, 1], 
                linewidth=1, 
                alpha=0.5, 
                color='red',
                linestyle='--')
    ######################################


    ######################### QE bands plot ################################
    data_qe = data_qe.reshape((nbnd_num_bands, -1, 2))

    for i in range(data_qe.shape[0]):        
        indices = np.arange(data_qe.shape[1])
        if i >= num_electrons / 2 - num_wann / 2 and i < num_electrons / 2 + num_wann / 2:
            # Opaque black bands to compare with Wannier bands
            plt.plot(indices, 
                    data_qe[i, :, 1], 
                    linewidth=1, 
                    alpha=0.5, 
                    color='k',
                    linestyle='-') 
        else:
            # Faint bands for reference
            plt.plot(indices, 
                    data_qe[i, :, 1], 
                    linewidth=1, 
                    alpha=0.2, 
                    color='k',
                    linestyle='-')
    ####################################


    ######################### MSE calculation ##############################
    with open(sys.argv[4], 'a') as mse_file:
        # Finds the MSE of all 8 ands and the center 2 bands
        # The mse_center is the more importance benchmark, as they are
        #   the top valence and bottom conduction bands
        mse_all = 0
        mse_center = 0
        
        for i in range(num_wann):
            mse = np.mean((data_w[i, :, 1] - data_qe[i + int(num_electrons / 2) - int(num_wann / 2), :, 1]) ** 2)
            mse_all += mse
            if i >= num_wann / 2 - 1 and i < num_wann / 2 + 1:
                mse_center += mse
        mse_all /= num_wann
        mse_center /= 2

        mse_file.write(f"{material_name} {num_wann} {num_electrons} {mse_all} {mse_center}\n")
    ###################################

    # ##################### Plot high symmetry k-points ######################    
    # Read in high symmetry k-points from {prefix}_band.gnu
    hs_k = []
    hs_lbls = []
    with open(sys.argv[3], 'r') as f:
        lines = f.readlines()
    # Round each value of data_w to 5 decimal places for matching
    k_path = np.round(data_w[0, :, 0], 5)
    for line in lines:
        if "set xtics" in line:
            # Grab the string of tics inside the parentheses
            xtics_str = line.split("set xtics",1)[1].strip()
            # Turn the string into a list of strings
            xtics_list = xtics_str.strip('()').split(',')
            for xtic in xtics_list:
                xtic = xtic.split()
                # Turn the first element from double string into string
                hs_lbls.append(xtic[0].strip('"'))
                projected_k = round(float(xtic[1]), 5)
                # Find the closest matching k-point in data_w
                idx = np.abs(k_path - projected_k).argmin()
                hs_k.append(idx)

    for x in hs_k:
        plt.axvline(x, linewidth=0.75, alpha=0.2, color="k")
    plt.xticks(ticks=hs_k, labels=hs_lbls)
    # ####################################
    
    # Disentanglement windows [DEPRICATED]
    # controller = EigController(f"../3-wan/{seedname}.eig", num_electrons, nbnd_num_bands, num_wann)
    # dis_bot, dis_top, froz_bot, froz_top = controller.get_windows()
    # plt.axhline(dis_bot, linewidth=0.75, alpha=0.5, color='teal', label="Disentanglement window")
    # plt.axhline(dis_top, linewidth=0.75, alpha=0.5, color='teal')
    # plt.axhline(froz_bot, linewidth=0.75, alpha=0.5, color='navy', label="Frozen window")
    # plt.axhline(froz_top, linewidth=0.75, alpha=0.5, color='navy')

    # Fermi energy
    plt.axhline(fermi_energy, linewidth=0.75, alpha=0.5, color='k', linestyle='--')    

    # Set the y-axis limits from data_w
    ymin = np.min(data_w[:, :, 1]) - 1
    ymax = np.max(data_w[:, :, 1]) + 1
    plt.ylim(ymin, ymax)

    plt.ylabel("Energy (eV)")
    plt.savefig(f'{material_name}_bands.png', dpi=600)
    storage_path = params["paths"]["storage"]
    plt.savefig(f'{storage_path}/{num_wann}_{material_name}.png', dpi=600)