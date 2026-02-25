import os
import sys
import h5py
import numpy as np
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from helpers import get_materials, get_chemical_formula


def get_u_matrix(filename):
    """
    Reads a Wannier90 seedname_u.mat ASCII file and returns
    an array of shape (n_kpoints, n_bands, n_bands) of complex numbers.
    """
    with open(filename, 'r') as f:
        # skip the timestamp line
        next(f)
        # read dims: num_kps, n_b, _
        dims = next(f).split()
        num_kps, n_b, _ = map(int, dims)

        u_matrix = np.zeros((num_kps, n_b, n_b), dtype=np.complex128)

        for k_i in range(num_kps):
            # skip the empty line
            next(f)
            # read kpoint line
            _kpoint = next(f).split()
            # read next 64 lines into numpy array
            for i in range(n_b):
                for j in range(n_b):
                    line = next(f).split()
                    u_matrix[k_i, i, j] = complex(float(line[0]), float(line[1]))

        return u_matrix, n_b


def sort_mse_txt(mse_out_txt_path, output_path='mse_sorted.txt'):
    # Load file with the correct data types: string, int, float, float
    mse_data = np.loadtxt(mse_out_txt_path, 
                      dtype={'names': ('name', 'num_wann', 'num_electrons', 'mse_all', 'mse_center'),
                             'formats': ('U20', int, int, float, float)})

    # Sort by the 4th value (field name 'mse_center')
    order = np.argsort(mse_data['mse_center'])
    sorted_mse_data = mse_data[order]

    # Write sorted mse_data back to file
    with open(output_path, 'w') as f:
        for rec in sorted_mse_data:
            f.write(f"{rec['name']} {rec['num_wann']} {rec['num_electrons']} {rec['mse_all']} {rec['mse_center']}\n")

    print(f"Sorted mse_data saved to {output_path}")


if __name__ == "__main__":
    json_path = 'manager/parameters.json'
    with open(json_path, 'r') as f:
        params = json.load(f)
    
    data_active_path = params['paths']['data_active']
    materials = get_materials(data_active_path)


    with h5py.File('u_matrices.h5','a') as hf:
        os.chdir(f"{data_active_path}")
        for material in materials:
            os.chdir(material)
            os.chdir("3-wan")

            seedname = get_chemical_formula(f"../stru.cif")

            filename = f'{seedname}_u.mat'
            # Check if the file exists
            if not os.path.isfile(filename):
                print(f"File {filename} does not exist in /{material}. Skipping...")
                os.chdir("../../")
                continue
            
            u_matrix, n_b = get_u_matrix(filename)
            grp = hf.create_group(f"{n_b}_{material}")
            grp.create_dataset('u_matrix', data=u_matrix)

            os.chdir("../../")
        os.chdir("..")

    # sort_mse_txt('mse_out.txt')
