import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, Delaunay, cKDTree
from scipy.interpolate import griddata
from interface import wfn
import torch.nn.functional as F
from scipy.ndimage import zoom
import plotly.graph_objects as go
from tqdm import tqdm
############# Toy data
au2ang = 0.52917721067

wf = wfn('../../examples/flows/mat-5/02-wfn/wfn.h5')
w00 = abs(wf.get_wfn_dataset()['wfn'][0,0])
lattice = wf.crystal['avec'] * wf.crystal['alat'] * au2ang


# periodic condition
shift = 20
w00 = np.concatenate([ w00[:,shift:],w00[:,:shift]], axis=1)
############## 2D case
# Function to fold a point into the Wigner-Seitz cell
wf_original_size = w00.shape
wf_frac_upsampling_factor = 1
upsampling_size = (round(wf_original_size[0] * wf_frac_upsampling_factor), 
                   round(wf_original_size[1] * wf_frac_upsampling_factor),
                   round(wf_original_size[2] * wf_frac_upsampling_factor)) 
final_grid_res_x = 40
final_grid_res_y = 40
final_grid_res_z = 30

#############
def fold_to_wigner_seitz(points, lattice_points):
    tree = cKDTree(lattice_points)
    _, indices = tree.query(points)
    return points - lattice_points[indices]

grid_x = np.linspace(0, 1, upsampling_size[0], endpoint=False)
grid_y = np.linspace(0, 1, upsampling_size[1], endpoint=False)
grid_z = np.linspace(0, 1, upsampling_size[2], endpoint=False)
grid_points_frac = np.array([[x, y, z] for x in grid_x for y in grid_y for z in grid_z])

# Convert to Cartesian coordinates
grid_points = np.dot(grid_points_frac, lattice)

grid_size = 2
lattice_points = np.array([
    m * lattice[0] + n * lattice[1] + p * lattice[2]
    for m in range(-grid_size, grid_size+1)
    for n in range(-grid_size, grid_size+1)
    for p in range(-grid_size, grid_size+1)
])

folded_grid_points = fold_to_wigner_seitz(grid_points, lattice_points)


xi, yi, zi = np.mgrid[
    folded_grid_points[:, 0].min():folded_grid_points[:, 0].max():final_grid_res_x*1j,
    folded_grid_points[:, 1].min():folded_grid_points[:, 1].max():final_grid_res_y*1j,
    folded_grid_points[:, 2].min():folded_grid_points[:, 2].max():final_grid_res_z*1j
]

print("interpolating")

density_matrix = zoom(w00, (1, wf_frac_upsampling_factor, wf_frac_upsampling_factor), order=3).flatten()
# zi_interp = griddata(folded_grid_points, density_matrix, (xi, yi, zi), method='linear', fill_value=0)

xyz_points = np.vstack([xi.flatten(), yi.flatten(), zi.flatten()]).T

# Create an empty array to store the interpolated values
zi_interp = np.zeros(xyz_points.shape[0])

# Process each point one by one (or in batches if needed)
# for i in tqdm(range(xyz_points.shape[0]), desc="Interpolating grid points"):
#     # Interpolate at each point
#     zi_interp[i] = griddata(folded_grid_points, density_matrix, xyz_points[i], method='linear', fill_value=0)


# fig = go.Figure(data=go.Volume(
#     x=xi.flatten(), y=yi.flatten(), z=zi.flatten(),
#     value=zi_interp.flatten(),
#     opacity=0.2, surface_count=20, colorscale='Viridis'
# ))
# fig.update_layout(title='3D Wigner-Seitz Density Visualization')
# fig.show()
