
# import numpy as np
# from scipy.spatial import Voronoi, ConvexHull
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# # Define the lattice vectors (3x3 matrix)
# lattice = np.array([[1,0,0], 
#                     [0.5,0.86607,0], 
#                     [0,0, 3]])

# # Generate lattice points (integer linear combinations)
# grid_range = range(-1, 2) # Small neighborhood around origin
# points = np.array([
#     i*lattice[0] + j*lattice[1] + k*lattice[2]
#     for i in grid_range for j in grid_range for k in grid_range
# ])

# # Compute the Voronoi diagram
# vor = Voronoi(points)

# # Extract Wigner-Seitz cell (region corresponding to the origin)
# origin_index = np.where((points == [0, 0, 0]).all(axis=1))[0][0]
# region_index = vor.point_region[origin_index]
# vertices = vor.vertices[vor.regions[region_index]]

# # Plotting
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# hull = ConvexHull(vertices)
# ax.add_collection3d(Poly3DCollection(vertices[hull.simplices], alpha=0.5, edgecolor='k'))
# ax.scatter(*vertices.T, color='r', s=50)  # Plot vertices
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.spatial import Voronoi, voronoi_plot_2d

# # Define the lattice basis vectors
# a1 = np.array([1.0, 0.0])
# a2 = np.array([0.5, np.sqrt(3)/2])

# # Generate a grid of lattice points
# grid_size = 1  # Controls how many points we generate around the origin
# lattice_points = [m * a1 + n * a2 for m in range(-grid_size, grid_size+1) 
#                                      for n in range(-grid_size, grid_size+1)]
# lattice_points = np.array(lattice_points)

# # Compute the Voronoi diagram
# vor = Voronoi(lattice_points)

# # Plot the Wigner-Seitz cell
# fig, ax = plt.subplots(figsize=(6,6))
# voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='k', line_width=1.5)

# # Highlight the original lattice points
# ax.scatter(lattice_points[:, 0], lattice_points[:, 1], c='r', marker='o', label='Lattice Points')

# # Mark the central cell
# ax.set_xlim(-1.5, 1.5)
# ax.set_ylim(-1.5, 1.5)
# ax.set_aspect('equal')
# plt.legend()
# plt.title("Wigner-Seitz Cell from Unit Cell")
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.spatial import Voronoi

# # Define lattice basis vectors
# a1 = np.array([1.0, 0.0])
# a2 = np.array([0.5, np.sqrt(3)/2])
# lattice_basis = np.array([a1, a2])

# # Atomic positions in the primitive cell (can be outside the Wigner-Seitz cell)
# atom_positions = np.array([[0.1, 0.1], [0.6, 0.6]])

# # Generate a grid of nearby lattice points
# grid_size = 2  # Size of lattice considered
# lattice_points = np.array([m * a1 + n * a2 for m in range(-grid_size, grid_size+1) 
#                                                 for n in range(-grid_size, grid_size+1)])

# # Compute the Voronoi diagram
# vor = Voronoi(lattice_points)

# # Function to fold an atom into the Wigner-Seitz cell
# def fold_to_wigner_seitz(atom, lattice_points):
#     """Finds the closest equivalent atom position inside the Wigner-Seitz cell."""
#     distances = np.linalg.norm(lattice_points - atom, axis=1)  # Compute distances to all lattice points
#     closest_lattice_point = lattice_points[np.argmin(distances)]  # Find the nearest lattice point
#     return atom - closest_lattice_point  # Translate the atom into the Wigner-Seitz cell

# # Fold atomic positions
# folded_atoms = np.array([fold_to_wigner_seitz(atom, lattice_points) for atom in atom_positions])

# # Plot Wigner-Seitz Cell
# fig, ax = plt.subplots(figsize=(6,6))
# ax.set_xlim(-1.5, 1.5)
# ax.set_ylim(-1.5, 1.5)
# ax.set_aspect('equal')

# # Plot Voronoi diagram to show Wigner-Seitz cell
# for simplex in vor.ridge_vertices:
#     simplex = np.array(simplex)
#     if np.all(simplex >= 0):  # Ignore points at infinity
#         ax.plot(vor.vertices[simplex, 0], vor.vertices[simplex, 1], 'k-')

# # Plot lattice points
# ax.scatter(lattice_points[:, 0], lattice_points[:, 1], c='r', marker='o', label="Lattice Points")

# # Plot original and folded atomic positions
# ax.scatter(atom_positions[:, 0], atom_positions[:, 1], c='blue', marker='x', label="Original Atoms")
# ax.scatter(folded_atoms[:, 0], folded_atoms[:, 1], c='green', marker='s', label="Folded Atoms")

# plt.legend()
# plt.title("Folding Atoms into Wigner-Seitz Cell")
# plt.show()
#######################################################
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.spatial import Voronoi

# # Define lattice basis vectors
# a1 = np.array([1.0, 0.0])
# a2 = np.array([0.5, np.sqrt(3)/2])
# lattice_basis = np.array([a1, a2])

# # Generate a 20x20 uniform grid in the primitive cell
# N = 20  # Grid size
# grid_x = np.linspace(0, 1, N, endpoint=False)  # Fractional coordinates
# grid_y = np.linspace(0, 1, N, endpoint=False)
# grid_points_frac = np.array([[x, y] for x in grid_x for y in grid_y])  # Flatten the grid

# # Convert to Cartesian coordinates
# grid_points = np.dot(grid_points_frac, lattice_basis)  # Fractional → Cartesian

# # Generate a grid of nearby lattice points
# grid_size = 2  # Extending the grid to consider neighboring unit cells
# lattice_points = np.array([m * a1 + n * a2 for m in range(-grid_size, grid_size+1) 
#                                                 for n in range(-grid_size, grid_size+1)])

# # Compute the Voronoi diagram
# vor = Voronoi(lattice_points)

# # Function to fold a point into the Wigner-Seitz cell
# def fold_to_wigner_seitz(point, lattice_points):
#     """Finds the closest equivalent point inside the Wigner-Seitz cell."""
#     distances = np.linalg.norm(lattice_points - point, axis=1)  # Compute distances to all lattice points
#     closest_lattice_point = lattice_points[np.argmin(distances)]  # Find the nearest lattice point
#     return point - closest_lattice_point  # Translate the point into the Wigner-Seitz cell

# # Fold all grid points
# folded_grid_points = np.array([fold_to_wigner_seitz(pt, lattice_points) for pt in grid_points])

# # Plot Wigner-Seitz Cell and the Folded Grid
# fig, ax = plt.subplots(figsize=(6,6))
# ax.set_xlim(-1.5, 1.5)
# ax.set_ylim(-1.5, 1.5)
# ax.set_aspect('equal')

# # Plot Voronoi diagram (Wigner-Seitz Cell)
# for simplex in vor.ridge_vertices:
#     simplex = np.array(simplex)
#     if np.all(simplex >= 0):  # Ignore points at infinity
#         ax.plot(vor.vertices[simplex, 0], vor.vertices[simplex, 1], 'k-')

# # Plot lattice points
# ax.scatter(lattice_points[:, 0], lattice_points[:, 1], c='r', marker='o', label="Lattice Points")

# # Plot original and folded grid points
# ax.scatter(grid_points[:, 0], grid_points[:, 1], c='blue', marker='.', alpha=0.3, label="Original Grid")
# ax.scatter(folded_grid_points[:, 0], folded_grid_points[:, 1], c='green', marker='.', alpha=0.8, label="Folded Grid")

# plt.legend()
# plt.title("Folding a Uniform Grid into the Wigner-Seitz Cell")
# plt.show()
#######################################################

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.spatial import Voronoi

# # Define lattice basis vectors
# a1 = np.array([1.0, 0.0])
# a2 = np.array([0.5, np.sqrt(3)/2])
# lattice_basis = np.array([a1, a2])

# # Generate a 20x20 uniform grid in the primitive cell
# N = 20  # Grid size
# grid_x = np.linspace(0, 1, N, endpoint=False)  # Fractional coordinates
# grid_y = np.linspace(0, 1, N, endpoint=False)
# grid_points_frac = np.array([[x, y] for x in grid_x for y in grid_y])  # Flatten the grid

# # Convert to Cartesian coordinates
# grid_points = np.dot(grid_points_frac, lattice_basis)  # Fractional → Cartesian

# # Generate a synthetic density matrix (random values as a toy dataset)
# density_matrix = np.random.rand(N, N).flatten()  # Flatten to match grid points

# # Generate a grid of nearby lattice points
# grid_size = 2  # Extending the grid to consider neighboring unit cells
# lattice_points = np.array([m * a1 + n * a2 for m in range(-grid_size, grid_size+1) 
#                                                 for n in range(-grid_size, grid_size+1)])

# # Compute the Voronoi diagram
# vor = Voronoi(lattice_points)

# # Function to fold a point into the Wigner-Seitz cell
# def fold_to_wigner_seitz(point, lattice_points):
#     """Finds the closest equivalent point inside the Wigner-Seitz cell."""
#     distances = np.linalg.norm(lattice_points - point, axis=1)  # Compute distances to all lattice points
#     closest_lattice_point = lattice_points[np.argmin(distances)]  # Find the nearest lattice point
#     return point - closest_lattice_point  # Translate the point into the Wigner-Seitz cell

# # Fold all grid points
# folded_grid_points = np.array([fold_to_wigner_seitz(pt, lattice_points) for pt in grid_points])

# # Accumulate density values for folded positions
# # Since multiple original points may fold to the same position, we sum their densities
# unique_positions, indices = np.unique(np.round(folded_grid_points, decimals=6), axis=0, return_inverse=True)
# folded_density = np.zeros(len(unique_positions))
# for i, idx in enumerate(indices):
#     folded_density[idx] += density_matrix[i]  # Sum contributions

# # Plot Wigner-Seitz Cell and the Folded Density
# fig, ax = plt.subplots(figsize=(6,6))
# ax.set_xlim(-1.5, 1.5)
# ax.set_ylim(-1.5, 1.5)
# ax.set_aspect('equal')

# # Plot Voronoi diagram (Wigner-Seitz Cell)
# for simplex in vor.ridge_vertices:
#     simplex = np.array(simplex)
#     if np.all(simplex >= 0):  # Ignore points at infinity
#         ax.plot(vor.vertices[simplex, 0], vor.vertices[simplex, 1], 'k-')

# # Scatter plot of folded density values with colormap
# sc = ax.scatter(unique_positions[:, 0], unique_positions[:, 1], c=folded_density, cmap='viridis', marker='s', edgecolor='k')

# # Add colorbar
# plt.colorbar(sc, label="Density Intensity")

# plt.title("Folded Density in Wigner-Seitz Cell")
# plt.show()

################################################2D

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, Delaunay
from scipy.interpolate import griddata


#set random seed for reproducibility
np.random.seed(42)

# Define lattice basis vectors
a1 = np.array([1.0, 0.0])
a2 = np.array([0.5, np.sqrt(3)/2])
lattice_basis = np.array([a1, a2])

# Generate a 20x20 uniform grid in the primitive cell
N = 60  # Grid size
grid_x = np.linspace(0, 1, N, endpoint=False)  # Fractional coordinates
grid_y = np.linspace(0, 1, N, endpoint=False)
grid_points_frac = np.array([[x, y] for x in grid_x for y in grid_y])  # Flatten the grid

# Convert to Cartesian coordinates
grid_points = np.dot(grid_points_frac, lattice_basis)  # Fractional → Cartesian

# Generate a synthetic density matrix (random values as a toy dataset)
density_matrix = np.random.rand(N, N).flatten()  # Flatten to match grid points

# Generate a grid of nearby lattice points
grid_size = 2  # Extending the grid to consider neighboring unit cells
lattice_points = np.array([m * a1 + n * a2 for m in range(-grid_size, grid_size+1) 
                                                for n in range(-grid_size, grid_size+1)])

# Compute the Voronoi diagram
vor = Voronoi(lattice_points)

# Function to fold a point into the Wigner-Seitz cell
def fold_to_wigner_seitz(point, lattice_points):
    """Finds the closest equivalent point inside the Wigner-Seitz cell."""
    distances = np.linalg.norm(lattice_points - point, axis=1)  # Compute distances to all lattice points
    closest_lattice_point = lattice_points[np.argmin(distances)]  # Find the nearest lattice point
    return point - closest_lattice_point  # Translate the point into the Wigner-Seitz cell

# Fold all grid points
folded_grid_points = np.array([fold_to_wigner_seitz(pt, lattice_points) for pt in grid_points])

# Accumulate density values for folded positions
unique_positions, indices = np.unique(np.round(folded_grid_points, decimals=6), axis=0, return_inverse=True)
folded_density = np.zeros(len(unique_positions))
for i, idx in enumerate(indices):
    folded_density[idx] += density_matrix[i]  # Sum contributions

# Create a uniform rectangular grid for interpolation
grid_res = 30  # Resolution of the upsampled image
x_min, x_max = unique_positions[:, 0].min(), unique_positions[:, 0].max()
y_min, y_max = unique_positions[:, 1].min(), unique_positions[:, 1].max()
xi = np.linspace(x_min, x_max, grid_res)
yi = np.linspace(y_min, y_max, grid_res)
xi, yi = np.meshgrid(xi, yi)

# Interpolate the scattered data onto the uniform grid
zi = griddata(unique_positions, folded_density, (xi, yi), method='cubic')

# Plot the padded & interpolated density as an image
fig, ax = plt.subplots(figsize=(6,6))
im = ax.imshow(zi, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='viridis', aspect='auto')
plt.colorbar(im, label="Interpolated Density")
plt.title("Upsampled Density Map in Wigner-Seitz Cell")
plt.show()

import numpy as np
import plotly.graph_objects as go
from scipy.spatial import Voronoi, cKDTree
from scipy.interpolate import griddata

# Define 3D lattice basis vectors
a1 = np.array([1.0, 0.0, 0.0])
a2 = np.array([0.5, np.sqrt(3)/2, 0.0])
a3 = np.array([0,0,2])
lattice_basis = np.array([a1, a2, a3])

# Generate a uniform 3D grid in the primitive cell
N = 70  # Grid size in each direction
grid_x = np.linspace(0, 1, N, endpoint=False)
grid_y = np.linspace(0, 1, N, endpoint=False)
grid_z = np.linspace(0, 1, 5, endpoint=False)
grid_points_frac = np.array([[x, y, z] for x in grid_x for y in grid_y for z in grid_z])

# Convert to Cartesian coordinates
grid_points = np.dot(grid_points_frac, lattice_basis)

# Generate a synthetic density matrix (random values as a toy dataset)
density_matrix = np.random.rand(len(grid_points))

# Generate a grid of nearby lattice points
grid_size = 2
lattice_points = np.array([
    m * a1 + n * a2 + p * a3
    for m in range(-grid_size, grid_size+1)
    for n in range(-grid_size, grid_size+1)
    for p in range(-grid_size, grid_size+1)
])

# Compute the 3D Voronoi diagram
vor = Voronoi(lattice_points)

# Function to fold points into the Wigner-Seitz cell
def fold_to_wigner_seitz(points, lattice_points):
    tree = cKDTree(lattice_points)
    _, indices = tree.query(points)
    return points - lattice_points[indices]

# Fold all grid points
folded_grid_points = fold_to_wigner_seitz(grid_points, lattice_points)

# Accumulate density values at unique positions
unique_positions, indices = np.unique(np.round(folded_grid_points, decimals=6), axis=0, return_inverse=True)
folded_density = np.zeros(len(unique_positions))
for i, idx in enumerate(indices):
    folded_density[idx] += density_matrix[i]

# Interpolation for 3D visualization
grid_res = 30
xi, yi, zi = np.mgrid[
    unique_positions[:, 0].min():unique_positions[:, 0].max():grid_res*1j,
    unique_positions[:, 1].min():unique_positions[:, 1].max():grid_res*1j,
    unique_positions[:, 2].min():unique_positions[:, 2].max():grid_res*1j
]
zi_interp = griddata(unique_positions, folded_density, (xi, yi, zi), method='linear', fill_value=0)

# Plot 3D Density using Plotly
fig = go.Figure(data=go.Volume(
    x=xi.flatten(), y=yi.flatten(), z=zi.flatten(),
    value=zi_interp.flatten(),
    opacity=0.2, surface_count=20, colorscale='Viridis'
))
fig.update_layout(title='3D Wigner-Seitz Density Visualization')
fig.show()
