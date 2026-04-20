import numpy as np
import matplotlib.pyplot as plt
import re
import h5py
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import get_num_electrons, get_chemical_formula
from step_helpers import psi_slice_to_3_panels

def read_wfck2r(filename):
    """
    Read the unkr array from a QE wfck2r output file.
    RETURNS:
    unkr: (nr1x, nr2x, nr3x, nbnd, nk) complex array of plane-wave coefficients
    (nr1x, nr2x, nr3x): grid sizes along each reciprocal lattice vector
    """
    with open(filename, "r") as f:
        text = f.read()

    # read grid sizes
    nr1x = int(re.search(r"# name: nr1x\s+# type: scalar\s+(\d+)", text, re.S).group(1))
    nr2x = int(re.search(r"# name: nr2x\s+# type: scalar\s+(\d+)", text, re.S).group(1))
    nr3x = int(re.search(r"# name: nr3x\s+# type: scalar\s+(\d+)", text, re.S).group(1))

    # find the unkr block
    m = re.search(
        r"# name: unkr\s+# type: complex matrix\s+# ndims: 5\s+"
        r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*",
        text,
        re.S
    )
    dims = tuple(map(int, m.groups()))

    # parse all complex numbers after that point
    data_text = text[m.end():]
    pairs = re.findall(r"\(\s*([Ee0-9+\-.]+)\s*,\s*([Ee0-9+\-.]+)\s*\)", data_text)

    vals = np.array([complex(float(r), float(i)) for r, i in pairs], dtype=np.complex128)

    # reshape using Fortran ordering
    unkr = vals.reshape(dims, order="F")

    return unkr, (nr1x, nr2x, nr3x)


def read_qe_wfc_hdf5(filename):
    """
    Read plane-wave coefficients from QE HDF5 file and convert to complex array.
    RETURNS:
        miller: (npw, 3) integer Miller indices
        evc: (nbnd, npw) complex plane-wave coefficients
        attrs: dict of file attributes
        mi_attrs: dict of MillerIndices dataset attributes
        evc_attrs: dict of evc dataset attributes
    """
    with h5py.File(filename, "r") as f:
        miller = f["MillerIndices"][:]          # (npw, 3)
        evc_raw = f["evc"][:]                   # (nbnd, 2*npw)

        attrs = dict(f.attrs)
        mi_attrs = dict(f["MillerIndices"].attrs)
        evc_attrs = dict(f["evc"].attrs)

    nbnd = evc_raw.shape[0]
    npw = miller.shape[0]

    if evc_raw.shape[1] != 2 * npw:
        raise ValueError(
            f"evc second dimension is {evc_raw.shape[1]}, expected 2*npw = {2*npw}"
        )

    # Convert real/imag pairs into complex coefficients
    evc = evc_raw[:, 0::2] + 1j * evc_raw[:, 1::2]   # shape (nbnd, npw)

    return miller, evc, attrs, mi_attrs, evc_attrs

def coeffs_to_realspace_qegrid(miller, coeffs_band, nr1x, nr2x, nr3x):
    """
    Reconstruct periodic part u_nk(r) from plane-wave coefficients.

    miller: (npw, 3) integer Miller indices
    coeffs_band: (npw,) complex coefficients for one band
    grid_shape: (n1, n2, n3). Inherited from the QE native FFT.
    RETURNS:
        psi_manual: (n1, n2, n3) complex array of the reconstructed wavefunction on the real-space grid
    """
    fft_grid = np.zeros((nr1x, nr2x, nr3x), dtype=np.complex128)

    i1 = miller[:, 0] % nr1x
    i2 = miller[:, 1] % nr2x
    i3 = miller[:, 2] % nr3x

    fft_grid[i1, i2, i3] = coeffs_band

    psi_manual = np.fft.ifftn(fft_grid)
    return psi_manual

def plot_wfc_hdf5(filename, band_index, ik, nrx1, nrx2, nrx3):
    """
    Generates the 3-panel plot for a given band and k-point from the HDF5 file.
    Leaves the saving and titling to the caller outside the function.
    """
    miller, evc, _1, _2, _3 = read_qe_wfc_hdf5(filename)

    psi_manual = coeffs_to_realspace_qegrid(miller, evc[band_index], nrx1, nrx2, nrx3)

    n1, n2, n3 = psi_manual.shape
    iz = n3 // 2
    psi_slice = psi_manual[:, :, iz].T

    fig, axes = psi_slice_to_3_panels(psi_slice)

    for ax in axes:
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")

    return fig, axes


def read_u_mat(filename):
    """
    Reads the U matrices from a Wannier90 .mat file.
    RETURNS:
    kpts: (nk, 3) array of k-point coordinates
    U: (nk, num_bands, num_wann) array of unitary matrices
    """
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # skip timestamp/header line
    header_idx = 1
    nk, num_wann, num_bands = map(int, lines[header_idx].split())

    cursor = header_idx + 1

    kpts = np.zeros((nk, 3), dtype=float)
    U = np.zeros((nk, num_bands, num_wann), dtype=np.complex128)

    for ik in range(nk):
        # k-point line
        kpts[ik] = np.array(list(map(float, lines[cursor].split())))
        cursor += 1

        # num_bands * num_wann complex entries
        vals = []
        for _ in range(num_bands * num_wann):
            re_im = lines[cursor].split()
            vals.append(float(re_im[0]) + 1j * float(re_im[1]))
            cursor += 1

        # Wannier90 writes in Fortran order
        U[ik] = np.array(vals, dtype=np.complex128).reshape(
            (num_bands, num_wann), order="F"
        )

    return kpts, U

def rotate_coeffs_with_U(evc, U_k, band_indices):
    """
    evc: (nbnd, npw) complex
    U_k: (num_bands, num_wann) complex
    band_indices: the indices over which Wannier90 performed the disentanglement/rotation.
    RETURNS:
    C_rot: (npw, num_wann) complex array of rotated coefficients
    """
    C_all = evc.T                              # (npw, nbnd)
    C_sub = C_all[:, band_indices]             # (npw, num_bands)

    if C_sub.shape[1] != U_k.shape[0]:
        raise ValueError(
            f"Band subspace has {C_sub.shape[1]} bands but U_k expects {U_k.shape[0]}"
        )

    C_rot = C_sub @ U_k                        # (npw, num_wann)
    return C_rot

def write_rotated_wfc_to_hdf5(outfile, miller, C_rot, file_attrs=None, mi_attrs=None, evc_attrs=None):
    """
    miller: (npw, 3)
    C_rot: (npw, nwann) complex
    Writes the rotated coefficients to an HDF5 file in the same format as QE, with real/imag interleaved.
    Allows it to be picked up by plot_wfc_hdf5.
    RETURNS:
    None
    """
    npw, nwann = C_rot.shape

    # convert back to QE-style real/imag interleaved storage:
    # output shape (nwann, 2*npw)
    evc_out = np.empty((nwann, 2 * npw), dtype=np.float64)
    evc_out[:, 0::2] = C_rot.T.real
    evc_out[:, 1::2] = C_rot.T.imag

    with h5py.File(outfile, "w") as f:
        if file_attrs is not None:
            for k, v in file_attrs.items():
                try:
                    f.attrs[k] = v
                except Exception:
                    pass

        dset_mi = f.create_dataset("MillerIndices", data=miller)
        if mi_attrs is not None:
            for k, v in mi_attrs.items():
                try:
                    dset_mi.attrs[k] = v
                except Exception:
                    pass

        dset_evc = f.create_dataset("evc", data=evc_out)
        if evc_attrs is not None:
            for k, v in evc_attrs.items():
                try:
                    dset_evc.attrs[k] = v
                except Exception:
                    pass


if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print("Usage: python step1.py <json_path>")
        sys.exit(1)
    
    json_path = sys.argv[1]

    with open(json_path, "r") as param_file:
        params = json.load(param_file)
    
    seedname = get_chemical_formula("../stru.cif")
    num_electrons = get_num_electrons("../1-scf/scf.out")
    num_wann = params["win"]["num_wann"]

    k_indices = [0, 2, 4, 6, 8, 10]

    ########## THE QE WFCK2R FFT WFNS FOR VERIFICATION ############
    filename = "wfck2r.oct"
    unkr, (nr1x, nr2x, nr3x) = read_wfck2r(filename)

    band_indices = [0, 1]

    # --- Plotting Logic ---
    for band_index in band_indices:
        # Assuming you are iterating through k-points
        for ik in range(unkr.shape[4]):
            # Data extraction
            psi = unkr[:, :, :, band_index, ik]
            iz = nr3x // 2
            psi_slice = psi[:, :, iz].T  # Transpose here once for all plots        
            
            # Create the 3-panel plot
            fig, axes = psi_slice_to_3_panels(psi_slice)

            # Formatting
            for ax in axes:
                ax.set_xlabel("x index")
                ax.set_ylabel("y index")
            
            fig.suptitle(f"Wavefunction Slice at band: {band_index+1}, k-point {ik+1}", fontsize=16)
            plt.savefig(f"wfn_bnd{band_index+1}_k{ik+1}.png", dpi=300)
            plt.close(fig)
    ##################################

    ######### THE MANUAL FFT OF THE WFNS #########
    band_indices = [int(num_electrons//2 - num_wann//2), int(num_electrons//2 - num_wann//2 + 1)]  # top valence and bottom conduction bands # TODO REMOVE -NUM_WANN//2
    print(f"Using band indices for manual FFT: {band_indices}")
    
    for band_index in band_indices:
        for ik in k_indices:
            filename = f"{seedname}.save/wfc{ik+1}.hdf5"

            fig, axes = plot_wfc_hdf5(filename, 
                                    band_index, 
                                    ik, 
                                    nr1x, nr2x, nr3x)
            
            fig.suptitle(f"HDF5 manual inverse FFT, band {band_index+1}, k-point {ik+1}", fontsize=16)
            plt.savefig(f"wfn_manual_bnd{band_index+1}_k{ik+1}.png", dpi=300)
            plt.close(fig)
    #######################################

    ########## THE ROTATED WFNS ##########
    u_filename = f"{seedname}_u.mat"
    kpts, U_all = read_u_mat(u_filename)

    band_indices = [0, 1] # TODO MAKE NUM_WANN//2
    
    wannier_window_band_indices = np.arange(int(num_electrons//2 - num_wann//2), 
                                            int(num_electrons//2 + num_wann//2))
    print(f"Using Wannier window band indices: {wannier_window_band_indices}")

    for band_index in band_indices:
        for ik in k_indices:
            wfc_filename = f"{seedname}.save/wfc{ik+1}.hdf5"
            miller, evc, file_attrs, mi_attrs, evc_attrs = read_qe_wfc_hdf5(wfc_filename)

            U_k = U_all[ik]

            C_rot = rotate_coeffs_with_U(evc, U_k, wannier_window_band_indices)

            write_rotated_wfc_to_hdf5(
                f"rotated.save/wfc{ik+1}_rotated.hdf5",
                miller,
                C_rot,
                file_attrs=file_attrs,
                mi_attrs=mi_attrs,
                evc_attrs=evc_attrs
            )

            fig, axes = plot_wfc_hdf5(f"rotated.save/wfc{ik+1}_rotated.hdf5", 
                                    band_index, 
                                    ik, 
                                    nr1x, nr2x, nr3x)
            
            fig.suptitle(f"Rotated WFC, band {band_index+1}, k-point {ik+1}", fontsize=16)

            plt.savefig(f"wfn_rotated_bnd{band_index+1}_k{ik+1}.png", dpi=300)
            plt.close(fig)




# if i == 0:
#     # First compute all heatmaps
#     heats = []
#     for ik in k_indices:
#         psi = unkr[:, :, :, 0, ik]
#         heat = np.abs(psi[:, :, iz])**2
#         heats.append(heat)

#     # Shared color scale
#     vmin = 0
#     vmax = max(h.max() for h in heats)

#     for ax, heat, title in zip(axes.flat, heats, titles):
#         im = ax.imshow(
#             heat.T,
#             origin="lower",
#             aspect="equal",
#             vmin=vmin,
#             vmax=vmax
#         )
#         ax.set_title(title)
#         ax.set_xlabel("x index")
#         ax.set_ylabel("y index")

#     fig.colorbar(im, ax=axes, shrink=0.8, label=r"$|\psi|^2$")
#     plt.savefig("psi_slice_real.png", dpi=300)

# elif i == 1:
#     for ax, ik, title in zip(axes.flat, k_indices, titles):
#         psi = unkr[:, :, :, 1, ik]
#         iz = nr3x // 2
#         psi_slice = psi[:, :, iz]

#         rgb = complex_to_rgb(psi_slice)

#         ax.imshow(rgb.transpose(1, 0, 2), origin="lower", aspect="equal")
#         ax.set_xlabel("x index")
#         ax.set_ylabel("y index")
#         ax.set_title(f"Complex wavefunction: {title}")

#     plt.savefig("psi_slice_imag2.png", dpi=300)

# elif i == 2:
#     phases = np.linspace(-np.pi, np.pi, 500)
#     phase_img = np.tile(phases, (30, 1))

#     hue = (phase_img + np.pi) / (2*np.pi)
#     sat = np.ones_like(hue)
#     val = np.ones_like(hue)

#     phase_rgb = hsv_to_rgb(np.stack([hue, sat, val], axis=-1))

#     plt.figure(figsize=(8, 1.5))
#     plt.imshow(phase_rgb, aspect="auto", origin="lower",
#             extent=[-np.pi, np.pi, 0, 1])
#     plt.yticks([])
#     plt.xlabel("phase")
#     plt.title("Phase color key")
#     plt.savefig("phase_color_key.png", dpi=300)

# elif i == 3:
#     ik = 2
#     psi = unkr[:, :, :, 0, ik]
#     psi_slice = psi[:, :, iz]

#     axes = axes.flatten()

#     # Magnitude
#     im0 = axes[0].imshow(
#         np.abs(psi_slice).T,
#         origin="lower",
#         aspect="equal",
#         cmap="viridis"
#     )
#     axes[0].set_title(r"$|\psi|$")
#     fig.colorbar(im0, ax=axes[0])

#     # Real part: symmetric around 0
#     real_part = np.real(psi_slice).T
#     vmax_real = np.max(np.abs(real_part))
#     im1 = axes[1].imshow(
#         real_part,
#         origin="lower",
#         aspect="equal",
#         cmap="RdBu_r",
#         vmin=-vmax_real,
#         vmax=vmax_real
#     )
#     axes[1].set_title(r"$\mathrm{Re}(\psi)$")
#     fig.colorbar(im1, ax=axes[1])

#     # Imag part: symmetric around 0
#     imag_part = np.imag(psi_slice).T
#     vmax_imag = np.max(np.abs(imag_part))
#     im2 = axes[2].imshow(
#         imag_part,
#         origin="lower",
#         aspect="equal",
#         cmap="RdBu_r",
#         vmin=-vmax_imag,
#         vmax=vmax_imag
#     )
#     axes[2].set_title(r"$\mathrm{Im}(\psi)$")
#     fig.colorbar(im2, ax=axes[2])

#     # Phase: fixed range from -pi to pi
#     phase = np.angle(psi_slice).T
#     im3 = axes[3].imshow(
#         phase,
#         origin="lower",
#         aspect="equal",
#         cmap="twilight",
#         vmin=-np.pi,
#         vmax=np.pi
#     )
#     axes[3].set_title(r"$\arg(\psi)$")
#     fig.colorbar(im3, ax=axes[3])

#     for ax in axes:
#         ax.set_xlabel("x index")
#         ax.set_ylabel("y index")

#     plt.savefig("psi_slice_all.png", dpi=300)