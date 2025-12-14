import os
import json
import subprocess
import seekpath
import numpy as np
import re
from ase.io import read


def get_materials(data_path):
    """
    Get a list of materials in the data directory.
    Materials are directories that contain 'mat' in their name, except 'manager'.
    """
    materials = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d)) and 'mat' in d and d != 'manager']
    materials.sort()
    return materials


def get_chemical_formula(cif_path):
    with open(cif_path, "r") as cif_file:
        for line in cif_file:
            if line.strip().startswith("_chemical_formula_structural"):
                formula = line.split(None, 1)[-1].strip()
                return formula
    return "unknown"


def get_num_electrons(scf_out_path):
    """
    Read the number of electrons from the scf.out file.
    """
    with open(scf_out_path, "r") as scf_out:
        for line in scf_out:
            if "number of electrons" in line:
                num_electrons = float(line.split()[-1])
                return num_electrons
    raise ValueError("Could not find number of electrons in scf.out")


def get_fermi_energy(scf_out_path):
    """
    Read the Fermi energy from the scf.out file.
    """
    with open(scf_out_path, "r") as scf_out:
        for line in scf_out:
            if "Fermi energy" in line:
                # Example line: "  the Fermi energy is     0.2969 ev"
                for part in line.strip().split():
                    try:
                        fermi_energy = float(part)
                        return fermi_energy
                    except ValueError:
                        continue
    raise ValueError("Could not find Fermi energy in scf.out")
   

def get_evs(out_path, num_bands, num_kpoints, calculation_type):
    '''
    Get the energy values for each band and k-point from the output file.
    Return will look like a 2D numpy array with shape (num_bands, num_kpoints).
    '''
    
    final_energies = np.zeros((num_bands, num_kpoints), dtype=float)
    with open(out_path, "r") as out_file:
        if calculation_type == "nscf":
            def sanitize_e_line(s: str) -> str:
                # Insert space before '-' that follows a digit and is followed by a digit
                # (e.g., turn '123-456' into '123 -456'). Does not touch 'E-3' or 'e-3'.
                s = re.sub(r'(?<=\d)-(?=\d)', ' -', s)
                # Optional: do the same for '+', if QE ever wraps positives together
                s = re.sub(r'(?<=\d)\+(?=\d)', ' +', s)
                return s
            
            # advance until the line that contains the marker
            for line in out_file:
                if "End of band structure calculation" in line:
                    break
            
            # now out_file is positioned *after* that line
            # read subsequent lines using next() as needed
            try:
                k_idx = 0
                while True:
                    line = next(out_file)
                    if "bands (ev)" in line:
                        next(out_file)  # skip the empty line
                        temp_evs = []   # init array
                        e_line = next(out_file)  # read first line with energies
                        while e_line.strip():
                            e_line = sanitize_e_line(e_line)
                            temp_evs.extend(float(x) for x in e_line.split())
                            e_line = next(out_file)
                        final_energies[:, k_idx] = temp_evs
                        k_idx += 1
            except StopIteration:
                # end of file reached
                pass

        elif calculation_type == "wan":
            for line in out_file:
                band_idx, kpt_idx, energy = line.split()
                band_idx = int(band_idx) - 1
                kpt_idx = int(kpt_idx) - 1
                final_energies[band_idx, kpt_idx] = float(energy)

    return final_energies


def get_mp_grid_kpoints(json_path):
    '''
    Generate k-points using Wannier90's kmesh.pl. For example, if the json parameter
    is "mp_grid": ["4", "4", "2"], the return might look like:
    ["K_POINTS crystal", "32", "0.00000000  0.00000000  0.00000000  3.125000e-02",
    "0.00000000  0.00000000  0.50000000  3.125000e-02",
    "0.00000000  0.25000000  0.00000000  3.125000e-02", ...]
    '''
    with open(json_path, "r") as param_file:
        params = json.load(param_file)

    kmeshpl_path = params["paths"]["kmeshpl"]
    kmesh = params["win"]["mp_grid"]
    result = subprocess.run(
        [kmeshpl_path] + kmesh,
        check=True,
        capture_output=True,
        text=True
    )
    # Return an array of k-point strings to be printed to nscf and .win
    return result.stdout.strip().splitlines()


class HighSymmetryKPoints:
    def __init__(self, cif_path):
        self.atoms = read(cif_path)
        cell = self.atoms.get_cell().tolist()
        positions = self.atoms.get_scaled_positions().tolist()
        numbers = self.atoms.get_atomic_numbers().tolist()
        self.kp = seekpath.get_explicit_k_path((cell, positions, numbers))

    def get_segment_indices(self):
        '''
        Returns the indices of the start of end kpoints of each segment. For example, 
        a segment with a connected and two disconnected segments might return:
        [165, 196], [196, 217], [218, 227], [228, 237],
        '''
        index_pairs = []
        for start, end in self.kp["explicit_segments"]:
            # end index in explicit_kpoints_rel needs subtracting 1
            # (because it is inclusive in the segments)
            index_pairs.append([start, end - 1])
        return index_pairs

    def get_segment_labels(self):
        '''
        Returns the labels of the start and end kpoints of each segment.
        For example, a segment with a connected and two disconnected segments might return:
        ['R', 'T'], ['T', 'Z'], ['X', 'U'], ['Y', 'T'],
        '''
        label_pairs = []
        for start, end in self.kp["explicit_segments"]:
            raw_labels = self.kp["explicit_kpoints_labels"][slice(start, end)]
            label_pairs.append([raw_labels[0], raw_labels[-1]])
        return label_pairs

    def _format_kpt(self, kpt):
        # Concatenate three floats with space delimeter.
        # Also, add one leading spaces if positive to align with negative numbers
        parts = []
        for x in kpt:
            s = f"{x:.6f}"
            if x >= 0:
                s = " " + s
            parts.append(s)
        return " ".join(parts)

    def _format_label(self, label):
        # 1) Change GAMMA to G
        # 2) Add spaces based on label length
        #      add three spaces if label is a single character, two spaces if two characters,
        #      and one space if three characters. 
        if label == "GAMMA":
            label = "G"
        if len(label) == 1:
            return f"{label}   "
        elif len(label) == 2:
            return f"{label}  "
        else:
            return f"{label} "

    def get_hskps_win(self):
        '''
        Returns a list of strings representing the high-symmetry k-point segments
        in the format expected by wannier90's .win files. Return might look like:
        ["  G     0.000000  0.000000  0.000000     X     0.500000  0.000000  0.000000",
        "  X     0.500000  0.000000  0.000000     S     0.500000  0.500000  0.000000",
        "  S     0.500000  0.500000  0.000000     Y     0.000000  0.500000  0.000000"]
        '''
        index_pairs = self.get_segment_indices()
        label_pairs = self.get_segment_labels()
        
        segs = []
        for i in range(len(index_pairs)):
            lbl0 = self._format_label(label_pairs[i][0])
            kpt0 = self._format_kpt(self.kp['explicit_kpoints_rel'][index_pairs[i][0]])
            lbl1 = self._format_label(label_pairs[i][1])
            kpt1 = self._format_kpt(self.kp['explicit_kpoints_rel'][index_pairs[i][1]])
            
            segment_str = f"  {lbl0} {kpt0}     {lbl1} {kpt1}"
            segs.append(segment_str)
        return segs

    def get_hskps_bands(self):
        '''
        Returns a list of strings representing the high-symmetry k-point segments
        in the format expected by pw.x bands input. Return might look like:
        ["K_POINTS crystal_b", "11", 
        "0.000000  0.000000  0.000000  35  !G  ", 
        "0.000000  0.500000  0.000000  8  !Z   ",
        "0.000000  0.500000  0.500000  35  !D   "]
        '''
        index_pairs = self.get_segment_indices()
        label_pairs = self.get_segment_labels()
        
        # Complicated printing loop to tape together connected and disconnected segments
        # into a single block for pw.x bands input.
        lines = []
        kpt_init = self._format_kpt(self.kp['explicit_kpoints_rel'][index_pairs[0][0]])
        lines.append(kpt_init)

        for i in range(len(index_pairs)):
            lbl0 = self._format_label(label_pairs[i][0])
            kpt0 = self._format_kpt(self.kp['explicit_kpoints_rel'][index_pairs[i][0]])
            lbl1 = self._format_label(label_pairs[i][1])
            kpt1 = self._format_kpt(self.kp['explicit_kpoints_rel'][index_pairs[i][1]])
            interpolation = index_pairs[i][1] - index_pairs[i][0] - 1

            if kpt0 == lines[-1]:
                # If the segment is connected to the previous, continue printing on that line
                lines[-1] += f"  {interpolation}  !{lbl0}"
                lines.append(kpt1)
            else:
                # If the segment is disconnected, finish previous line with 0 interpolation
                old_lbl = self._format_label(label_pairs[i - 1][1])
                lines[-1] += f"  0  !{old_lbl}"
                # Start new line
                lines.append(kpt0)
                lines[-1] += f"  {interpolation}  !{lbl0}"
                lines.append(kpt1)
        # Finish the last line with its label
        last_lbl = self._format_label(label_pairs[-1][1])
        lines[-1] += f"  0  !{last_lbl}"
        
        return lines


class EigController:
    def __init__(self, eig_path, num_electrons, num_bands, num_wann):
        self.eig_path = eig_path
        self.fermi_index = num_electrons / 2
        self.num_bands = num_bands
        self.band_radius = num_wann / 2
        with open(self.eig_path, "r") as eig_file:
            eig_lines = eig_file.readlines()
        
        # Populate eig_data
        num_kpoints = int(eig_lines[-1].split()[1])
        self.eig_data = np.zeros((self.num_bands, num_kpoints), dtype=float)
        for eig_line in eig_lines:
            band = int(eig_line.split()[0]) - 1
            kpt = int(eig_line.split()[1]) - 1
            energy = float(eig_line.split()[2])
            self.eig_data[band, kpt] = energy

    def get_eig_data(self):
        # Sanity check: ensure that band energies do not decrease
        for i in range(len(self.eig_data[0])):
            for j in range(1, len(self.eig_data)):
                if self.eig_data[j, i] < self.eig_data[j - 1, i]:
                    print(f"Warning: Band energies decrease at k-point {i} between bands {j-1} and {j}")
        
        return self.eig_data

    def get_windows(self): # OBSOLETE
        bottom_idx = int(self.fermi_index - self.band_radius)
        top_idx = int(self.fermi_index + self.band_radius)

        dis_bottom = np.min(self.eig_data[bottom_idx:top_idx, :]) - 0.001
        dis_top = np.max(self.eig_data[bottom_idx:top_idx, :]) + 0.001

        froz_bottom = np.max(self.eig_data[:bottom_idx, :]) + 0.001
        froz_top = np.min(self.eig_data[top_idx:, :]) - 0.001

        return (dis_bottom, dis_top, froz_bottom, froz_top)

    @staticmethod
    def get_exclude_bands(num_electrons, num_wann):
        fermi_index = num_electrons / 2
        band_radius = num_wann / 2
        
        lower_cutoff = int(fermi_index - band_radius)
        upper_cutoff = int(fermi_index + band_radius) + 1

        return (lower_cutoff, upper_cutoff)


if __name__ == "__main__":
    # Just testing the new get_evs functions
    import sys
    out_path = sys.argv[1]
    num_bands = int(sys.argv[2])
    num_kpoints = int(sys.argv[3])
    calc_type = sys.argv[4]  # "nscf" or "bands"
    evs = get_evs(out_path, num_bands, num_kpoints, calc_type)
    print(evs)


# def get_high_symmetry_kpoints(cif_path):
#     atoms = read(cif_path)
    
#     seek_cell = atoms.get_cell().tolist()
#     seek_positions = atoms.get_scaled_positions().tolist()
#     seek_numbers = atoms.get_atomic_numbers().tolist()
#     kp = seekpath.get_explicit_k_path((seek_cell, seek_positions, seek_numbers))

#     # Get the indices and labels of the high-symmetry segments
#     indices = get_seek_segment_indices(kp)
#     labels = get_seek_segment_labels(kp)

#     # Contruct the high-symmetry k-point strings to be written into .win
#     high_symmetry_segments = []
#     for i in range(len(indices)):
#         kpt0 = " ".join(f"{x:.6f}" for x in kp['explicit_kpoints_rel'][indices[i][0]])
#         kpt1 = " ".join(f"{x:.6f}" for x in kp['explicit_kpoints_rel'][indices[i][1]])
#         segment_str = f"  {labels[i][0]} {kpt0}    {labels[i][1]} {kpt1}"
#         high_symmetry_segments.append(segment_str)
#     return high_symmetry_segments

# def get_seek_segment_indices(kp):
#     segments = []
#     for i in range(len(kp["explicit_segments"])):
#         segment_indices = list(kp["explicit_segments"][i])
#         # Usable segment indices need to subtract 1
#         segment_indices[1] -= 1
#         segments.append(segment_indices)
#     return segments

# def get_seek_segment_labels(kp):
#     labels = []
#     for i in range(len(kp["explicit_segments"])):
#         segment_indices = kp["explicit_segments"][i]
#         # These segment indices are used for slicing, so no need to subtract 1
#         raw_labels = kp["explicit_kpoints_labels"][slice(*segment_indices)]
#         label = (raw_labels[0], raw_labels[-1])
#         if label[0] == "GAMMA":
#             label = ("G", label[1])
#         if label[1] == "GAMMA":
#             label = (label[0], "G")
#         labels.append(label)
#     return labels
