import json
from ase.io import read
from .helpers import get_chemical_formula, get_mp_grid_kpoints, HighSymmetryKPoints


def generate_qe_ins(calc_type, cif_path, json_path):
    atoms = read(cif_path)
    with open(json_path, "r") as param_file:
        params = json.load(param_file)
    formula = get_chemical_formula(cif_path)

    with open(f"{calc_type}.in", "w") as f:
        f.write('&CONTROL\n')
        f.write(f'  calculation  = \'{calc_type}\'\n')
        f.write(f'  prefix       = \'{formula}\'\n')
        f.write(f'  outdir       = \'{params["control"]["outdir"]}\'\n')
        f.write(f'  pseudo_dir   = \'{params["control"]["pseudo_dir"]}\'\n')
        f.write(f'  verbosity    = \'{params["control"]["verbosity"]}\'\n')
        f.write(f'  restart_mode = \'{params["control"]["restart_mode"]}\'\n')
        f.write('/\n\n')

        f.write('&SYSTEM\n')
        nat = len(atoms.get_chemical_symbols())
        ntyp = len(set(atoms.get_chemical_symbols()))
        f.write(f'  nat         = {nat}\n')
        f.write(f'  ntyp        = {ntyp}\n')
        f.write(f'  ibrav       = {params["system"]["ibrav"]}\n')
        f.write(f'  ecutwfc     = {params["system"]["ecutwfc"]}\n')
        f.write(f'  occupations = \'{params["system"]["occupations"]}\'\n')
        f.write(f'  smearing    = \'{params["system"]["smearing"]}\'\n')
        f.write(f'  degauss     = {params["system"]["degauss"]}\n')
        if calc_type == "nscf":
            f.write(f'  nbnd        = {params["system"]["nbnd"]}\n')
            f.write(f'  nosym       = {params["system"]["nosym"]}\n')
            f.write(f'  noinv       = {params["system"]["noinv"]}\n')
        elif calc_type == "bands":
            f.write(f'  nbnd        = {params["system"]["nbnd"]}\n')
        f.write('/\n\n')

        f.write('&ELECTRONS\n')
        f.write(f'  electron_maxstep = {params["electrons"]["electron_maxstep"]}\n')
        f.write(f'  conv_thr         = {params["electrons"]["conv_thr"]}\n')
        f.write(f'  mixing_mode      = \'{params["electrons"]["mixing_mode"]}\'\n')
        f.write(f'  mixing_beta      = {params["electrons"]["mixing_beta"]}\n')
        f.write(f'  mixing_ndim      = {params["electrons"]["mixing_ndim"]}\n')
        f.write(f'  diagonalization  = \'{params["electrons"]["diagonalization"]}\'\n')
        f.write(f'  diago_david_ndim = {params["electrons"]["diago_david_ndim"]}\n')
        f.write(f'  diago_full_acc   = {params["electrons"]["diago_full_acc"]}\n')
        f.write('/\n\n')

        f.write('ATOMIC_SPECIES\n')
        for atom in set(atoms.get_chemical_symbols()):
            f.write(f'  {atom}  {params["atomic_species"][atom]}  {atom}.upf\n')
        f.write(f'\n')
        
        f.write("CELL_PARAMETERS angstrom\n")
        cell = atoms.get_cell()
        for v in cell:
            f.write(f"  {v[0]}  {v[1]}  {v[2]}\n")
        f.write('\n')

        f.write('ATOMIC_POSITIONS crystal\n')
        for symbol, frac in zip(atoms.get_chemical_symbols(), atoms.get_scaled_positions()):
            f.write(f'  {symbol}  {frac[0]}  {frac[1]}  {frac[2]}\n')
        f.write('\n')

        if calc_type == "scf":
            f.write('K_POINTS automatic\n')
            f.write(f'  {params["kpoints"]["automatic"]}\n')
        elif calc_type == "nscf":
            kpoints_lines = get_mp_grid_kpoints(json_path)
            for line in kpoints_lines:
                # Each line should be: kx ky kz weight
                f.write(f'{line}\n')
        elif calc_type == "bands":
            f.write('K_POINTS crystal_b\n')
            f.write(f'  ! {params["kpoints"]["high_symmetry"]}\n')
            # hskp = HighSymmetryKPoints(cif_path)
            # kpoints_lines = hskp.get_hskps_bands()
            
            # f.write(f'{len(kpoints_lines)}\n')
            # for line in kpoints_lines:
            #     f.write(f'  {line}\n')


def generate_wfck2r_in(cif_path, json_path):
    atoms = read(cif_path)
    with open(json_path, "r") as param_file:
        params = json.load(param_file)
    formula = get_chemical_formula(cif_path)

    with open(f"wfck2r.in", "w") as f:
        f.write('&inputpp\n')
        f.write(f'  prefix     = \'{formula}\'\n')
        f.write(f'  outdir     = \'{params["wfck2r"]["outdir"]}\'\n')
        f.write(f'  first_k    = {params["wfck2r"]["first_k"]}\n')
        f.write(f'  last_k     = {params["wfck2r"]["last_k"]}\n')
        f.write(f'  first_band = {params["wfck2r"]["first_band"]}\n')
        f.write(f'  last_band  = {params["wfck2r"]["last_band"]}\n')
        # f.write(f'  write_unk  = {params["wfck2r"]["write_unk"]}\n')
        f.write(f'  loctave    = {params["wfck2r"]["loctave"]}\n')
        f.write(f'/\n')


def generate_pw2bgw_in(cif_path, json_path):
    atoms = read(cif_path)
    with open(json_path, "r") as param_file:
        params = json.load(param_file)
    formula = get_chemical_formula(cif_path)

    with open(f"pw2bgw.in", "w") as f:
        f.write('&input_pw2bgw\n')
        f.write(f'  prefix          = \'{formula}\'\n')
        f.write(f'  outdir          = \'{params["pw2bgw"]["outdir"]}\'\n')
        f.write(f'  real_or_complex = {params["pw2bgw"]["real_or_complex"]}\n')
        f.write(f'  wfng_flag       = {params["pw2bgw"]["wfng_flag"]}\n')
        f.write(f'  wfng_file       = \'{params["pw2bgw"]["wfng_file"]}\'\n')
        f.write(f'/\n')

def generate_bands_pp_in(cif_path, json_path):
    atoms = read(cif_path)
    with open(json_path, "r") as param_file:
        params = json.load(param_file)
    formula = get_chemical_formula(cif_path)

    with open(f"bands_pp.in", "w") as f:
        f.write('&BANDS\n')
        f.write(f'  prefix  = \'{formula}\'\n')
        f.write(f'  outdir  = \'{params["control"]["outdir"]}\'\n')
        f.write(f'  filband = \'{formula}.bands.dat\'\n')
        f.write(f'/\n')


def generate_win(cif_path, json_path):
    atoms = read(cif_path)
    with open(json_path, "r") as param_file:
        params = json.load(param_file)
    formula = get_chemical_formula(cif_path)

    with open(f"{formula}.win", "w") as f:
        f.write(f'num_wann  = {params["win"]["num_wann"]}\n')
        f.write(f'num_bands = {params["win"]["num_bands"]}\n')
        f.write(f'spinors   = {params["win"]["spinors"]}\n')
        f.write(f'num_iter  = {params["win"]["num_iter"]}\n\n')

        f.write(f'! dis_froz_min  = {params["win"]["dis_froz_min"]}\n')
        f.write(f'! dis_froz_max  = {params["win"]["dis_froz_max"]}\n')
        f.write(f'! dis_win_min   = {params["win"]["dis_win_min"]}\n')
        f.write(f'! dis_win_max   = {params["win"]["dis_win_max"]}\n')
        f.write(f'! dis_num_iter  = {params["win"]["dis_num_iter"]}\n')
        f.write(f'! dis_mix_ratio = {params["win"]["dis_mix_ratio"]}\n\n')

        f.write(f'auto_projections = {params["win"]["auto_projections"]}\n\n')

        # f.write(f'conv_tol???)
        # f.write(f'conv_window???)

        f.write(f'! exclude_bands = {params["win"]["exclude_bands"]}\n\n')
        
        f.write(f'bands_plot       = {params["win"]["bands_plot"]}\n')
        f.write(f'bands_num_points = {params["win"]["bands_num_points"]}\n\n')

        f.write(f'begin kpoint_path\n')
        # f.write(f'  G 0.00000 0.00000 0.00000     X     0.50000 0.00000 0.00000\n')
        # f.write(f'  X 0.50000 0.00000 0.00000     J     0.33333 0.33333 0.00000\n')
        # f.write(f'  J 0.33333 0.33333 0.00000     G     0.00000 0.00000 0.00000\n')
        hskp = HighSymmetryKPoints(cif_path)
        for line in hskp.get_hskps_win():
            f.write(f'{line}\n')
        f.write(f'end kpoint_path\n\n')

        f.write(f'write_u_matrices = {params["win"]["write_u_matrices"]}\n')
        f.write(f'write_tb         = {params["win"]["write_tb"]}\n\n')
        
        f.write('begin unit_cell_cart\n')
        cell = atoms.get_cell()
        for v in cell:
            f.write(f'  {v[0]}  {v[1]}  {v[2]}\n')
        f.write('end unit_cell_cart\n\n')

        f.write('begin atoms_frac\n')
        for symbol, frac in zip(atoms.get_chemical_symbols(), atoms.get_scaled_positions()):
            f.write(f'  {symbol}  {frac[0]}  {frac[1]}  {frac[2]}\n')
        f.write('end atoms_frac\n\n')

        f.write(f'mp_grid = {" ".join(params["win"]["mp_grid"])}\n\n')

        f.write(f'begin kpoints\n')
        kpoints_lines = get_mp_grid_kpoints(json_path)
        for line in kpoints_lines[2:]:
            # Each line should be: kx ky kz weight
            f.write(f'{line}\n')
        f.write('end kpoints\n')


def generate_pw2wan(cif_path, json_path):
    atoms = read(cif_path)
    with open(json_path, "r") as param_file:
        params = json.load(param_file)
    formula = get_chemical_formula(cif_path)

    with open(f"{formula}.pw2wan", "w") as f:
        f.write(f'&inputpp\n')
        f.write(f'  outdir         = \'{params["control"]["outdir"]}\'\n')
        f.write(f'  prefix         = \'{formula}\'\n')
        f.write(f'  seedname       = \'{formula}\'\n')
        f.write(f'  spin_component = \'{params["pw2wan"]["spin_component"]}\'\n')
        f.write(f'  write_mmn      = {params["pw2wan"]["write_mmn"]}\n')
        f.write(f'  write_amn      = {params["pw2wan"]["write_amn"]}\n')
        f.write(f'  write_unk      = {params["pw2wan"]["write_unk"]}\n\n')

        f.write(f'  scdm_proj         = {params["pw2wan"]["scdm_proj"]}\n')
        f.write(f'  scdm_entanglement = \'{params["pw2wan"]["scdm_entanglement"]}\'\n')
        f.write(f'  scdm_mu           = {params["pw2wan"]["scdm_mu"]}\n')
        f.write(f'  scdm_sigma        = {params["pw2wan"]["scdm_sigma"]}\n')
        f.write(f'/\n')


# if  __name__ == "__main__":
#     if len(sys.argv) != 4:
#         print("Usage error: python generate_inputs.py <cif_path> <json_path> <calculation>")
#         sys.exit(1)
    
#     cif_path = sys.argv[1]
#     json_path = sys.argv[2]
#     calculation = sys.argv[3]
#     if not os.path.exists(cif_path):
#         print(f"Error: The file {cif_path} does not exist.")
#         sys.exit(1)
#     if not os.path.exists(json_path):
#         print(f"Error: The file {json_path} does not exist.")
#         sys.exit(1)

#     if calculation == "1-scf":
#         generate_scf_in(cif_path, json_path)
#     elif calculation == "2-nscf":
#         generate_nscf_in(cif_path, json_path)
    


# def generate_nscf_in(cif_path = "../stru.cif", json_path = "../../manager/parameters.json"):
#     atoms = read(cif_path)
#     with open(json_path, "r") as param_file:
#         params = json.load(param_file)
    
#     with open("nscf.in", "w") as f:
#         f.write('&CONTROL\n')
#         f.write(f'  calculation  = \'nscf\'\n')
#         f.write(f'  prefix       = \'{get_chemical_formula(cif_path)}\'\n')
#         f.write(f'  outdir       = \'{params["control"]["outdir"]}\'\n')
#         f.write(f'  pseudo_dir   = \'{params["control"]["pseudo_dir"]}\'\n')
#         f.write(f'  verbosity    = \'{params["control"]["verbosity"]}\'\n')
#         f.write(f'  restart_mode = \'{params["control"]["restart_mode"]}\'\n')
#         f.write('/\n\n')

#         f.write('&SYSTEM\n')
#         nat = len(atoms.get_chemical_symbols())
#         ntyp = len(set(atoms.get_chemical_symbols()))
#         f.write(f'  nat         = {nat}\n')
#         f.write(f'  ntyp        = {ntyp}\n')
#         f.write(f'  ibrav       = {params["system"]["ibrav"]}\n')
#         f.write(f'  ecutwfc     = {params["system"]["ecutwfc"]}\n')
#         f.write(f'  occupations = \'{params["system"]["occupations"]}\'\n')
#         f.write(f'  smearing    = \'{params["system"]["smearing"]}\'\n')
#         f.write(f'  degauss     = {params["system"]["degauss"]}\n')
#         f.write(f'  nbnd        = {params["system"]["nbnd"]}\n')
#         f.write(f'  nosym       = {params["system"]["nosym"]}\n')
#         f.write(f'  noinv       = {params["system"]["noinv"]}\n')
#         f.write('/\n\n')

#         f.write('&ELECTRONS\n')
#         f.write(f'  electron_maxstep = {params["electrons"]["electron_maxstep"]}\n')
#         f.write(f'  conv_thr         = {params["electrons"]["conv_thr"]}\n')
#         f.write(f'  mixing_mode      = \'{params["electrons"]["mixing_mode"]}\'\n')
#         f.write(f'  mixing_beta      = {params["electrons"]["mixing_beta"]}\n')
#         f.write(f'  mixing_ndim      = {params["electrons"]["mixing_ndim"]}\n')
#         f.write(f'  diagonalization  = \'{params["electrons"]["diagonalization"]}\'\n')
#         f.write(f'  diago_david_ndim = {params["electrons"]["diago_david_ndim"]}\n')
#         f.write(f'  diago_full_acc   = {params["electrons"]["diago_full_acc"]}\n')
#         f.write('/\n\n')

#         f.write('ATOMIC_SPECIES\n')
#         for atom in set(atoms.get_chemical_symbols()):
#             f.write(f'  {atom}  {params["atomic_species"][atom]}  {atom}.upf\n')
#         f.write(f'\n')
        
#         f.write("CELL_PARAMETERS angstrom\n")
#         cell = atoms.get_cell()
#         for v in cell:
#             f.write(f"  {v[0]}  {v[1]}  {v[2]}\n")
#         f.write('\n')

#         f.write('ATOMIC_POSITIONS crystal\n')
#         for symbol, frac in zip(atoms.get_chemical_symbols(), atoms.get_scaled_positions()):
#             f.write(f'  {symbol}  {frac[0]}  {frac[1]}  {frac[2]}\n')
#         f.write('\n')

#         # Generate k-points using Wannier90's kmesh.pl and write in QE format
#         kmeshpl_path = params["paths"]["kmeshpl"]
#         kmesh = params["win"]["mp_grid"]
#         result = subprocess.run(
#             [kmeshpl_path] + kmesh,
#             check=True,
#             capture_output=True,
#             text=True
#         )
#         kpoints_lines = result.stdout.strip().splitlines()
#         for line in kpoints_lines:
#             # Each line should be: kx ky kz weight
#             f.write(f'{line}\n')
#         f.write('\n')
