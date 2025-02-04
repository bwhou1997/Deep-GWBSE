from ase.build import bulk, molecule
from gpaw import GPAW, FermiDirac
from gpaw import PW
from gpaw.response.g0w0 import G0W0

# a = 3.567
# atoms = bulk('C', 'diamond', a=a)

# calc = GPAW(mode=PW(300),  # energy cutoff for plane wave basis (in eV)
#             kpts={'size': (3, 3, 3), 'gamma': True},
#             xc='LDA',
#             occupations=FermiDirac(0.001),
#             parallel={'domain': 1},
#             txt='C_groundstate.txt')

# atoms.calc = calc
# atoms.get_potential_energy()

# calc.diagonalize_full_hamiltonian()  # determine all bands
# calc.write('C_groundstate.gpw', 'all')  # write out wavefunctions

# from gpaw.response.g0w0 import G0W0

# gw = G0W0(calc='C_groundstate.gpw',
#           nbands=30,  # number of bands for calculation of self-energy
#           bands=(3, 5),          # VB and CB
#           ecut=20.0,             # plane-wave cutoff for self-energy
#           integrate_gamma='WS',  # Use supercell Wigner-Seitz truncation for W.
#           filename='C-g0w0')
# result = gw.calculate()


####################################
"""
i) convergence
ii) parallelization
"""

mol = molecule("H2O", vacuum=5)
calc = GPAW(mode=PW(ecut=100, force_complex_dtype=True), 
                    basis='dzp',
                    nbands="100%",
                    parallel={'band': 2},)
mol.calc = calc
mol.get_potential_energy()

calc.diagonalize_full_hamiltonian()
calc.write('%s_fulldiag.gpw'%"H2O", 'all')

gw = G0W0(calc='%s_fulldiag.gpw'%"H2O",
        ecut=20,
        filename='H2O_g0w0',
    #   truncation='0D',
    #   ppa=True,
        bands=(1,8), # slice(1, 8)
        nbands=300)
gw_result = gw.calculate()

print(gw_result['qp'])