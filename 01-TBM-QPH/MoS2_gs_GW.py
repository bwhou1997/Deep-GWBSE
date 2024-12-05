import time
from gpaw.response.g0w0 import G0W0
from ase.build import mx2
from gpaw import GPAW, PW, FermiDirac
from ase.parallel import parprint

structure = mx2(formula='MoS2', kind='2H', a=3.184, thickness=3.127,
                size=(1, 1, 1), vacuum=3.5)

Ecut = 300

print('start GS')
start_time_gs = time.time()

calc = GPAW(mode=PW(Ecut),
            # parallel={'domain': 1},
            xc='PBE',
            basis='dzp',
            kpts={'size': (6, 6, 1), 'gamma': True},
            occupations=FermiDirac(0.01),
            txt='MoS2_out_gs.txt',
            # parallel={'kpt':16,'band':16}
            )

structure.calc = calc
structure.get_potential_energy()
calc.write('MoS2_gs.gpw', 'all')

calc.diagonalize_full_hamiltonian()
calc.write('MoS2_fulldiag.gpw', 'all')



e, P = calc.get_orbital_ldos(a=0, angular='d')



# print('start GW')
# start_time_gw = time.time()

# for ecut in [80]:
#     gw = G0W0(calc='MoS2_fulldiag.gpw',
#               bands=(8, 18),
#               ecut=ecut,
#               truncation='2D',
#               nblocksmax=True,
#               q0_correction=True,
#               filename=f'MoS2_g0w0_{ecut}')

#     result = gw.calculate()

# end_time_gw = time.time()
# print(f'GW step completed in {end_time_gw - start_time_gw:.2f} seconds')
