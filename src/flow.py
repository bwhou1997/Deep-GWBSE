from from_bgwpy import Structure, EpsilonTask, GWFlow
from config import config

class flow:
    def __init__(self):
        pass

    def write(self):
        pass
    


# flow = GWFlow(
#     dirname='GW-workflow',
#     structure = Structure.from_file('./input/mat-1/SiH4.cif'),
#     ecuteps = 30.0,
#     ibnd_min = 1,
#     ibnd_max = 8,
#     ngkpt = [8,8,8],
#     qshift = [.0,.0,.001],
#     nbnd = 400,
#     ecutwfc = 200.0,
#     prefix = 'SiH',
#     pseudo_dir = './input/mat-1/pseudo_qe',
#     pseudos = ['Si.upf','H.upf'],
# )

# flow.write()

if __name__ == "__main__":
    pass
