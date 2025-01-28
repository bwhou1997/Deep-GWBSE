from BGWpy import Structure, EpsilonTask, GWFlow

# task = EpsilonTask(
#     dirname='Epsilon',
#     structure = Structure.from_file('Si.cif'),
#     ngkpt = [2,2,2],
#     qshift = [.001,.0,.0],
#     ecuteps = 10.0,
#     wfn_fname = 'Wfn/WFN',
#     wfnq_fname= 'WFNq/WFNq',
#     nproc = 64,
# )
# task.write()

flow = GWFlow(
    dirname='GW-workflow',
    structure = Structure.from_file('./input/mat-1/SiH4.cif'),
    ecuteps = 30.0,
    ibnd_min = 1,
    ibnd_max = 8,
    ngkpt = [8,8,8],
    qshift = [.0,.0,.001],
    nbnd = 400,
    ecutwfc = 200.0,
    prefix = 'SiH',
    pseudo_dir = './input/mat-1/pseudo_qe',
    pseudos = ['H.upf', 'Si.upf'],
)

flow.write()