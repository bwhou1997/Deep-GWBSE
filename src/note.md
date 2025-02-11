## Folder Structure
Here is the basic workflow
```
external src──(collect)─>fp-input ──(QE/SIESTA/GW + HPRO)─>─┌── ml-train-set──(ML)─> model
                                                            └── ml-test-set
```
                                              
### 1. **fp-input** folder
The fp-input folder serves as the starting point for the workflow. It contains the configuration, crystal structures, and pseudopotential files necessary for first-principles calculations. The structure is organized as follows:
```bash
fp-input
├── fpconfig.json
├── mat-1 # (extensible)
|   └── stru.cif
├── mat-2
|   └── stru.cif
└── ...

pseudo_src/ # (built-in)
├── ele1.upf
├── ele2.upf
├── ...
├── ele1.psf/psml
├── ele2.psf/psml
└── ...
```

### 2. **ml-train/test** folder
```bash
ml-train/test
├── mat-1
|   ├── config.json
|   ├── stru.cif
|   ├── pp/ # (built-in)
|   |   ├── ele1.upf
|   |   ├── ele2.upf
|   |   ├── ...
|   |   ├── ele1.psf/psml
|   |   ├── ele2.psf/psml
|   |   └── ...
|   ├──01-density
|   |   ├── VSC # (DFT Ham.)
|   |   └── ...
|   ├──02-wfn
|   ├──03-wfnq
|   ├──05-band
|   ├──06-wfnq-nns
|   ├──07-aobasis
|   |   ├── ele1.ion # (LCAO basis)
|   |   ├── ele2.ion
|   |   └── ...
|   ├──11-epsilon
|   ├──11-epsilon-nns
|   ├──13-sigma
|   |   ├── eqp1.dat # (G0W0 corr.)
|   |   └── ...
|   ├──14-inteqp
|   ├──16-reconstruction
|   |   ├──aohamiltonian
|   |   |   ├── element.dat
|   |   |   ├── hamiltonians.h5
|   |   |   ├── info.json
|   |   |   ├── lat.dat
|   |   |   ├── orbital_types.dat
|   └── └── └── site_positions.dat
|
├── mat-2
|   └──  ...
└── ...
```