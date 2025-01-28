## Folder Structure
Here is the basic workflow
```
fp-input ──(QE/SIESTA/GW + HPRO)─>─┌── ml-train-set──(ML)──> model
                                   └── ml-test-set
```
                                              

### 1. **fp-input** folder
The fp-input folder serves as the starting point for the workflow. It contains the configuration, crystal structures, and pseudopotential files necessary for first-principles calculations. The structure is organized as follows:
```bash
fp-input
├── fpconfig.json
├── mat-1
|   ├── mat-1.cif
|   ├── pseudo_qe
|   |   ├── ele1.upf
|   |   ├── ele2.upf
|   |   └── ...
|   ├── pseudo_siesta
|   |   ├── ele1.psf/psml
|   |   ├── ele2.psf/psml
|   |   └── ...
├── mat-2
|   └──  ...
└── ...
```

### 2. **ml-train/test** folder
```bash
ml-train/test
├── mat-1
|   ├── pseudo_qe
|   |   ├── ele1.upf
|   |   ├── ele2.upf
|   |   └── ...
|   ├── pseudo_siesta
|   |   ├── ele1.psf/psml
|   |   ├── ele2.psf/psml
|   |   └── ...
|   ├──01-density
|   |   ├── VSC (DFT Ham.)
|   |   └── ...
|   ├──02-wfn
|   ├──03-wfnq
|   ├──04-band
|   ├──05-aobasis
|   |   ├── ele1.ion (LCAO basis)
|   |   ├── ele2.ion
|   |   └── ...
|   ├──11-epsilon
|   ├──12-sigma
|   |   ├── eqp.dat (G0W0 corr.)
|   |   └── ...
|   ├──15-inteqp
|   ├──16-reconstruction
|   |   ├── element.dat
|   |   ├── hamiltonians.h5
|   |   ├── info.json
|   |   ├── lat.dat
|   |   ├── orbital_types.dat
|   |   └── site_positions.dat
├── mat-2
|   └──  ...
└── ...
```