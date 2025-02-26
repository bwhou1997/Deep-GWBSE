# Deep-GWBSE

Deep-GWBSE is a deep learning model designed for DFT-GW-BSE calculations. 

Author: Bowen Hou (bowen.hou@yale.edu)

## TODO List:
- Checkpoint systems 
- eqp.dat -> HPRO (working)
- CSI for BSE

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Features
This package provides multiple deep learning models for DFT-GW-BSE calculations from crystal structures, including the following:
- Fully-automatic GW+BSE workflow ([high-throughput workflow](./src/note.md))
- Equivariant graph neural networks for DFT Hamiltonian

- VAE+MBFormer: attention-based many-body transformer for GW-BSE
  - model scheme:
![model](src/from_model/fig/01-model.png)
  - GW scheme:
![GW](src/from_model/fig/02-GW.png)
  - BSE scheme:
![BSE](src/from_model/fig/03-BSE.png)

## Installation
Pre-requisites First-principles Packages:
- [Quantum ESPRESSO](https://www.quantum-espresso.org/) version 6.8
- [BerkeleyGW](https://berkeleygw.org/documentation/tutorial/) version 3+
- [SIESTA](https://docs.siesta-project.org/projects/siesta/en/stable/index.html) version 5+ `conda install -c conda-forge siesta=5.2.1`
- [Pseudo-dojo](https://www.pseudo-dojo.org/)

Pre-requisites python Packages:
- pymatgen `conda install conda-forge::pymatgen`

Useful python packages:
- bgwpy
- HPRO (Note: for testing, export it to PYTHONPATH, integrate it later)
- DeepH-E3

To install Deep-GWBSE, clone the repository and install the required dependencies:

```bash
git clone https://github.com/bwhou1997/Deep-GWBSE.git
cd Deep-GWBSE
pip install -r requirements.txt
```

## Usage
see the [examples](examples) folder for more details.


## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


## Acknowledgements
We would like to acknowledge the following open-source projects that have made this work possible:
[Quantum ESPRESSO](https://www.quantum-espresso.org/), [BerkeleyGW](https://berkeleygw.org/), [SIESTA](https://docs.siesta-project.org/projects/siesta/en/stable/index.html), [DeepH-E3](https://github.com/Xiaoxun-Gong/DeepH-E3), [HPRO](https://github.com/Xiaoxun-Gong/HPRO), bgwpy


