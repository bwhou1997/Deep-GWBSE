# Deep-GWBSE

Deep-GWBSE is a deep learning model designed for DFT-GW-BSE calculations. 

Author: Bowen Hou (bowen.hou@yale.edu)

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Introduction
This package provides multiple deep learning models for DFT-GW-BSE calculations from crystal structures, including the following:
- equivariant graph neural networks
- equivariant attention networks

## Installation
Pre-requisites First-principles Packages:
- [Quantum ESPRESSO](https://www.quantum-espresso.org/) version 7+
- [BerkeleyGW](https://berkeleygw.org/documentation/tutorial/) version 3+
- [SIESTA](https://docs.siesta-project.org/projects/siesta/en/stable/index.html) version 5+ (see requirements.txt)
- [Pseudo-dojo](https://www.pseudo-dojo.org/)

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
[Quantum ESPRESSO](https://www.quantum-espresso.org/), [BerkeleyGW](https://berkeleygw.org/), [SIESTA](https://docs.siesta-project.org/projects/siesta/en/stable/index.html), [DeepH-E3](https://github.com/Xiaoxun-Gong/DeepH-E3), [HPRO](https://github.com/Xiaoxun-Gong/HPRO)


