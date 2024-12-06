# Deep-GWBSE

Deep-GWBSE is a deep learning model designed for GW+BSE calculations. This model aims to provide accurate and efficient predictions for electronic structure calculations.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction
GW+BSE (Green's function and Bethe-Salpeter Equation) are advanced methods used in computational materials science to study the electronic properties of materials. Deep-GWBSE leverages deep learning techniques to enhance the accuracy and efficiency of these calculations.

### Idea

In $G_0W_0$, we assume $[H^{DFT}, \Sigma^{GW}]=0$, so the quasi-particle (QP) Hamiltonian reads:

$$
H^{QP} = \sum_{nk} | nk \rangle  \left( \epsilon^{DFT} + \epsilon^{G_0W_0} \right)  \langle nk |
$$

We can project the QP Hamiltonian to atomic orbital basis $| l \alpha \rangle$ as:

$$
H_{la, jb} = \sum_{nk}  \epsilon^{QP} \langle la | nk \rangle  \langle nk | jb \rangle 
$$

$$
= \sum_{nk}  \epsilon^{QP} \langle a | nk \rangle  \langle nk | b \rangle e^{ikR(l-j)}
$$

In term of tight-binding model, $H_{la, jb}$ can be understood as a hopping term between two nodes. To learn this hopping term, we can first build a graph structure on crystal system { $v_l$, $e_{lj}$ }, where node feature $v_i=0\oplus 0\oplus 0...$ embeds element information, $e_{lj}=0\oplus 1 \oplus 2...$ embeds geometry information between two nodes. Then, e3nn convolution layer can conduct message passing for this graph and map it to hopping term $H_{la, jb}=e^{L}_{lj}$, where L denotes the last layer

### Useful Reference:
- e3nn: https://arxiv.org/abs/1802.08219
- DeepH: https://www.nature.com/articles/s43588-022-00265-6
- DeepH-e3nn: https://www.nature.com/articles/s41467-023-38468-8
- HamGNN: https://iopscience.iop.org/article/10.1088/0256-307X/41/7/077103/meta
- el-ph GNN: https://www.nature.com/articles/s43588-024-00668-7
- Planewave2Orbital: https://www.nature.com/articles/s43588-024-00701-9
- AO basis: https://journals.aps.org/prb/pdf/10.1103/PhysRevB.80.195112, https://gpaw.readthedocs.io/documentation/lcao/lcao.html#lcao
- LCAO Hamiltonian: https://gpaw.readthedocs.io/tutorialsexercises/localorbitals/localorbitals.html
### Useful GPAW function

Coefficient $\langle nk | l \alpha \rangle$

dos.py 
```
(line 113): def raw_orbital_LDOS(paw, a, spin, angular='spdf', nbands=None)
```

calculator.py
```
(line 1642): get_orbital_ldos(self, a,
                         spin=0, angular='spdf', npts=201, width=None,
                         nbands=None, spinorbit=False)
```


## Installation
To install Deep-GWBSE, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/Deep-GWBSE.git
todo
```

## Usage
Todo

Example command:
```bash
Todo
```

## Contributing
We welcome contributions to Deep-GWBSE. Please fork the repository and submit pull requests for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.