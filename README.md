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

We can project it to atomic orbital basis $| i \alpha \rangle$ as:

$$
H^{QP}_{i\alpha, j\beta} = \sum_{nk} \sum_{i\alpha, j\beta}  \epsilon^{QP} \langle i\alpha | nk \rangle  \langle nk | j\beta \rangle 
$$

In term of tight-binding model, $H^{QP}_{i\alpha, j\beta}$ can be understood as a hopping term between two nodes. To learn this hopping term, we can first build a graph structure on crystal system {$v_i$, $e_{ij}$}, where node feature $v_i=0\oplus 0\oplus 0...$ embeds element information, $e_{ij}=0\oplus 1 \oplus 2...$ embeds geometry information between two codes. Then, e3nn convolution layer can conduct message passing for this graph and map it to hopping term $H^{QP}=e^{L}_{ij}$.

## Installation
To install Deep-GWBSE, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/Deep-GWBSE.git
cd Deep-GWBSE
pip install -r requirements.txt
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