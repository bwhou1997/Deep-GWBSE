# Deep-GWBSE

Deep-GWBSE is an end-to-end deep learning pipeline designed for DFT-GW-BSE. 

Author: Bowen Hou (bowen.hou@yale.edu)

Contributors: Xian Xu (xian.xu@yale.edu), Jinyuan Wu (jinyuan.wu@yale.edu)

## Outline
- [Deep-GWBSE](#deep-gwbse)
  - [Features](#features)
  - [Documentation](#documentation)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)


## Features
This package provides multiple deep learning models for DFT-GW-BSE calculations from crystal structures, including the following:
- Fully-automatic GW+BSE workflow
  - Parabands + Pseudobands
  - NNS
  - HPRO + DeepH
- VAE+MBFormer: transformer-based model for many-body GW-BSE  
  - model scheme:  
    <p align="center">
      <img src="src/from_model/fig/01-model.png" width="100%">
    </p>
  <!-- - GW scheme:  
    <p align="center">
      <img src="src/from_model/fig/02-GW.png" width="50%">
    </p>
  - BSE scheme:  
    <p align="center">
      <img src="src/from_model/fig/03-BSE.png" width="50%">
    </p> -->

## Documentation:

If you only want to use ``flow`` module to quickly setup GW-BSE calculation workflow, you might skip this part

For developer and advanced user, please carefully read this [**Documentation**](./src/note.md) for more details.


## Installation
First-principles Packages:
- [Quantum ESPRESSO](https://www.quantum-espresso.org/) version 6.8
- [BerkeleyGW](https://berkeleygw.org/documentation/tutorial/) version 3
- [SIESTA](https://docs.siesta-project.org/projects/siesta/en/stable/index.html)(Optional) version 5+ `conda install -c conda-forge siesta=5.2.1`
- [Pseudo-dojo](https://www.pseudo-dojo.org/)(Optional)

Deep-GWBSE Installation:

You first can create your own conda environment:
```
conda create -n mbformer python==3.9.5
conda activate mbformer
```

```bash
git clone https://github.com/bwhou1997/Deep-GWBSE.git
cd Deep-GWBSE
pip install -r requirements.txt
```

Export your environment variables:

```bash
export PATH=/path/to/Deep-GWBSE/src:$PATH
export PYTHONPATH="${PYTHONPATH}:/path/to/Deep-GWBSE/src"
```

## Quick Start
### 0. setup your own QE and BGW path
```
cd src/
cp config/single_mat_config.json ./
cp config/fpconfig.json ./
```
modify `"QE_path"`, `"BGW_path"`, `"pseudo_dir_source"` based on your own software path. Note: for `"pseudo_dir_source"`, we have already had a built-in pseudo potential package from oncvpsp, and you can simply link it to `'./from_oncvpsp'` 

### 1. GW-BSE workflow part (stay at `src` directory)

For single material (here is hBN), run it on an **interactive node**:

```
python flow.py -c single_mat_config.json
cd flow-hBN
sbatch run.sh 
```
(It might take a while, you can do something else...)

For multiple materials, run them as a batch on an **interactive node**

```
python flows.py -c fpconfig.json
cd flows
sbatch run.sh
```
(It will **for sure** take a while, you can do something else...)

### 2. Preprocessing the raw data from GWBSE
Note1: If you don't want to do ML, you can stop here and enjoy your life.

Note2: for advanced user and developer, again, **please carefully read this [**Documentation**](./src/note.md) for more details.** Most of python files have a `test` part following `if __name__ == "__main__"`. **Please run it everytime you modify the code to prevent introducing bugs**.

All the machine learning code is located at `from_model`, so:

`cd from_model`

Here, I have already prepared some dummy raw data from GW-BSE, which is saved in `Deep-GWBSE/examples/flows`, and you can take a look. Then, read the main function of `data.py` for more details and run it:

`python data.py`

it will create three hdf5 files `dataset_WFN.h5` (used for training a VAE), `dataset_GW.h5` (used for training a GW-MBFormer) and `dataset_BSE.h5` (used for training a BSE-MBFormer). 


### 3. Training your first MBFormer model!

Train an E2-VAE to embed KS wavefunction:

```
python e2vaetrainer.py
```

The VAE model then will be saved in `./vae_e2_wfn.save` and you can train your BSE-MBFormer!

```
python bsetrainer.py
```



## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


## Acknowledgements
We would like to acknowledge the following open-source projects that have made this work possible:
[Quantum ESPRESSO](https://www.quantum-espresso.org/), [BerkeleyGW](https://berkeleygw.org/), [SIESTA](https://docs.siesta-project.org/projects/siesta/en/stable/index.html), [DeepH-E3](https://github.com/Xiaoxun-Gong/DeepH-E3), [HPRO](https://github.com/Xiaoxun-Gong/HPRO), bgwpy

