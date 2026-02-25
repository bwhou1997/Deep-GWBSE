#!/bin/bash

#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=64
#SBATCH -o myjob.o
#SBATCH -e myjob.e
#SBATCH -C cpu

export OMP_NUM_THREADS=2
module load espresso/7.3.1-libxc-6.2.2-cpu
module load python
conda activate nersc-python

cd /pscratch/sd/t/taviandj/Deep-GWBSE/src/from_wanniers/data_active

cd 0001-mat-14
echo "0001-mat-14"
bash ./run.sh
cd ..

cd 0002-mat-43
echo "0002-mat-43"
bash ./run.sh
cd ..

cd 0004-mat-50
echo "0004-mat-50"
bash ./run.sh
cd ..

cd 0005-mat-92
echo "0005-mat-92"
bash ./run.sh
cd ..

cd 0006-mat-119
echo "0006-mat-119"
bash ./run.sh
cd ..

cd 0008-mat-137
echo "0008-mat-137"
bash ./run.sh
cd ..

cd 0009-mat-178
echo "0009-mat-178"
bash ./run.sh
cd ..

cd 0011-mat-292
echo "0011-mat-292"
bash ./run.sh
cd ..

cd ..
