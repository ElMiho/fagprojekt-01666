#!/bin/sh 
#BSUB -q hpc
#BSUB -J sampling
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 1:00
#BSUB -u s214753@dtu.dk
#BSUB -B 
#BSUB -N 
#BSUB -o Output_%J.out 
#BSUB -e Output_%J.err

module load matplotlib/3.7.0-numpy-1.24.2-python-3.11.2
python3 random_sampling.py