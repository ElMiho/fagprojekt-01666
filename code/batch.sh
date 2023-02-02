#!/bin/sh 
#BSUB -q hpc
#BSUB -J test_batch
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 24:00
#BSUB -u s214753@dtu.dk
#BSUB -B 
#BSUB -N 

module load mathematica/12.2.0
WolframKernel -script generate_polynomials.wls

#BSUB -o Output_%J.out 
#BSUB -e Output_%J.err 