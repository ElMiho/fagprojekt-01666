#!/bin/sh 
#BSUB -q hpc
#BSUB -J test_batch
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 24:00
#BSUB -u s214753@dtu.dk
#BSUB -B 
#BSUB -N 
#BSUB -o Output_%J.out 
#BSUB -e Output_%J.err 

module load mathematica/12.2.0
wolframscript -config WOLFRAMSCRIPT_KERNELPATH="/appl/mathematica/12.2.0/Executables/WolframKernel"
wolframscript -file generate_test_data.wls 1 5
wolframscript -file generate_test_data.wls 5 10
