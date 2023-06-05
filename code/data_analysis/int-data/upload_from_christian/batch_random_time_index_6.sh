#!/bin/sh
#BSUB -q hpc
#BSUB -J plots-for-reports-6
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -n 8
#BSUB -W 24
#BSUB -u s2114693@dtu.dk
#BSUB -B
#BSUB -N

module load mathematica/12.2.0
wolframscript -config WOLFRAMSCRIPT_KERNELPATH="/appl/mathematica/12.2.0/Executables/WolframKernel"
wolframscript --file evaluate_sums.wls 1 44 6
    