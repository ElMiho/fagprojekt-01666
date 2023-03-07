#!/bin/sh 
#BSUB -q hpc
#BSUB -J test_batch
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1GB]"
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
wolframscript -file generate_test_data.wls 10 15
wolframscript -file generate_test_data.wls 15 20
wolframscript -file generate_test_data.wls 20 25
wolframscript -file generate_test_data.wls 25 30
wolframscript -file generate_test_data.wls 30 35
wolframscript -file generate_test_data.wls 35 40
wolframscript -file generate_test_data.wls 40 45