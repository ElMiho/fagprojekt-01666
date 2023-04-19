#!/bin/sh
#BSUB -q hpc
#BSUB -J status_script_random
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 72:00
#BSUB -u s214753@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -n 8
#BSUB -o Output_%J.out
#BSUB -e Output_%J.err

while true
do
    ./status_script.sh random-answers-partition-8-4-2023
    sleep 900
done