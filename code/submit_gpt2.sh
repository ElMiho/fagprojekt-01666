#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpuv100
### -- set the job Name -- 
#BSUB -J SumAI
### -- ask for number of cores (default: 1) -- 
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 32GB of memory per core/slot -- 
#BSUB -R "rusage[mem=32GB]"
### -- specify that we want 32 GB GPU
#BSUB -R "select[gpu32gb]"
### -- specify that we want the job to get killed if it exceeds 32 GB per core/slot -- 
#BSUB -M 32GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00 
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u s214734@student.dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o Output_%J.out 
#BSUB -e Output_%J.err 

nvidia-smi
# Load the cuda module
module load cuda/11.8

#/appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery
python3 "GPT2 - Training Time.py"  > output.out