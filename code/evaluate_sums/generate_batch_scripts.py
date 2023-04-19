import os

def file_content(a, b):
    content = f"""#!/bin/sh
#BSUB -q hpc
#BSUB -J random_partition_{a}-{b}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=10GB]"
#BSUB -n 8
#BSUB -W 72:00
#BSUB -u s214753@dtu.dk
#BSUB -B
#BSUB -N

module load mathematica/12.2.0
wolframscript -config WOLFRAMSCRIPT_KERNELPATH="/appl/mathematica/12.2.0/Executables/WolframKernel"
wolframscript --file evaluate_sums.wls {a} {b}
    """
    return content

date = "8_4_2023"

for i in range(7, 44+1):
    # Generate batch script
    f = open(f"batch_random_{i}_{date}.sh", "w")
    f.write(
        file_content(i, i)
    )
    f.close()

    # Runt it!
    os.system(f"bsub < batch_random_{i}_{date}.sh")