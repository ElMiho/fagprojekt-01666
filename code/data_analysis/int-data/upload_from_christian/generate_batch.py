import os

def file_content(seconds):
    content = f"""#!/bin/sh
#BSUB -q hpc
#BSUB -J plots-for-reports-{seconds}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -n 8
#BSUB -W 24
#BSUB -u s2114693@dtu.dk
#BSUB -B
#BSUB -N

module load mathematica/12.2.0
wolframscript -config WOLFRAMSCRIPT_KERNELPATH="/appl/mathematica/12.2.0/Executables/WolframKernel"
wolframscript --file evaluate_sums.wls 1 44 {seconds}
    """
    return content

for i in range(4, 10 + 1):
    # Generate batch script
    f = open(f"batch_random_time_index_{i}.sh", "w")
    f.write(
        file_content(i)
    )
    f.close()

    # Runt it!
    os.system(f"bsub < batch_random_time_index_{i}.sh")
