#!/usr/bin/env python3
import subprocess

def main():
    # Here-doc style content with a placeholder {ntasks} for the tasks per node
    script_template = """#!/bin/bash
#SBATCH --account=pMI24_MatBa
#SBATCH --partition=g100_usr_prod
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={ntasks}
#SBATCH --mem=94000
#SBATCH --job-name=navier_stokes_job

source ~/.bashrc

set -e  # Exit on error

# Configuration ##################################################

EXE_DIR=${HOME}/Navier_Stokes_Solver
EXE_NAME=/build/main_uncoupled

OUT_DIR=${CINECA_SCRATCH}/Navier_Stokes_Output
OUT_FILE=${OUT_DIR}/stdout.txt

COMMAND="./build/main_uncoupled"

##################################################################

cd ${EXE_DIR}

mkdir -p ${OUT_DIR}
rm -f ${OUT_FILE}
touch ${OUT_FILE}

echo "---------------------------------------------------" &>> ${OUT_FILE}
echo "Working directory: $(pwd)"                           &>> ${OUT_FILE}
echo "Running simulation"                                  &>> ${OUT_FILE}
echo "Command: ${COMMAND}"                                 &>> ${OUT_FILE}
echo "Nodes: ${SLURM_NNODES}, Cores per node: ${SLURM_NTASKS_PER_NODE}" &>> ${OUT_FILE}
echo "---------------------------------------------------" &>> ${OUT_FILE}

srun --cpu-bind=cores ${COMMAND} &>> ${OUT_FILE}
"""

    # Loop from 2 to 48 in steps of 2
    for ntasks in range(2, 49, 2):
        # Create/overwrite submit_job.sh
        with open("submit_job.sh", "w") as f:
            f.write(script_template.format(ntasks=ntasks))

        # Print progress message
        print(f"Submitting job for --ntasks-per-node={ntasks}...")

        # Submit the job
        subprocess.run(["sbatch", "submit_job.sh"], check=True)

if __name__ == "__main__":
    main()
