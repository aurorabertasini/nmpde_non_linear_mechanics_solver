#!/bin/bash

# Loop from 2 to 48 in steps of 2
for ntasks in $(seq 2 2 48); do
    
    # Create the submit_job.sh file
    cat > submit_job.sh <<EOF
#!/bin/bash
#SBATCH --account=pMI24_MatBa
#SBATCH --partition=g100_usr_prod
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${ntasks}
#SBATCH --mem=94000
#SBATCH --job-name=navier_stokes_job

source ~/.bashrc

set -e  # Exit on error

# Configuration ##################################################

EXE_DIR=\${HOME}/Navier_Stokes_Solver
EXE_NAME=/build/main_uncoupled

OUT_DIR=\${CINECA_SCRATCH}/Navier_Stokes_Output
OUT_FILE=\${OUT_DIR}/stdout.txt

COMMAND="./build/main_uncoupled"

##################################################################

cd \${EXE_DIR}

mkdir -p \${OUT_DIR}
rm -f \${OUT_FILE}
touch \${OUT_FILE}

echo "---------------------------------------------------" &>> \${OUT_FILE}
echo "Working directory: \$(pwd)"                           &>> \${OUT_FILE}
echo "Running simulation"                                  &>> \${OUT_FILE}
echo "Command: \${COMMAND}"                                 &>> \${OUT_FILE}
echo "Nodes: \${SLURM_NNODES}, Cores per node: \${SLURM_NTASKS_PER_NODE}" &>> \${OUT_FILE}
echo "---------------------------------------------------" &>> \${OUT_FILE}

srun --cpu-bind=cores \${COMMAND} &>> \${OUT_FILE}
EOF

    # Submit the job
    echo "Submitting job for ntasks-per-node=${ntasks}..."
    sbatch submit_job.sh

done
