#!/bin/bash

# Task counts and meshes to test
task_counts=(1 10 20 30)
meshes=("mesh/Cylinder2D_fine_fine.msh" "mesh/Cylinder2D_fine.msh" "mesh/Cylinder2D_normal.msh" "mesh/Cylinder2D_coarse.msh" "mesh/Cylinder2D_coarse_coarse.msh")  # Replace with actual mesh paths

for ntasks in "${task_counts[@]}"; do
    for mesh_path in "${meshes[@]}"; do
        # Extract just the mesh filename for naming
        mesh_name=$(basename "${mesh_path}")
        
        # Create unique submit file for each combination
        cat > submit_job_${ntasks}_${mesh_name}.sh <<EOF
#!/bin/bash
#SBATCH --account=pMI24_MatBa
#SBATCH --partition=g100_usr_prod
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${ntasks}
#SBATCH --mem=94000
#SBATCH --job-name=ns_${ntasks}_${mesh_name}

source ~/.bashrc

set -e  # Exit on error

# Configuration ##################################################
EXE_DIR=\${HOME}/Navier_Stokes_Solver
EXE_NAME=/build/main_uncoupled
OUT_DIR=\${CINECA_SCRATCH}/Navier_Stokes_Output
OUT_FILE=\${OUT_DIR}/stdout_${ntasks}_${mesh_name}.txt
COMMAND="./build/main_uncoupled ${mesh_path}"
##################################################################

cd \${EXE_DIR}
mkdir -p \${OUT_DIR}

echo "---------------------------------------------------" > \${OUT_FILE}
echo "Task count: ${ntasks}, Mesh: ${mesh_name}" >> \${OUT_FILE}
echo "Nodes: \${SLURM_NNODES}, Cores/node: \${SLURM_NTASKS_PER_NODE}" >> \${OUT_FILE}
echo "Start time: \$(date)" >> \${OUT_FILE}
echo "---------------------------------------------------" >> \${OUT_FILE}

srun --cpu-bind=cores \${COMMAND} &>> \${OUT_FILE}

echo "---------------------------------------------------" >> \${OUT_FILE}
echo "End time: \$(date)" >> \${OUT_FILE}
echo "---------------------------------------------------" >> \${OUT_FILE}
EOF

        # Submit the job
        echo "Submitting ${ntasks} tasks with mesh ${mesh_name}..."
        sbatch submit_job_${ntasks}_${mesh_name}.sh
    done
done