#!/bin/bash
#SBATCH --job-name="M3_QUICK"
#SBATCH --output="outputs/m3/m3_quick_%j.out"
#SBATCH --error="outputs/m3/m3_quick_%j.err"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint="projects"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 00:10:00

module reset

TRIAL="${TRIAL:-unnamed}"
echo "=== Trial: ${TRIAL} | Job: ${SLURM_JOBID} ==="
echo "=== M3 quick test @ B=10000 ==="
srun ./m3 10000
echo "=== B=100 accuracy check ==="
srun ./m3 100
