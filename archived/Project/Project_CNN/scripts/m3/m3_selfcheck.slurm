#!/bin/bash
#SBATCH --job-name="M3_SELFCHK"
#SBATCH --output="outputs/m3/m3_selfcheck.out"
#SBATCH --error="outputs/m3/m3_selfcheck.err"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint="projects"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 01:00:00

module reset

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"

srun bash scripts/m3/m3_selfcheck.sh --with-edge
