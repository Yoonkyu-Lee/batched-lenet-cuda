#!/bin/bash
#SBATCH --job-name="M1_SELFCHK"
#SBATCH --output="outputs/m1/m1_selfcheck.out"
#SBATCH --error="outputs/m1/m1_selfcheck.err"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint="projects"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 00:30:00

module reset

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"

# Default behavior is GPU checks only. Add --cpu if you want the slower CPU path too.
srun bash scripts/m1/m1_selfcheck.sh
