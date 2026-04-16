#!/bin/bash
#SBATCH --job-name="M3_COMP"
#SBATCH --output="outputs/m3/m3_competition_%j.out"
#SBATCH --error="outputs/m3/m3_competition_%j.err"
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
echo "=== ./m3 --competition ==="
srun ./m3 --competition
