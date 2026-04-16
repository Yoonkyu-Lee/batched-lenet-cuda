#!/bin/bash
#SBATCH --job-name="LENET_BENCH_ALL"
#SBATCH --output="bench/run_all_%j.out"
#SBATCH --error="bench/run_all_%j.err"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint="projects"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 00:20:00

module reset
srun bash bench/run_all.sh
