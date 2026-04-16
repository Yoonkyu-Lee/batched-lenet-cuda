#!/bin/bash
#SBATCH --job-name="LENET_BENCH"
#SBATCH --output="bench/bench_%j.out"
#SBATCH --error="bench/bench_%j.err"
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

echo "=== baseline ==="
srun bin/baseline 1000 3 5
echo
echo "=== fused ==="
srun bin/fused 1000 3 5
echo
echo "=== tensor_cores ==="
srun bin/tensor_cores 1000 3 5
echo
echo "=== register_tiled ==="
srun bin/register_tiled 1000 3 5
