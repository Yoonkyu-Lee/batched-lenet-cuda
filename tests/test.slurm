#!/bin/bash
#SBATCH --job-name="LENET_TEST"
#SBATCH --output="tests/test_%j.out"
#SBATCH --error="tests/test_%j.err"
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

for bin in bin/correctness_baseline bin/correctness_fused bin/correctness_tensor_cores bin/correctness_register_tiled; do
    echo "--- $bin ---"
    srun "$bin"
    echo
done
