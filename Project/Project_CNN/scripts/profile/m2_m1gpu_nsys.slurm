#!/bin/bash
#SBATCH --job-name="M2_M1GPU_NSYS"
#SBATCH --output="outputs/m2/m2_m1gpu_nsys.out"
#SBATCH --error="outputs/m2/m2_m1gpu_nsys.err"
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

srun nsys profile --stats=true -o profiles/nsys/m1_gpu_profile ./m1_gpu 10000 > profiles/nsys/m1_gpu_nsys_profile.out
