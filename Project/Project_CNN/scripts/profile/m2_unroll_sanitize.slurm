#!/bin/bash
#SBATCH --job-name="M2_UNROLL_SAN"
#SBATCH --output="outputs/m2/m2_unroll_sanitize.out"
#SBATCH --error="outputs/m2/m2_unroll_sanitize.err"
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

srun compute-sanitizer ./m2_unroll 100 > profiles/sanitize/m2_unroll_sanitize_run.out
