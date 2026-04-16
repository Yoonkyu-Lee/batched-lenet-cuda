#!/bin/bash
#SBATCH --job-name="M2_CPU_GPROF"
#SBATCH --output="outputs/m2/m2_cpu_gprof.out"
#SBATCH --error="outputs/m2/m2_cpu_gprof.err"
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

srun ./m1_cpu 1000 > outputs/m2/m2_cpu_profile_run.out
gprof -Q ./m1_cpu gmon.out > outputs/m2/m2_cpu_gprof_report.txt
