#!/bin/bash
#SBATCH --job-name="M3_NCU"
#SBATCH --output="outputs/m3/m3_ncu_%j.out"
#SBATCH --error="outputs/m3/m3_ncu_%j.err"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint="projects,perf,nvperf"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 00:30:00

module reset
mkdir -p profiles/ncu

echo "=== ncu --set full on ./m3 100 ==="
srun ncu --set full -f -o profiles/ncu/m3_stacked_analysis ./m3 100 > profiles/ncu/m3_stacked_ncu.out 2>&1

echo "=== Summary (bank conflicts + occupancy) ==="
srun ncu --metrics \
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__inst_executed.sum,\
smsp__sass_average_data_bytes_per_wavefront_mem_shared_op_ld.pct,\
smsp__sass_average_data_bytes_per_wavefront_mem_shared_op_st.pct \
  ./m3 100
