#!/bin/bash
# Create output/profile/selfcheck directories expected by our slurm scripts.
# Run this once after ./run.sh clean or on a fresh checkout.
# Safe to re-run (idempotent).

mkdir -p outputs/m1 outputs/m2 outputs/m3 outputs/
mkdir -p profiles/nsys profiles/ncu profiles/sanitize
mkdir -p selfcheck/m1 selfcheck/m2 selfcheck/m3

echo "Created: outputs/{m1,m2,m3}, profiles/{nsys,ncu,sanitize}, selfcheck/{m1,m2,m3}"
