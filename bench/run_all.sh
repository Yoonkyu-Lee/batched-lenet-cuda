#!/bin/bash
# Run every variant at multiple batch sizes, capture Op Times, print a summary
# table and write a local CSV. Run from the repo root after `make`.
#
# On Delta: submit via bench/run_all.slurm instead (login nodes have no GPU).

set -eu
set -o pipefail

VARIANTS=${VARIANTS:-"baseline fused tensor_cores register_tiled"}
BATCHES=${BATCHES:-"100 1000 10000"}
WARMUP=${WARMUP:-5}
MEASURED=${MEASURED:-10}

OUT_DIR=${OUT_DIR:-bench}
CSV="${OUT_DIR}/results_local.csv"
RAW_DIR="${OUT_DIR}/raw"
mkdir -p "$RAW_DIR"

if ! ls bin/baseline bin/fused bin/tensor_cores bin/register_tiled >/dev/null 2>&1; then
    echo "Error: binaries missing. Run 'make' first." >&2
    exit 1
fi

# CSV header
echo "variant,batch,conv1_ms,conv2_ms,total_ms" > "$CSV"

run_one() {
    local variant="$1"
    local batch="$2"
    local raw="${RAW_DIR}/${variant}_B${batch}.out"

    ./bin/"$variant" "$batch" "$WARMUP" "$MEASURED" > "$raw"

    # Extract the two "Op Time:" lines (one per layer), median numbers only.
    local conv1 conv2 total
    conv1=$(awk '/Op Time:/ {print $3; exit}' "$raw")
    conv2=$(awk '/Op Time:/ {n++; if (n==2) {print $3; exit}}' "$raw")
    total=$(awk '/Total Op Time:/ {print $4}' "$raw")

    echo "$variant,$batch,$conv1,$conv2,$total" >> "$CSV"
}

for variant in $VARIANTS; do
    for batch in $BATCHES; do
        echo "Running $variant @ B=$batch ..."
        run_one "$variant" "$batch"
    done
done

echo
echo "=== Summary ==="
# Pretty table grouped by batch size
for batch in $BATCHES; do
    echo
    echo "Batch = $batch"
    printf "  %-45s  %10s  %10s  %10s\n" "variant" "Conv1 (ms)" "Conv2 (ms)" "Total (ms)"
    printf "  %-45s  %10s  %10s  %10s\n" "---------------------------------------------" "---------" "---------" "---------"
    awk -F',' -v b="$batch" '
        NR>1 && $2==b {
            printf "  %-45s  %10s  %10s  %10s\n", $1, $3, $4, $5
        }
    ' "$CSV"
done

echo
echo "Raw outputs: $RAW_DIR/"
echo "CSV: $CSV"
