#!/bin/bash

set -u
set -o pipefail

# Milestone 3 self-check runner.
# Checks accuracy and Op Time for the m3 binary across a batch sweep,
# including edge / prime batches that stress tiling boundaries.
# Enforces the M3 full-credit cap (<=60 ms total @ B=10000).

M3_BIN="./m3"

RESULT_ROOT="${RESULT_ROOT:-selfcheck/m3}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="${RESULT_ROOT}/${TIMESTAMP}"

# Standard public batches + edge/prime cases that caught prior bugs.
GPU_BATCHES=(100 1000 4095 4096 8767 10000)
EDGE_BATCHES=(1 127 4097 7919 9999)

ACCURACY_TOL="${ACCURACY_TOL:-0.002}"
SANITY_MIN_ACC="${SANITY_MIN_ACC:-0.80}"
M3_MAX_OP_MS="${M3_MAX_OP_MS:-60}"
M3_COMPETITION_PER_LAYER_MS="${M3_COMPETITION_PER_LAYER_MS:-40}"

declare -A EXPECTED_ACC=(
    [100]="0.86"
    [1000]="0.886"
    [10000]="0.8714"
)

usage() {
    cat <<'EOF'
Usage: bash m3_selfcheck.sh [options]

Options:
  --with-edge        Include edge/prime batches (1,127,4097,7919,9999)
  --with-sanitize    Run compute-sanitizer on B=100 (much slower)
  --with-competition Also check --competition mode at B=10000
  --result-dir DIR   Write outputs under DIR instead of selfcheck/m3
  --help             Show this help

Environment overrides:
  ACCURACY_TOL                 Public-batch absolute tolerance (default: 0.002)
  SANITY_MIN_ACC               Min accuracy for edge batches  (default: 0.80)
  M3_MAX_OP_MS                 Total Op Time cap @ B=10000    (default: 60)
  M3_COMPETITION_PER_LAYER_MS  Per-layer Op Time cap          (default: 40)
EOF
}

RUN_EDGE=0
RUN_SANITIZE=0
RUN_COMPETITION=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-edge) RUN_EDGE=1 ;;
        --with-sanitize) RUN_SANITIZE=1 ;;
        --with-competition) RUN_COMPETITION=1 ;;
        --result-dir) shift; RESULT_ROOT="$1"; RESULT_DIR="${RESULT_ROOT}/${TIMESTAMP}" ;;
        --help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
    shift
done

mkdir -p "$RESULT_DIR"
SUMMARY_FILE="${RESULT_DIR}/summary.txt"

overall_status=0

note() { echo "$1" | tee -a "$SUMMARY_FILE"; }

have_binary() {
    if [[ ! -x "$1" ]]; then
        note "MISSING: $1 is not built or not executable."
        overall_status=1
        return 1
    fi
    return 0
}

abs_diff_leq() {
    awk -v a="$1" -v e="$2" -v t="$3" \
        'BEGIN { d = a - e; if (d < 0) d = -d; exit !(d <= t); }'
}

float_ge() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a >= b) }'; }
float_le() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a <= b) }'; }

parse_accuracy()    { awk '/Test Accuracy:/ {print $3}' "$1" | tail -n 1; }
parse_op_time_sum() { awk '/Op Time:/ {sum += $3; count += 1} END { if (count > 0) printf "%.6f", sum; }' "$1"; }
parse_op_time_max() { awk '/Op Time:/ {if ($3 > m) m = $3} END { if (m > 0) printf "%.6f", m; }' "$1"; }

run_case() {
    local batch="$1"
    local outfile="${RESULT_DIR}/m3_${batch}.out"
    local errfile="${RESULT_DIR}/m3_${batch}.err"

    note "=== M3 batch ${batch} ==="

    local exit_code=0
    "$M3_BIN" "$batch" >"$outfile" 2>"$errfile" || exit_code=$?

    if [[ $exit_code -ne 0 ]]; then
        note "FAIL: process exited with code ${exit_code}"
        note "  out: ${outfile}"; note "  err: ${errfile}"
        overall_status=1
        return
    fi

    local accuracy op_sum op_max
    accuracy="$(parse_accuracy "$outfile")"
    op_sum="$(parse_op_time_sum "$outfile")"
    op_max="$(parse_op_time_max "$outfile")"

    if [[ -z "$accuracy" ]]; then
        note "FAIL: could not parse Test Accuracy from ${outfile}"
        overall_status=1
        return
    fi

    note "PASS: execution completed"
    note "  accuracy=${accuracy}"
    [[ -n "$op_sum" ]] && note "  op_time_sum_ms=${op_sum}"
    [[ -n "$op_max" ]] && note "  op_time_max_layer_ms=${op_max}"

    if [[ -n "${EXPECTED_ACC[$batch]:-}" ]]; then
        local expected="${EXPECTED_ACC[$batch]}"
        if abs_diff_leq "$accuracy" "$expected" "$ACCURACY_TOL"; then
            note "  public-accuracy-check=PASS (expected ${expected} ± ${ACCURACY_TOL})"
        else
            note "  public-accuracy-check=FAIL (expected ${expected} ± ${ACCURACY_TOL})"
            overall_status=1
        fi
    else
        if float_ge "$accuracy" "$SANITY_MIN_ACC"; then
            note "  edge-batch-sanity=PASS (>= ${SANITY_MIN_ACC})"
        else
            note "  edge-batch-sanity=FAIL (< ${SANITY_MIN_ACC})"
            overall_status=1
        fi
    fi

    if [[ "$batch" -eq 10000 && -n "$op_sum" ]]; then
        if float_le "$op_sum" "$M3_MAX_OP_MS"; then
            note "  op-time-check=PASS (sum ${op_sum} <= ${M3_MAX_OP_MS} ms, full credit)"
        elif float_le "$op_sum" "100"; then
            note "  op-time-check=PARTIAL (sum ${op_sum} between 60 and 100 ms — linear-scaled credit)"
        else
            note "  op-time-check=FAIL (sum ${op_sum} > 100 ms — no credit)"
            overall_status=1
        fi
        if [[ -n "$op_max" ]] && float_le "$op_max" "$M3_COMPETITION_PER_LAYER_MS"; then
            note "  competition-per-layer=PASS (max layer ${op_max} <= ${M3_COMPETITION_PER_LAYER_MS} ms)"
        else
            note "  competition-per-layer=INFO (max layer ${op_max:-?} ms)"
        fi
    fi

    if [[ "$batch" -ge 1000 && -n "$op_sum" ]]; then
        if awk -v t="$op_sum" 'BEGIN { exit !(t < 0.1) }'; then
            note "  warning=suspiciously tiny Op Time sum (likely silent launch failure)"
            overall_status=1
        fi
    fi
}

run_sanitize() {
    local batch=100
    local outfile="${RESULT_DIR}/m3_sanitize_${batch}.out"
    note "=== compute-sanitizer batch ${batch} ==="
    compute-sanitizer "$M3_BIN" "$batch" >"$outfile" 2>&1 || true
    if grep -q "ERROR SUMMARY: 0 errors" "$outfile"; then
        note "  sanitizer=PASS (0 errors)"
    else
        note "  sanitizer=FAIL (see ${outfile})"
        overall_status=1
    fi
}

run_competition() {
    local outfile="${RESULT_DIR}/m3_competition.out"
    note "=== M3 --competition @ B=10000 ==="
    "$M3_BIN" --competition >"$outfile" 2>&1 || true
    if grep -q "Test Accuracy:" "$outfile"; then
        note "  competition-run=PASS (did not abort)"
    else
        note "  competition-run=FAIL or aborted early (max Op Time exceeded 40 ms per layer)"
        overall_status=1
    fi
}

{
    echo "Milestone 3 self-check"
    echo "Result directory: ${RESULT_DIR}"
    echo "Accuracy tol: ${ACCURACY_TOL}  Min edge accuracy: ${SANITY_MIN_ACC}"
    echo "M3 max Op Time @10000: ${M3_MAX_OP_MS} ms  Competition per-layer: ${M3_COMPETITION_PER_LAYER_MS} ms"
    echo
} > "$SUMMARY_FILE"

if ! have_binary "$M3_BIN"; then
    note "OVERALL RESULT: FAIL"
    exit 1
fi

for batch in "${GPU_BATCHES[@]}"; do run_case "$batch"; done

if [[ $RUN_EDGE -eq 1 ]]; then
    note "--- Edge / prime batches ---"
    for batch in "${EDGE_BATCHES[@]}"; do run_case "$batch"; done
fi

if [[ $RUN_SANITIZE -eq 1 ]]; then
    note "--- compute-sanitizer ---"
    run_sanitize
fi

if [[ $RUN_COMPETITION -eq 1 ]]; then
    note "--- --competition mode ---"
    run_competition
fi

if [[ $overall_status -eq 0 ]]; then
    note "OVERALL RESULT: PASS"
else
    note "OVERALL RESULT: FAIL"
fi

echo
echo "Summary written to ${SUMMARY_FILE}"
exit "$overall_status"
