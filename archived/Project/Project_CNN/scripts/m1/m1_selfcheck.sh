#!/bin/bash

set -u
set -o pipefail

# Milestone 1 self-check runner.
# This script approximates the public autograder checks and adds a few useful
# edge-batch sanity checks that catch large-batch launch/configuration bugs.

GPU_BIN="./m1_gpu"
CPU_BIN="./m1_cpu"

RUN_GPU=1
RUN_CPU=0
RESULT_ROOT="${RESULT_ROOT:-selfcheck/m1}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="${RESULT_ROOT}/${TIMESTAMP}"

GPU_BATCHES=(100 1000 4095 4096 8767 10000)
CPU_BATCHES=(100 1000 3000)

ACCURACY_TOL="${ACCURACY_TOL:-0.002}"
SANITY_MIN_ACC="${SANITY_MIN_ACC:-0.80}"
GPU_MAX_OP_MS="${GPU_MAX_OP_MS:-300}"

declare -A GPU_EXPECTED_ACC=(
    [100]="0.86"
    [1000]="0.886"
    [10000]="0.8714"
)

declare -A CPU_EXPECTED_ACC=(
    [100]="0.86"
    [1000]="0.886"
)

usage() {
    cat <<'EOF'
Usage: bash m1_selfcheck.sh [options]

Options:
  --cpu               Run CPU checks in addition to GPU checks
  --cpu-only          Run only CPU checks
  --gpu-only          Run only GPU checks (default)
  --result-dir DIR    Write outputs under DIR instead of selfcheck/m1
  --help              Show this help message

Environment overrides:
  ACCURACY_TOL     Public-batch absolute tolerance (default: 0.002)
  SANITY_MIN_ACC   Minimum acceptable accuracy for edge batches (default: 0.80)
  GPU_MAX_OP_MS    Max total GPU Op Time for batch 10000 (default: 300)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu)
            RUN_CPU=1
            ;;
        --cpu-only)
            RUN_CPU=1
            RUN_GPU=0
            ;;
        --gpu-only)
            RUN_GPU=1
            RUN_CPU=0
            ;;
        --result-dir)
            shift
            RESULT_ROOT="$1"
            RESULT_DIR="${RESULT_ROOT}/${TIMESTAMP}"
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
    shift
done

if [[ $RUN_GPU -eq 0 && $RUN_CPU -eq 0 ]]; then
    echo "Nothing to run: both CPU and GPU checks are disabled." >&2
    exit 1
fi

mkdir -p "$RESULT_DIR"
SUMMARY_FILE="${RESULT_DIR}/summary.txt"

overall_status=0

note() {
    echo "$1" | tee -a "$SUMMARY_FILE"
}

have_binary() {
    local bin="$1"
    if [[ ! -x "$bin" ]]; then
        note "MISSING: ${bin} is not built or not executable."
        overall_status=1
        return 1
    fi
    return 0
}

abs_diff_leq() {
    local actual="$1"
    local expected="$2"
    local tol="$3"
    awk -v a="$actual" -v e="$expected" -v t="$tol" '
        BEGIN {
            d = a - e;
            if (d < 0) d = -d;
            exit !(d <= t);
        }
    '
}

float_ge() {
    local lhs="$1"
    local rhs="$2"
    awk -v a="$lhs" -v b="$rhs" 'BEGIN { exit !(a >= b) }'
}

parse_accuracy() {
    local file="$1"
    awk '/Test Accuracy:/ {print $3}' "$file" | tail -n 1
}

parse_op_time_sum() {
    local file="$1"
    awk '/Op Time:/ {sum += $3; count += 1} END { if (count > 0) printf "%.6f", sum; }' "$file"
}

run_case() {
    local mode="$1"
    local batch="$2"
    local bin="$3"
    local outfile="${RESULT_DIR}/${mode}_${batch}.out"
    local errfile="${RESULT_DIR}/${mode}_${batch}.err"

    note "=== ${mode^^} batch ${batch} ==="

    local exit_code=0
    "$bin" "$batch" >"$outfile" 2>"$errfile" || exit_code=$?

    local accuracy
    accuracy="$(parse_accuracy "$outfile")"
    local op_sum
    op_sum="$(parse_op_time_sum "$outfile")"

    if [[ $exit_code -ne 0 ]]; then
        note "FAIL: process exited with code ${exit_code}"
        note "  out: ${outfile}"
        note "  err: ${errfile}"
        overall_status=1
        return
    fi

    if [[ -z "$accuracy" ]]; then
        note "FAIL: could not parse Test Accuracy from ${outfile}"
        overall_status=1
        return
    fi

    note "PASS: execution completed"
    note "  accuracy=${accuracy}"
    if [[ -n "$op_sum" ]]; then
        note "  op_time_sum_ms=${op_sum}"
    fi

    if [[ "$mode" == "gpu" && -n "${GPU_EXPECTED_ACC[$batch]:-}" ]]; then
        local expected="${GPU_EXPECTED_ACC[$batch]}"
        if abs_diff_leq "$accuracy" "$expected" "$ACCURACY_TOL"; then
            note "  public-accuracy-check=PASS (expected ${expected} ± ${ACCURACY_TOL})"
        else
            note "  public-accuracy-check=FAIL (expected ${expected} ± ${ACCURACY_TOL})"
            overall_status=1
        fi
    elif [[ "$mode" == "cpu" && -n "${CPU_EXPECTED_ACC[$batch]:-}" ]]; then
        local expected="${CPU_EXPECTED_ACC[$batch]}"
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

    if [[ "$mode" == "gpu" && "$batch" -eq 10000 && -n "$op_sum" ]]; then
        if float_ge "$GPU_MAX_OP_MS" "$op_sum"; then
            note "  op-time-check=PASS (sum <= ${GPU_MAX_OP_MS} ms)"
        else
            note "  op-time-check=FAIL (sum > ${GPU_MAX_OP_MS} ms)"
            overall_status=1
        fi
    fi

    if [[ "$mode" == "gpu" && "$batch" -ge 1000 && -n "$op_sum" ]]; then
        if awk -v t="$op_sum" 'BEGIN { exit !(t < 0.1) }'; then
            note "  warning=suspiciously tiny Op Time sum; this often indicates a launch/configuration failure"
            overall_status=1
        fi
    fi
}

{
    echo "Milestone 1 self-check"
    echo "Result directory: ${RESULT_DIR}"
    echo "Accuracy tolerance: ${ACCURACY_TOL}"
    echo "Sanity minimum accuracy: ${SANITY_MIN_ACC}"
    echo "GPU max total Op Time @10000: ${GPU_MAX_OP_MS} ms"
    echo
} > "$SUMMARY_FILE"

if [[ $RUN_GPU -eq 1 ]]; then
    note "--- GPU checks ---"
    if have_binary "$GPU_BIN"; then
        for batch in "${GPU_BATCHES[@]}"; do
            run_case "gpu" "$batch" "$GPU_BIN"
        done
    fi
    note
fi

if [[ $RUN_CPU -eq 1 ]]; then
    note "--- CPU checks ---"
    if have_binary "$CPU_BIN"; then
        for batch in "${CPU_BATCHES[@]}"; do
            run_case "cpu" "$batch" "$CPU_BIN"
        done
    fi
    note
fi

if [[ $overall_status -eq 0 ]]; then
    note "OVERALL RESULT: PASS"
else
    note "OVERALL RESULT: FAIL"
fi

echo
echo "Summary written to ${SUMMARY_FILE}"
exit "$overall_status"
