#!/bin/bash

set -u
set -o pipefail

# Milestone 2 self-check runner.
# This script approximates the public grading checks for both the unrolled and
# fused GPU implementations and adds a few larger-batch sanity checks.

UNROLL_BIN="./m2_unroll"
FUSED_BIN="./m2_fused"

RUN_UNROLL=1
RUN_FUSED=1
RESULT_ROOT="${RESULT_ROOT:-selfcheck/m2}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="${RESULT_ROOT}/${TIMESTAMP}"

GPU_BATCHES=(100 1000 4095 4096 8767 10000)

ACCURACY_TOL="${ACCURACY_TOL:-0.002}"
SANITY_MIN_ACC="${SANITY_MIN_ACC:-0.80}"
UNROLL_MAX_OP_MS="${UNROLL_MAX_OP_MS:-1200}"
FUSED_MAX_OP_MS="${FUSED_MAX_OP_MS:-200}"

declare -A EXPECTED_ACC=(
    [100]="0.86"
    [1000]="0.886"
    [10000]="0.8714"
)

usage() {
    cat <<'EOF'
Usage: bash m2_selfcheck.sh [options]

Options:
  --unroll-only       Run only the unrolled implementation checks
  --fused-only        Run only the fused implementation checks
  --result-dir DIR    Write outputs under DIR instead of selfcheck/m2
  --help              Show this help message

Environment overrides:
  ACCURACY_TOL      Public-batch absolute tolerance (default: 0.002)
  SANITY_MIN_ACC    Minimum acceptable accuracy for edge batches (default: 0.80)
  UNROLL_MAX_OP_MS  Max total Op Time for unroll batch 10000 (default: 1200)
  FUSED_MAX_OP_MS   Max total Op Time for fused batch 10000 (default: 200)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --unroll-only)
            RUN_UNROLL=1
            RUN_FUSED=0
            ;;
        --fused-only)
            RUN_UNROLL=0
            RUN_FUSED=1
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

if [[ $RUN_UNROLL -eq 0 && $RUN_FUSED -eq 0 ]]; then
    echo "Nothing to run: both implementations are disabled." >&2
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
    local variant="$1"
    local batch="$2"
    local bin="$3"
    local op_limit="$4"
    local outfile="${RESULT_DIR}/${variant}_${batch}.out"
    local errfile="${RESULT_DIR}/${variant}_${batch}.err"

    note "=== ${variant^^} batch ${batch} ==="

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
        if float_ge "$op_limit" "$op_sum"; then
            note "  op-time-check=PASS (sum <= ${op_limit} ms)"
        else
            note "  op-time-check=FAIL (sum > ${op_limit} ms)"
            overall_status=1
        fi
    fi

    if [[ "$batch" -ge 1000 && -n "$op_sum" ]]; then
        if awk -v t="$op_sum" 'BEGIN { exit !(t < 0.1) }'; then
            note "  warning=suspiciously tiny Op Time sum; this often indicates a launch/configuration failure"
            overall_status=1
        fi
    fi
}

{
    echo "Milestone 2 self-check"
    echo "Result directory: ${RESULT_DIR}"
    echo
} > "$SUMMARY_FILE"

if [[ $RUN_UNROLL -eq 1 ]]; then
    note "--- Unroll checks ---"
    if have_binary "$UNROLL_BIN"; then
        for batch in "${GPU_BATCHES[@]}"; do
            run_case "unroll" "$batch" "$UNROLL_BIN" "$UNROLL_MAX_OP_MS"
        done
    fi
    note
fi

if [[ $RUN_FUSED -eq 1 ]]; then
    note "--- Fused checks ---"
    if have_binary "$FUSED_BIN"; then
        for batch in "${GPU_BATCHES[@]}"; do
            run_case "fused" "$batch" "$FUSED_BIN" "$FUSED_MAX_OP_MS"
        done
    fi
    note
fi

if [[ $overall_status -eq 0 ]]; then
    note "OVERALL RESULT: PASS"
else
    note "OVERALL RESULT: FAIL"
fi

note
note "Summary written to ${SUMMARY_FILE}"

exit $overall_status
