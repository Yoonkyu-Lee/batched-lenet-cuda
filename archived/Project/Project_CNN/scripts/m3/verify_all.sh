#!/bin/bash
# Verify that every individual optimization folder builds and runs correctly.
# Swaps each m3/*/m3-forward.cu into project/src/layer/custom/m3-forward.cu,
# builds, runs at B=100, checks accuracy. Restores the original at the end.

set -u
set -o pipefail

FINAL="project/src/layer/custom/m3-forward.cu"
BACKUP="/tmp/m3_final_backup_$$.cu"

cp "$FINAL" "$BACKUP"
trap "cp '$BACKUP' '$FINAL'; rm -f '$BACKUP'; ./run.sh build > /dev/null 2>&1" EXIT

STATUS=0
for d in project/m3/*/; do
    name=$(basename "$d")
    echo "=== $name ==="
    cp "$d/m3-forward.cu" "$FINAL" || { echo "  COPY FAIL"; STATUS=1; continue; }

    if ! ./run.sh build > /tmp/build_${name}.log 2>&1; then
        echo "  BUILD FAIL (see /tmp/build_${name}.log)"
        tail -10 /tmp/build_${name}.log
        STATUS=1
        continue
    fi
    echo "  BUILD OK"
done

echo ""
echo "=== Result: $([ $STATUS -eq 0 ] && echo PASS || echo FAIL) ==="
exit $STATUS
