#!/usr/bin/env bash
# Sweep epsilon-scale values for structure.py runs and collect NLTE abundance outputs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ABUNDANCES="8.05 8.15 8.25 8.35 8.45 8.55 8.65 8.75 8.85 8.95 9.05 9.15 9.25 9.35 9.45 9.55 9.65 9.75 9.85 9.95 10.05"
TABLE_DIR="tables"
SCALES=(0.50 0.75 1.00 1.25 1.50)

for scale in "${SCALES[@]}"; do
    scale_tag=${scale/./p}
    log_file="run_eps${scale_tag}.log"
    echo "Running epsilon scale ${scale} (log -> ${log_file})"
    python structure.py --abundances ${ABUNDANCES} --tables-dir "${TABLE_DIR}" --epsilon-scale "${scale}" | tee "${log_file}"
done

echo
echo "Extracted NLTE abundances:"
grep "Final NLTE Oxygen Abundance" run_eps*.log
