#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

log_dir="results/ring_radius25_widths"
mkdir -p "$log_dir"

for dim in 64 128 256 512; do
  echo "===== Ring | radius=25 | model=SW | dim=${dim} ====="
  python train.py --task_type Ring --min_radius 25 --max_radius 26 --model_type SW --dim "$dim" \
    2>&1 | tee "$log_dir/width_${dim}.log"
done

echo
echo "===== Summary: Ring radius 25 (all widths) ====="
for dim in 64 128 256 512; do
  grep '^GNN:' "$log_dir/width_${dim}.log" | tail -n 1
done
