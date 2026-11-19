#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

for dim in 64 128 256 512; do
  echo "===== Ring | radius=25 | model=SW | dim=${dim} ====="
  python train.py --task_type Ring --min_radius 25 --max_radius 26 --model_type SW --dim "$dim"
done
