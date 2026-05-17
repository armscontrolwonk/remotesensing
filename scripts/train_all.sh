#!/usr/bin/env bash
# Train all three band-pair models sequentially.
set -euo pipefail

CONFIG="${1:-configs/default.yaml}"

for pair in blue green red; do
    echo "=== training $pair ==="
    python -m superres.train --config "$CONFIG" --pair "$pair"
done
