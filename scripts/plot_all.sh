#!/usr/bin/env bash
set -e

CSV_DIR="bench_results"

FILES=(
    bench_32.csv
    bench_64.csv
    bench_128.csv
    bench_256.csv
    bench_512.csv
    bench_1024.csv
)

for file in "${FILES[@]}"; do
    csv_path="$CSV_DIR/$file"

    if [ -f "$csv_path" ]; then
        echo "Processing $csv_path"
        python3 scripts/plot_bench.py "$csv_path"
    else
        echo "Skipping missing file: $csv_path"
    fi
done

echo "All plots generated in ./plots/"