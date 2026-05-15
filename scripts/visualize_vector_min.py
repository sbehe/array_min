#!/usr/bin/env python3

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_file(csv_path):
    df = pd.read_csv(csv_path)

    if df.empty:
        print(f"Skipping empty file: {csv_path}")
        return

    size = df["array_size"].iloc[0]

    # Sort by range length for cleaner display
    df = df.sort_values(by="range_len")

    plt.figure(figsize=(10, 6))

    plt.scatter(df["range_len"], df["vector_cycles"], label="Vector", marker='o', s=20, alpha=0.6)

    plt.xlabel("Range Length", fontsize=12)
    plt.ylabel("Cycles", fontsize=12)
    plt.title(f"Vector Min Benchmark (Array Size = {size})", fontsize=14)

    # Set y-axis range from 20 to 100 with granularity of 20
    plt.ylim(20, 100)
    plt.yticks(range(20, 101, 20))

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs("plot", exist_ok=True)
    out_path = f"plot/bench_{size}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 plot_bench.py <csv_file>")
        sys.exit(1)

    plot_file(sys.argv[1])