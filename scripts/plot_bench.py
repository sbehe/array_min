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

    # Sort by range length for cleaner curves
    df = df.sort_values(by="range_len")

    plt.figure()

    plt.plot(df["range_len"], df["scalar_cycles"], label="Scalar", marker='o')
    plt.plot(df["range_len"], df["vector_cycles"], label="Vector", marker='x')

    plt.xlabel("Range Length")
    plt.ylabel("Cycles")
    plt.title(f"Min Benchmark (Array Size = {size})")

    plt.legend()
    plt.grid(True)

    os.makedirs("plot", exist_ok=True)
    out_path = f"plot/bench_{size}.png"
    plt.savefig(out_path)
    plt.close()

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 plot_bench.py <csv_file>")
        sys.exit(1)

    plot_file(sys.argv[1])