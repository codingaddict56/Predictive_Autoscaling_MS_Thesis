#!/usr/bin/env python3
import pandas as pd
import numpy as np

df = pd.read_csv("data/vmcloud_small.csv")

# Fill CPU usage based on normalized execution time pattern (0–1)
if "cpu_usage" in df.columns:
    df["cpu_usage"] = (
        df["execution_time"].fillna(df["execution_time"].mean()) / df["execution_time"].max()
    ).clip(0, 1)

# Memory usage roughly correlated with CPU
if "memory_usage" in df.columns:
    df["memory_usage"] = np.clip(df["cpu_usage"] * np.random.uniform(0.7, 1.2, size=len(df)), 0, 1)

# Simulate network traffic in MB/s
if "network_traffic" in df.columns:
    df["network_traffic"] = np.clip(
        (df["cpu_usage"] * np.random.uniform(0.5, 2.0, size=len(df))) + np.random.normal(0.1, 0.05, len(df)),
        0.01,
        None
    )

# Drop any remaining NaNs
df = df.fillna(method="ffill").fillna(method="bfill")

df.to_csv("data/vmcloud_small.csv", index=False)
print("[OK] Patched vmcloud_small.csv with synthetic realistic workload data")
print(df.head())
