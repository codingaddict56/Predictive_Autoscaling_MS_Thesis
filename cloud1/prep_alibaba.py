#!/usr/bin/env python3
import argparse, pandas as pd, numpy as np
from pathlib import Path

# --- helpers ---------------------------------------------------------------
TIME_CANDIDATES = ["timestamp","time","ts","start_time","end_time","record_time","gmt_create","gmt_modified"]
CPU_CANDIDATES  = ["cpu_usage","cpu_util","cpu","cpu_rate","mean_cpu_usage","avg_cpu_usage","machine_cpu","container_cpu"]
MEM_CANDIDATES  = ["memory_usage","mem_usage","mem_util","mem","memory","machine_mem","container_mem","mem_used"]
NET_IN_CANDS    = ["net_in","rx_bytes","rx","receive_bytes","bytes_received","net_rx"]
NET_OUT_CANDS   = ["net_out","tx_bytes","tx","transmit_bytes","bytes_sent","net_tx"]
EXEC_CANDIDATES = ["execution_time","duration","latency","latency_ms","response_time","elapsed"]

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns: return c
    # try case-insensitive
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower: return lower[c]
    return None

def ensure_timestamp(df, default_start="2025-01-01 00:00:00", freq="10S"):
    tcol = pick_col(df, TIME_CANDIDATES)
    if tcol:
        ts = pd.to_datetime(df[tcol], errors="coerce")
        if ts.notna().any():
            df = df.copy()
            df["timestamp"] = ts
            return df.dropna(subset=["timestamp"])
    # fallback: synthesize evenly spaced timestamps
    n = len(df)
    ts = pd.date_range(default_start, periods=n, freq=freq)
    out = df.copy()
    out["timestamp"] = ts
    return out

def as_numeric(df, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def safe_quantile(s: pd.Series, q: float, window: int, min_periods: int):
    return s.rolling(window=window, min_periods=min_periods).quantile(q)

# --- main prep -------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", default="data/machine_usage_sample_2000.csv")
    ap.add_argument("--container", default="data/container_usage_sample_2000.csv")
    ap.add_argument("--task", default="data/batch_task_sample_2000.csv")
    ap.add_argument("--out", default="data/vmcloud_small.csv")
    ap.add_argument("--freq", default="10S")          # resample to 10s
    ap.add_argument("--rt_window_sec", type=int, default=60)  # p95 window
    args = ap.parse_args()

    # Load what exists
    def try_read(p):
        pth = Path(p)
        if pth.exists():
            return pd.read_csv(pth, low_memory=False)
        return None

    df_mach = try_read(args.machine)
    df_cont = try_read(args.container)
    df_task = try_read(args.task)

    parts = []
    # container preferred (often closer to app)
    for name, df in [("container", df_cont), ("machine", df_mach)]:
        if df is None: continue
        df = ensure_timestamp(df, freq=args.freq)
        cpu_col = pick_col(df, CPU_CANDIDATES)
        mem_col = pick_col(df, MEM_CANDIDATES)
        rx_col  = pick_col(df, NET_IN_CANDS)
        tx_col  = pick_col(df, NET_OUT_CANDS)
        exec_col= pick_col(df, EXEC_CANDIDATES)

        keep = ["timestamp"]
        rename_map = {}

        if cpu_col: rename_map[cpu_col] = f"{name}_cpu"
        if mem_col: rename_map[mem_col] = f"{name}_mem"
        if rx_col:  rename_map[rx_col]  = f"{name}_rx_bytes"
        if tx_col:  rename_map[tx_col]  = f"{name}_tx_bytes"
        if exec_col:rename_map[exec_col]= f"{name}_execution_time"

        df = df.rename(columns=rename_map)
        keep += list(rename_map.values())
        df = df[keep]

        df = as_numeric(df, [c for c in df.columns if c!="timestamp"])
        parts.append(df)

    if not parts:
        raise SystemExit("No machine/container CSVs found. Please check paths.")

    # merge on timestamp
    base = parts[0]
    for p in parts[1:]:
        base = pd.merge_asof(
            base.sort_values("timestamp"),
            p.sort_values("timestamp"),
            on="timestamp", direction="nearest", tolerance=pd.Timedelta(args.freq)
        )
    base = base.sort_values("timestamp").drop_duplicates("timestamp")

    # build unified columns
    out = pd.DataFrame(index=base.index)
    out["timestamp"] = base["timestamp"]

    # Prefer container metrics; fallback to machine
    def coalesce(cols):
        if not cols: return None
        s = None
        for c in cols:
            if c in base.columns:
                s = base[c] if s is None else s.fillna(base[c])
        return s

    out["cpu_usage"]    = coalesce(["container_cpu","machine_cpu"])
    out["memory_usage"] = coalesce(["container_mem","machine_mem"])

    # Network: sum rx+tx if present, convert bytes/s → MB/s (if they were bytes increments per sample this is still a rough feature)
    rx = coalesce(["container_rx_bytes","machine_rx_bytes"])
    tx = coalesce(["container_tx_bytes","machine_tx_bytes"])
    net_bytes = None
    if rx is not None or tx is not None:
        rx = rx if rx is not None else 0
        tx = tx if tx is not None else 0
        net_bytes = (rx.fillna(0) + tx.fillna(0))
        out["network_traffic"] = (net_bytes.astype(float)) / (1024*1024)
    else:
        out["network_traffic"] = np.nan

    # Execution time: prefer real one
    out["execution_time"] = coalesce(["container_execution_time","machine_execution_time"])

    # Resample to desired freq
    out = out.set_index("timestamp").sort_index()
    out = out.resample(args.freq).mean()

    # If execution_time missing, synthesize from cpu + noise (ms)
    if out["execution_time"].isna().all():
        cpu = out["cpu_usage"].fillna(method="ffill").fillna(0)
        # simple synthetic latency model (milliseconds)
        exec_ms = 80 + 400 * cpu + np.random.normal(0, 10, size=len(cpu))
        out["execution_time"] = np.clip(exec_ms, 20, None)

    # Build response_time_p95 from rolling 95th percentile of execution_time
    # If your execution_time already in ms, convert to seconds right after p95.
    win = max(2, int(np.ceil(args.rt_window_sec / pd.Timedelta(args.freq).total_seconds())))
    rt_p95_ms = safe_quantile(out["execution_time"], 0.95, window=win, min_periods=max(2, win//2))
    rt_p95_ms = rt_p95_ms.bfill().ffill()
    out["response_time_p95"] = (rt_p95_ms / 1000.0)  # seconds

    # Final tidy
    out_final = out.reset_index()[["timestamp","cpu_usage","memory_usage","network_traffic","execution_time","response_time_p95"]]
    # Drop rows that are entirely empty except timestamp
    mask_nonempty = out_final[["cpu_usage","memory_usage","network_traffic","execution_time","response_time_p95"]].notna().any(axis=1)
    out_final = out_final[mask_nonempty]

    # Fill small gaps
    out_final[["cpu_usage","memory_usage","network_traffic","execution_time","response_time_p95"]] = \
        out_final[["cpu_usage","memory_usage","network_traffic","execution_time","response_time_p95"]].interpolate().bfill().ffill()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_final.to_csv(args.out, index=False)
    print(f"[OK] wrote {args.out} with {len(out_final)} rows")

if __name__ == "__main__":
    main()
