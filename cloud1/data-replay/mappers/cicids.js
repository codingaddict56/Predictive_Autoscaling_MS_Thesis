// CICIDS-like dataset mapping
// Use src_bytes + dst_bytes as traffic; label != 'normal' -> anomaly/error.
export function cicidsMapper(row) {
  const srcBytes = n(row.src_bytes);
  const dstBytes = n(row.dst_bytes);
  const dur = n(row.duration); // seconds
  const label = (row.label || "").toString().toLowerCase();

  const totalBytes = (isNum(srcBytes) ? srcBytes : 0) + (isNum(dstBytes) ? dstBytes : 0);

  // Basic proxy metrics
  const cpuPct = Math.min(100, Math.max(0, totalBytes / 4096)); // 4KB -> 1% cpu approx
  const memBytes = Math.max(0, totalBytes / 2); // synthetic proportion
  const latency = isNum(dur) ? Math.max(0.001, Math.min(10, dur)) : null;

  const isAttack = label && label !== "normal" && label !== "benign";
  const errors = isAttack ? 1 : 0;
  const anomalies = isAttack ? 1 : 0;

  return {
    cpu_percent: cpuPct,
    memory_bytes: memBytes,
    requests_per_sec: 1,
    requests: 1,
    errors,
    latency_seconds: latency,
    network_bytes: totalBytes,
    anomaly_events: anomalies
  };
}

function n(x){ return x == null ? NaN : Number(x); }
function isNum(x){ return Number.isFinite(x); }
