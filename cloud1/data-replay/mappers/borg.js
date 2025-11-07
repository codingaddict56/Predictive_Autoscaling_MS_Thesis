// Derived mapping for Borg traces subset
// Columns seen: assigned_memory (GB), average_usage (CPU cores share 0..1?), failed (0/1), time (ns?)
export function borgMapper(row) {
  const avg = n(row.average_usage);
  const assignedMem = n(row.assigned_memory);
  const failed = n(row.failed);
  const cpi = n(row.cycles_per_instruction);
  const mai = n(row.memory_accesses_per_instruction);

  // Interpret average_usage as CPU percent (0..1 -> 0..100)
  const cpuPct = isNum(avg) ? Math.max(0, Math.min(100, avg * 100)) : null;

  // assigned_memory appears to be in GB per sample
  const memBytes = isNum(assignedMem) ? assignedMem * 1024 * 1024 * 1024 : null;

  // Error events: count failed rows as errors, else 0
  const errors = isNum(failed) && failed > 0 ? 1 : 0;

  // Latency proxy: inverse of CPI is throughput-ish; map to a synthetic latency range
  const latencySec = isNum(cpi) ? Math.max(0.005, Math.min(5, cpi / 1e4)) : null;

  // Network proxy from memory-accesses per instruction (totally synthetic, but monotonic)
  const netBytes = isNum(mai) ? Math.max(0, Math.floor(mai * 1024)) : null;

  return {
    cpu_percent: cpuPct,
    memory_bytes: memBytes,
    requests_per_sec: 1,
    requests: 1,
    errors,
    latency_seconds: latencySec,
    network_bytes: netBytes,
    anomaly_events: 0
  };
}

function n(x){ return x == null ? NaN : Number(x); }
function isNum(x){ return Number.isFinite(x); }
