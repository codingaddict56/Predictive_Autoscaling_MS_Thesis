// Expected columns (example):
// timestamp,cpu_util,memory_used_gb,request_rate,error_rate,latency_p95_ms
export function vmcloudMapper(row) {
  const cpu = num(row.cpu_util);
  const memGB = num(row.memory_used_gb);
  const rps = num(row.request_rate);
  const errRate = num(row.error_rate); // errors per second
  const p95ms = num(row.latency_p95_ms);

  return {
    cpu_percent: isNum(cpu) ? clamp(cpu, 0, 100) : null,
    memory_bytes: isNum(memGB) ? memGB * 1024 * 1024 * 1024 : null,
    requests_per_sec: isNum(rps) ? Math.max(0, rps) : null,
    requests: isNum(rps) ? Math.max(0, rps) : null, // increment each tick by rps (1s tick approx)
    errors: isNum(errRate) ? Math.max(0, errRate) : 0,
    latency_seconds: isNum(p95ms) ? Math.max(0, p95ms / 1000) : null,
    network_bytes: null,
    anomaly_events: 0
  };
}

function num(x) { return x == null ? NaN : Number(x); }
function isNum(x) { return Number.isFinite(x); }
function clamp(v, a, b) { return Math.min(Math.max(v, a), b); }
