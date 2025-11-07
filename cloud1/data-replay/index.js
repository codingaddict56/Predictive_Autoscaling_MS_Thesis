import express from "express";
import helmet from "helmet";
import morgan from "morgan";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { parse } from "csv-parse";
import client from "prom-client";
import yargs from "yargs";
import { hideBin } from "yargs/helpers";
import { createMapper } from "./mappers/index.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// --- CLI / ENV -------------------------------------------------------
const argv = yargs(hideBin(process.argv))
  .option("file", { alias: "f", type: "string", describe: "CSV file path to replay", default: process.env.REPLAY_FILE || "/data/input.csv" })
  .option("dataset", { alias: "d", type: "string", choices: ["vmcloud","borg","cicids","auto"], default: process.env.REPLAY_DATASET || "auto", describe: "Dataset type (or auto-detect)" })
  .option("rate", { alias: "r", type: "number", default: Number(process.env.REPLAY_RATE || 5), describe: "Rows per second to replay" })
  .option("loop", { type: "boolean", default: process.env.REPLAY_LOOP === "true" || true, describe: "Loop when file ends" })
  .option("time-col", { type: "string", default: process.env.REPLAY_TIME_COL || "", describe: "Optional time column name to pace by deltas (overrides --rate if present)" })
  .option("time-scale", { type: "number", default: Number(process.env.REPLAY_TIME_SCALE || 1), describe: "Scale factor for time deltas when using time column (e.g., 60 = 1 minute of data per second)" })
  .help().argv;

const PORT = Number(process.env.METRICS_PORT || 8080);

// --- Prometheus metrics (names kept identical to your app) -----------
const register = new client.Registry();
client.collectDefaultMetrics({ register });

const appRequests = new client.Counter({
  name: "app_requests_total",
  help: "Total number of synthetic requests processed"
});
const appRequestsPerSecond = new client.Gauge({
  name: "app_requests_per_second",
  help: "Instantaneous requests per second derived from dataset"
});
const appErrors = new client.Counter({
  name: "app_errors_total",
  help: "Total number of synthetic errors"
});
const appCPU = new client.Gauge({
  name: "app_cpu_usage_percent",
  help: "CPU usage percent derived from dataset"
});
const appMem = new client.Gauge({
  name: "app_memory_usage_bytes",
  help: "Memory usage in bytes derived from dataset"
});
const respHist = new client.Histogram({
  name: "app_response_time_seconds",
  help: "Response time histogram derived from dataset",
  buckets: [0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
});
const netBytes = new client.Counter({
  name: "app_network_bytes_total",
  help: "Total bytes observed in dataset (if available)"
});
const anomalies = new client.Counter({
  name: "app_anomaly_events_total",
  help: "Total anomalous/attack events (if labeled datasets)"
});

register.registerMetric(appRequests);
register.registerMetric(appRequestsPerSecond);
register.registerMetric(appErrors);
register.registerMetric(appCPU);
register.registerMetric(appMem);
register.registerMetric(respHist);
register.registerMetric(netBytes);
register.registerMetric(anomalies);

// --- HTTP server -----------------------------------------------------
const app = express();
app.use(helmet());
app.use(morgan("tiny"));

app.get("/", (_req, res) => {
  res.json({ ok: true, replaying: path.basename(argv.file), dataset: argv.dataset, rate: argv.rate });
});
app.get("/health", (_req, res) => res.send("ok"));
app.get("/metrics", async (_req, res) => {
  res.set("Content-Type", register.contentType);
  res.end(await register.metrics());
});

// --- Replay engine ---------------------------------------------------
let stop = false;

async function startReplay() {
  while (!stop) {
    await replayOnce();
    if (!argv.loop) break;
  }
}

function openCSV() {
  const input = fs.createReadStream(argv.file);
  return input.pipe(parse({ columns: true, skip_empty_lines: true }));
}

async function replayOnce() {
  const datasetType = argv.dataset === "auto" ? await detectDatasetKind(argv.file) : argv.dataset;
  const mapRow = createMapper(datasetType);

  const reader = openCSV();
  let lastTs = null;
  for await (const row of reader) {
    const metrics = mapRow(row);
    // Apply mapped fields if present
    if (metrics.requests_per_sec != null) appRequestsPerSecond.set(metrics.requests_per_sec);
    if (metrics.requests != null) appRequests.inc(metrics.requests);
    if (metrics.errors != null) appErrors.inc(metrics.errors);
    if (metrics.cpu_percent != null) appCPU.set(metrics.cpu_percent);
    if (metrics.memory_bytes != null) appMem.set(metrics.memory_bytes);
    if (metrics.latency_seconds != null) respHist.observe(metrics.latency_seconds);
    if (metrics.network_bytes != null) netBytes.inc(metrics.network_bytes);
    if (metrics.anomaly_events != null) anomalies.inc(metrics.anomaly_events);

    // pacing
    if (argv.time_col && row[argv.time_col]) {
      const ts = Number(row[argv.time_col]);
      if (!Number.isNaN(ts)) {
        if (lastTs != null) {
          const deltaMs = (ts - lastTs) / argv.time_scale;
          if (deltaMs > 0 && deltaMs < 60_000) {
            await new Promise(r => setTimeout(r, deltaMs));
          }
        }
        lastTs = ts;
        continue;
      }
    }
    // default fixed rate pacing
    const delayMs = Math.max(1, Math.floor(1000 / Math.max(1, argv.rate)));
    await new Promise(r => setTimeout(r, delayMs));
  }
}

async function detectDatasetKind(filePath) {
  // naive detection by header keywords
  const header = await new Promise((resolve, reject) => {
    const s = fs.createReadStream(filePath, { encoding: "utf8" });
    let buf = "";
    s.on("data", (chunk) => {
      buf += chunk;
      if (buf.indexOf("\n") !== -1) {
        s.destroy();
        resolve(buf.split("\n")[0]);
      }
    });
    s.on("error", reject);
  });
  const h = header.toLowerCase();
  if (h.includes("latency_p95_ms") || h.includes("request_rate") || h.includes("cpu_util")) return "vmcloud";
  if (h.includes("assigned_memory") || h.includes("cycles_per_instruction") || h.includes("average_usage")) return "borg";
  if (h.includes("label") || h.includes("src_bytes") || h.includes("dst_bytes")) return "cicids";
  return "vmcloud"; // sensible default
}

// Start server
app.listen(PORT, () => {
  console.log(`[data-replay] listening on :${PORT}`);
  console.log(`[data-replay] Replaying ${argv.file} as ${argv.dataset} ...`);
  startReplay().catch(err => {
    console.error("Replay error:", err);
    process.exit(1);
  });
});
