// controller.js
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const winston = require('winston');
const cron = require('node-cron');
const axios = require('axios');
const { v4: uuidv4 } = require('uuid');

// Prometheus metrics
const promClient = require('prom-client');
const register = promClient.register;

// --- Prometheus gauges/counters (names avoid collisions with class fields) ---
const scalingDecisionsTotal = new promClient.Counter({
  name: 'scaling_decisions_total',
  help: 'Total number of scaling decisions made',
  labelNames: ['decision', 'reason'],
});

const scalingEventsTotal = new promClient.Counter({
  name: 'scaling_events_total',
  help: 'Total number of scaling events',
  labelNames: ['event_type', 'target'],
});

const currentReplicasGauge = new promClient.Gauge({
  name: 'current_replicas',
  help: 'Current number of replicas',
});

const targetReplicasGauge = new promClient.Gauge({
  name: 'target_replicas',
  help: 'Target number of replicas',
});

const scalingLatency = new promClient.Histogram({
  name: 'scaling_latency_seconds',
  help: 'Time taken to complete scaling operations',
  buckets: [0.1, 0.5, 1, 2, 5, 10],
});

// Optional predictive observability
let predGaugesInit = false;
const ensurePredGauges = () => {
  if (predGaugesInit) return;
  try {
    new promClient.Gauge({ name: 'predictive_mean', help: 'Predictive mean' });
    new promClient.Gauge({ name: 'predictive_sigma', help: 'Predictive sigma' });
    new promClient.Gauge({ name: 'predictive_upper', help: 'Predictive upper bound' });
    new promClient.Gauge({ name: 'predictive_threshold', help: 'Predictive threshold' });
    new promClient.Gauge({ name: 'predictive_wants_scale', help: 'Predictive scale desire 0/1' });
    predGaugesInit = true;
  } catch (_) {
    // ignore double-definition on hot reloads
  }
};

// Logging
const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  defaultMeta: { service: 'auto-scaling-controller' },
  transports: [
    new winston.transports.File({ filename: 'error.log', level: 'error' }),
    new winston.transports.File({ filename: 'combined.log' }),
  ],
});

if (process.env.NODE_ENV !== 'production') {
  logger.add(new winston.transports.Console({ format: winston.format.simple() }));
}

class AutoScalingController {
  constructor() {
    // Core
    this.prometheusUrl = process.env.PROMETHEUS_URL || 'http://prometheus:9090';
    this.app = express();
    this.scalingHistory = [];

    // Replica state
    this.currentReplicas = 2;
    this.minReplicas = 2;
    this.maxReplicas = 10;
    this.lastTargetReplicas = this.currentReplicas;

    // Reactive thresholds
    this.cpuThreshold = Number(process.env.CPU_THRESHOLD || 70);
    this.memoryThreshold = Number(process.env.MEMORY_THRESHOLD || 80); // NOTE: compare against GB below
    this.responseTimeThreshold = Number(process.env.RT_THRESHOLD || 2); // seconds
    this.errorRateThreshold = Number(process.env.ERROR_RATE_THRESHOLD || 0.1);

    // Predictive config (set BEFORE starting loop)
    this.sloP95 = Number(process.env.SLO_P95_SEC || 0.8); // seconds
    this.alpha = Number(process.env.SAFETY_FACTOR || 1.0);
    this.cooldownSec = Number(process.env.SCALE_COOLDOWN_SEC || 60);
    this.minStep = Number(process.env.MIN_SCALE_STEP || 1);
    this.maxStep = Number(process.env.MAX_SCALE_STEP || 3);
    this.predictionFile = process.env.PREDICTION_FILE || '/app/artifacts/predictive_signal.json';
    this.lastScaleAt = 0;

    logger.info(
      `predictive config: file=${this.predictionFile} sloP95=${this.sloP95} ` +
      `alpha=${this.alpha} cooldown=${this.cooldownSec}s`
    );

    // HTTP & metrics
    this.setupMiddleware();
    this.setupRoutes();
    this.setupMetrics();

    // Start loop
    this.startScalingLoop();
  }

  // ---------- HTTP ----------
  setupMiddleware() {
    this.app.use(helmet());
    this.app.use(cors());
    this.app.use(express.json());
  }

  setupRoutes() {
    this.app.get('/metrics', async (req, res) => {
      try {
        res.set('Content-Type', register.contentType);
        res.end(await register.metrics());
      } catch (error) {
        logger.error('Error generating metrics:', error);
        res.status(500).end();
      }
    });

    this.app.get('/health', (req, res) => {
      res.json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        currentReplicas: this.currentReplicas,
        scalingHistory: this.scalingHistory.length,
      });
    });

    this.app.get('/status', (req, res) => {
      res.json({
        currentReplicas: this.currentReplicas,
        targetReplicas: this.lastTargetReplicas,
        scalingHistory: this.scalingHistory.slice(-10),
        thresholds: {
          cpu: this.cpuThreshold,
          memory: this.memoryThreshold,
          responseTime: this.responseTimeThreshold,
          errorRate: this.errorRateThreshold,
        },
        predictive: {
          sloP95: this.sloP95,
          alpha: this.alpha,
          cooldownSec: this.cooldownSec,
          predictionFile: this.predictionFile,
        },
      });
    });

    this.app.post('/scale', async (req, res) => {
      const { replicas, reason = 'manual' } = req.body || {};
      if (!Number.isFinite(replicas)) {
        return res.status(400).json({ error: 'replicas must be a number' });
      }
      if (replicas < this.minReplicas || replicas > this.maxReplicas) {
        return res.status(400).json({ error: 'Invalid replica count', min: this.minReplicas, max: this.maxReplicas });
      }
      try {
        await this.performScaling(replicas, reason);
        res.json({ message: 'Scaling initiated', targetReplicas: replicas, reason });
      } catch (error) {
        logger.error('Manual scaling failed:', error);
        res.status(500).json({ error: 'Scaling failed' });
      }
    });

    this.app.post('/webhook', async (req, res) => {
      try {
        const alerts = req.body.alerts || [];
        logger.info(`Received ${alerts.length} alerts from Alertmanager`);
        for (const alert of alerts) {
          await this.handleAlert(alert);
        }
        res.status(200).json({ message: 'Alerts processed' });
      } catch (error) {
        logger.error('Error processing webhook:', error);
        res.status(500).json({ error: 'Webhook processing failed' });
      }
    });

    this.app.put('/thresholds', (req, res) => {
      const { cpu, memory, responseTime, errorRate } = req.body || {};
      if (cpu !== undefined) this.cpuThreshold = Number(cpu);
      if (memory !== undefined) this.memoryThreshold = Number(memory);
      if (responseTime !== undefined) this.responseTimeThreshold = Number(responseTime);
      if (errorRate !== undefined) this.errorRateThreshold = Number(errorRate);

      logger.info('Thresholds updated', {
        cpu: this.cpuThreshold,
        memory: this.memoryThreshold,
        responseTime: this.responseTimeThreshold,
        errorRate: this.errorRateThreshold,
      });

      res.json({
        message: 'Thresholds updated',
        thresholds: {
          cpu: this.cpuThreshold,
          memory: this.memoryThreshold,
          responseTime: this.responseTimeThreshold,
          errorRate: this.errorRateThreshold,
        },
      });
    });
  }

  // ---------- Metrics ----------
  setupMetrics() {
    currentReplicasGauge.set(this.currentReplicas);
    targetReplicasGauge.set(this.currentReplicas);
  }

  // ---------- Predictive JSON ----------
  readPredictiveSignal() {
    const fs = require('fs');
    try {
      if (!fs.existsSync(this.predictionFile)) return null;
      const obj = JSON.parse(fs.readFileSync(this.predictionFile, 'utf8'));
      if (obj && typeof obj.mean === 'number') return obj;
    } catch (e) {
      logger.warn('Failed to read predictive signal:', e.message);
    }
    return null;
  }

  // ---------- Prometheus (fixed: query each metric separately) ----------
  async fetchMetrics() {
   
    const Q = {
      cpu_usage: 'avg(app_cpu_usage_percent)', // %
      memory_usage: 'avg(app_memory_usage_bytes) / 1024 / 1024 / 1024', // GB
      
      response_time: 'histogram_quantile(0.95, sum(rate(app_response_time_seconds_bucket[5m])) by (le))', // seconds
     
      error_rate: 'sum(rate(app_errors_total[5m]))', // errors/sec
     
      throughput: 'sum(rate(app_requests_total[1m]))', // req/sec
      
    };

    const axiosCfg = {
      timeout: 5000,
      headers: { Accept: 'application/json' },
      // Optional: withCredentials: false
    };
    const base = `${this.prometheusUrl}/api/v1/query`;

    // Helper to run a single instant query safely
    const runQuery = async (name, expr) => {
      try {
        const { data } = await axios.get(base, { ...axiosCfg, params: { query: expr } });
        if (data?.status !== 'success') {
          logger.error(`PromQL failed [${name}]:`, data?.error || 'unknown error');
          return 0;
        }
        const result = data?.data?.result || [];
        if (result.length === 0) return 0;

        // Instant vector: value is [ ts, "number" ]
        // Scalar response: { resultType: "scalar", result: [ ts, "number" ] }
        if (data.data.resultType === 'scalar' && Array.isArray(data.data.result)) {
          const v = parseFloat(data.data.result[1]);
          return Number.isFinite(v) ? v : 0;
        }

        const sample = result[0];
        if (sample.value && Array.isArray(sample.value)) {
          const v = parseFloat(sample.value[1]);
          return Number.isFinite(v) ? v : 0;
        }
        if (sample.values && Array.isArray(sample.values) && sample.values.length > 0) {
          const last = sample.values[sample.values.length - 1];
          const v = parseFloat(last[1]);
          return Number.isFinite(v) ? v : 0;
        }
        return 0;
      } catch (e) {
        // Log concise server error message if present
        const msg = e?.response?.data?.error || e.message;
        logger.error(`PromQL error [${name}] expr="${expr}": ${msg}`);
        return 0;
      }
    };

    // Fire queries in parallel
    const names = Object.keys(Q);
    const promises = names.map((n) => runQuery(n, Q[n]).then((v) => [n, v]));
    const pairs = await Promise.all(promises);

    // Assemble output
    const out = {};
    for (const [k, v] of pairs) out[k] = v;

    return out; // { cpu_usage, memory_usage (GB), response_time (s), error_rate, throughput }
  }

  // ---------- Decision logic ----------
  async analyzeMetrics() {
    const metrics = await this.fetchMetrics();

    // ==== Predictive latency-first rule ====
    const signal = this.readPredictiveSignal();
    if (signal && typeof signal.mean === 'number') {
      ensurePredGauges();

      const metric = (signal.metric || '').toLowerCase();
      const mean = signal.mean; // already normalized to seconds by pipeline when latency
      const sigma = Number.isFinite(signal.sigma) ? signal.sigma : 0;
      const upper = mean + this.alpha * sigma;

      // Decide threshold/decision value by metric type
      let thresholdName = 'SLO p95 (seconds)';
      let thresholdValue = this.sloP95;
      let decisionValue = upper; // latency uses mean + α·σ

      if (metric.includes('cpu')) {
        thresholdName = 'CPU threshold (%)';
        thresholdValue = this.cpuThreshold;
        decisionValue = mean; // for CPU we use mean
      }

      // Export gauges (optional)
      try {
        register.getSingleMetric('predictive_mean').set(mean);
        register.getSingleMetric('predictive_sigma').set(sigma);
        register.getSingleMetric('predictive_upper').set(upper);
        register.getSingleMetric('predictive_threshold').set(thresholdValue);
      } catch (_) {}

      logger.info(
        `predictive read: metric=${metric || 'unknown'} mean=${mean.toFixed(4)} ` +
        `sigma=${sigma.toFixed(4)} alpha=${this.alpha} decision=${decisionValue.toFixed(4)} ` +
        `threshold(${thresholdName})=${thresholdValue}`
      );

      const wantsScale = decisionValue > thresholdValue ? 1 : 0;
      try { register.getSingleMetric('predictive_wants_scale').set(wantsScale); } catch (_) {}

      if (wantsScale) {
        const step = Math.max(this.minStep, Math.min(this.maxStep, 1));
        const target = Math.min(this.maxReplicas, this.currentReplicas + step);
        return {
          cpuUtilization: metrics.cpu_usage || 0,
          memoryUtilization: metrics.memory_usage || 0,
          responseTime: metrics.response_time || 0,
          errorRate: metrics.error_rate || 0,
          throughput: metrics.throughput || 0,
          shouldScale: target !== this.currentReplicas,
          scaleDirection: target > this.currentReplicas ? 'up' : 'none',
          reason: `predictive_${metric || 'p95'}: value=${decisionValue.toFixed(4)} ` +
                  `threshold=${thresholdValue} (mean=${mean.toFixed(4)}, σ=${sigma.toFixed(4)}, α=${this.alpha})`,
          targetReplicas: target,
        };
      } else {
        logger.info('predictive decision: no scale (decision <= threshold)');
      }
    }
    // ==== end predictive block ====

    // Reactive (fallback) analysis
    logger.info('Current metrics:', metrics);
    const analysis = {
      cpuUtilization: metrics.cpu_usage || 0,          // %
      memoryUtilization: metrics.memory_usage || 0,    // GB
      responseTime: metrics.response_time || 0,        // seconds
      errorRate: metrics.error_rate || 0,              // errors/sec
      throughput: metrics.throughput || 0,             // req/sec
      shouldScale: false,
      scaleDirection: 'none',
      reason: '',
      targetReplicas: this.currentReplicas,
    };

    if (analysis.cpuUtilization > this.cpuThreshold) {
      analysis.shouldScale = true;
      analysis.scaleDirection = 'up';
      analysis.reason = `High CPU utilization: ${analysis.cpuUtilization.toFixed(2)}%`;
      analysis.targetReplicas = Math.min(
        this.maxReplicas,
        Math.ceil(this.currentReplicas * (analysis.cpuUtilization / this.cpuThreshold))
      );
    }

    // NOTE: memoryThreshold is interpreted as GB here to match memory_usage metric.
    if (analysis.memoryUtilization > this.memoryThreshold) {
      if (!analysis.shouldScale || analysis.targetReplicas < this.currentReplicas * 1.5) {
        analysis.shouldScale = true;
        analysis.scaleDirection = 'up';
        analysis.reason = `High memory usage: ${analysis.memoryUtilization.toFixed(2)} GB`;
        analysis.targetReplicas = Math.min(this.maxReplicas, Math.ceil(this.currentReplicas * 1.5));
      }
    }

    if (analysis.responseTime > this.responseTimeThreshold) {
      if (!analysis.shouldScale || analysis.targetReplicas < this.currentReplicas * 1.3) {
        analysis.shouldScale = true;
        analysis.scaleDirection = 'up';
        analysis.reason = `High response time: ${analysis.responseTime.toFixed(2)}s`;
        analysis.targetReplicas = Math.min(this.maxReplicas, Math.ceil(this.currentReplicas * 1.3));
      }
    }

    if (analysis.errorRate > this.errorRateThreshold) {
      if (!analysis.shouldScale || analysis.targetReplicas < this.currentReplicas * 1.2) {
        analysis.shouldScale = true;
        analysis.scaleDirection = 'up';
        analysis.reason = `High error rate: ${analysis.errorRate.toFixed(3)} errors/sec`;
        analysis.targetReplicas = Math.min(this.maxReplicas, Math.ceil(this.currentReplicas * 1.2));
      }
    }

    if (!analysis.shouldScale && this.currentReplicas > this.minReplicas) {
      const lowUtil =
        analysis.cpuUtilization < this.cpuThreshold * 0.5 &&
        // interpret memory threshold as GB
        analysis.memoryUtilization < this.memoryThreshold * 0.5 &&
        analysis.responseTime < this.responseTimeThreshold * 0.5 &&
        analysis.errorRate < this.errorRateThreshold * 0.5;

      if (lowUtil) {
        analysis.shouldScale = true;
        analysis.scaleDirection = 'down';
        analysis.reason = 'Low resource utilization';
        analysis.targetReplicas = Math.max(this.minReplicas, Math.floor(this.currentReplicas * 0.8));
      }
    }

    return analysis;
  }

  // ---------- Actuator ----------
  async performScaling(targetReplicas, reason = 'auto') {
    const startTime = Date.now();
    try {
      logger.info(`Initiating scaling to ${targetReplicas} replicas. Reason: ${reason}`);

      // Simulate scaling (replace with real orchestration call if needed)
      await new Promise(resolve => setTimeout(resolve, 2000));

      const oldReplicas = this.currentReplicas;
      this.currentReplicas = targetReplicas;
      this.lastTargetReplicas = targetReplicas;

      // Update metrics
      currentReplicasGauge.set(this.currentReplicas);
      targetReplicasGauge.set(this.currentReplicas);

      const scalingTime = (Date.now() - startTime) / 1000;
      scalingLatency.observe(scalingTime);

      const scalingEvent = {
        id: uuidv4(),
        timestamp: new Date().toISOString(),
        oldReplicas,
        newReplicas: targetReplicas,
        reason,
        duration: scalingTime,
      };

      this.scalingHistory.push(scalingEvent);
      if (this.scalingHistory.length > 100) {
        this.scalingHistory = this.scalingHistory.slice(-100);
      }

      scalingDecisionsTotal.inc({ decision: 'scale', reason });
      scalingEventsTotal.inc({
        event_type: targetReplicas > oldReplicas ? 'scale_up' : 'scale_down',
        target: 'app',
      });

      logger.info(`Scaling completed: ${oldReplicas} -> ${targetReplicas} replicas in ${scalingTime}s`);
      return scalingEvent;
    } catch (error) {
      logger.error('Scaling operation failed:', error);
      scalingDecisionsTotal.inc({ decision: 'failed', reason });
      throw error;
    }
  }

  async handleAlert(alert) {
    const { labels = {} } = alert || {};
    const severity = labels.severity || 'warning';
    const component = labels.component || 'unknown';

    logger.info(`Processing alert: ${labels.alertname || 'unknown'} (${severity})`);

    if (component === 'auto-scaling') {
      if (severity === 'critical') {
        const emergencyReplicas = Math.min(this.maxReplicas, this.currentReplicas * 2);
        await this.performScaling(emergencyReplicas, `emergency_${labels.alertname || 'unknown'}`);
      } else if (severity === 'warning') {
        const conservativeReplicas = Math.min(this.maxReplicas, this.currentReplicas + 1);
        await this.performScaling(conservativeReplicas, `warning_${labels.alertname || 'unknown'}`);
      }
    }

    if (component === 'performance' && severity === 'critical') {
      const performanceReplicas = Math.min(this.maxReplicas, this.currentReplicas * 1.5);
      await this.performScaling(performanceReplicas, `performance_${labels.alertname || 'unknown'}`);
    }
  }

  // ---------- Loop ----------
  startScalingLoop() {
    cron.schedule('*/30 * * * * *', async () => {
      try {
        const now = Date.now();
        if (now - this.lastScaleAt < this.cooldownSec * 1000) {
          logger.debug(
            `Cooldown active: ${Math.ceil((this.cooldownSec * 1000 - (now - this.lastScaleAt)) / 1000)}s remaining`
          );
          return;
        }

        const analysis = await this.analyzeMetrics();
        if (analysis.shouldScale && analysis.targetReplicas !== this.currentReplicas) {
          await this.performScaling(analysis.targetReplicas, analysis.reason);
          this.lastScaleAt = Date.now();
        }
      } catch (error) {
        logger.error('Error in scaling loop:', error);
      }
    });

    logger.info('Auto-scaling loop started (cooldown enabled)');
  }

  start(port = 8081) {
    this.app.listen(port, () => {
      logger.info(`Auto-scaling controller started on port ${port}`);
      logger.info(`Metrics available at http://localhost:${port}/metrics`);
      logger.info(`Health check available at http://localhost:${port}/health`);
    });
  }
}

// Start the controller
const controller = new AutoScalingController();
controller.start();

// Graceful shutdown
process.on('SIGTERM', () => {
  logger.info('SIGTERM received, shutting down gracefully');
  process.exit(0);
});
process.on('SIGINT', () => {
  logger.info('SIGINT received, shutting down gracefully');
  process.exit(0);
});

module.exports = AutoScalingController;
