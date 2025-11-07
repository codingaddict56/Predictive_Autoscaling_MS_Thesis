#!/usr/bin/env node

const axios = require('axios');
const { performance } = require('perf_hooks');

class LoadTester {
  constructor(baseUrl = 'http://localhost:8080') {
    this.baseUrl = baseUrl;
    this.results = [];
    this.isRunning = false;
  }

  async makeRequest(endpoint, method = 'GET', data = null) {
    const startTime = performance.now();
    try {
      const response = await axios({
        method,
        url: `${this.baseUrl}${endpoint}`,
        data,
        timeout: 10000,
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      const endTime = performance.now();
      const duration = endTime - startTime;
      
      return {
        success: true,
        status: response.status,
        duration,
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      const endTime = performance.now();
      const duration = endTime - startTime;
      
      return {
        success: false,
        status: error.response?.status || 0,
        error: error.message,
        duration,
        timestamp: new Date().toISOString()
      };
    }
  }

  async simulateLoad(duration = 60000, requestsPerSecond = 10, endpoints = ['/', '/api/data']) {
    console.log(`Starting load test for ${duration/1000} seconds at ${requestsPerSecond} req/sec`);
    console.log(`Target: ${this.baseUrl}`);
    
    this.isRunning = true;
    this.results = [];
    
    const startTime = Date.now();
    const endTime = startTime + duration;
    const interval = 1000 / requestsPerSecond;
    
    let requestCount = 0;
    
    while (Date.now() < endTime && this.isRunning) {
      const promises = [];
      
      // Create multiple concurrent requests
      for (let i = 0; i < Math.ceil(requestsPerSecond / 10); i++) {
        const endpoint = endpoints[Math.floor(Math.random() * endpoints.length)];
        promises.push(this.makeRequest(endpoint));
      }
      
      const batchResults = await Promise.all(promises);
      this.results.push(...batchResults);
      requestCount += batchResults.length;
      
      // Progress update
      if (requestCount % 100 === 0) {
        const elapsed = (Date.now() - startTime) / 1000;
        const rate = requestCount / elapsed;
        console.log(`Completed ${requestCount} requests (${rate.toFixed(1)} req/sec)`);
      }
      
      await new Promise(resolve => setTimeout(resolve, interval));
    }
    
    this.isRunning = false;
    return this.generateReport();
  }

  async simulateSpikeLoad(spikeDuration = 30000, spikeIntensity = 50) {
    console.log(`Starting spike load test: ${spikeIntensity} req/sec for ${spikeDuration/1000} seconds`);
    
    // Normal load for 30 seconds
    console.log('Phase 1: Normal load (30 seconds)');
    await this.simulateLoad(30000, 5);
    
    // Spike load
    console.log('Phase 2: Spike load');
    await this.simulateLoad(spikeDuration, spikeIntensity);
    
    // Return to normal
    console.log('Phase 3: Return to normal load (30 seconds)');
    await this.simulateLoad(30000, 5);
    
    return this.generateReport();
  }

  async simulateStressTest() {
    console.log('Starting stress test with increasing load');
    
    const phases = [
      { duration: 30000, rps: 5, name: 'Low load' },
      { duration: 30000, rps: 20, name: 'Medium load' },
      { duration: 30000, rps: 50, name: 'High load' },
      { duration: 30000, rps: 100, name: 'Stress load' },
      { duration: 30000, rps: 5, name: 'Recovery' }
    ];
    
    for (const phase of phases) {
      console.log(`\nPhase: ${phase.name} (${phase.rps} req/sec for ${phase.duration/1000}s)`);
      await this.simulateLoad(phase.duration, phase.rps);
      await new Promise(resolve => setTimeout(resolve, 5000)); // Wait between phases
    }
    
    return this.generateReport();
  }

  async simulateResourceLoad(duration = 60000) {
    console.log(`Starting resource load simulation for ${duration/1000} seconds`);
    
    const loadTypes = [
      { intensity: 'low', duration: 10000 },
      { intensity: 'medium', duration: 20000 },
      { intensity: 'high', duration: 20000 },
      { intensity: 'medium', duration: 10000 }
    ];
    
    for (const loadType of loadTypes) {
      console.log(`\nSimulating ${loadType.intensity} load for ${loadType.duration/1000}s`);
      
      try {
        await axios.post(`${this.baseUrl}/load`, {
          duration: loadType.duration,
          intensity: loadType.intensity
        });
        
        console.log(`${loadType.intensity} load simulation completed`);
      } catch (error) {
        console.error(`Failed to simulate ${loadType.intensity} load:`, error.message);
      }
      
      await new Promise(resolve => setTimeout(resolve, 2000));
    }
  }

  generateReport() {
    const totalRequests = this.results.length;
    const successfulRequests = this.results.filter(r => r.success).length;
    const failedRequests = totalRequests - successfulRequests;
    
    const durations = this.results.map(r => r.duration);
    const avgDuration = durations.reduce((a, b) => a + b, 0) / durations.length;
    const minDuration = Math.min(...durations);
    const maxDuration = Math.max(...durations);
    
    const successRate = (successfulRequests / totalRequests) * 100;
    
    const report = {
      summary: {
        totalRequests,
        successfulRequests,
        failedRequests,
        successRate: `${successRate.toFixed(2)}%`,
        avgResponseTime: `${avgDuration.toFixed(2)}ms`,
        minResponseTime: `${minDuration.toFixed(2)}ms`,
        maxResponseTime: `${maxDuration.toFixed(2)}ms`
      },
      statusCodes: this.results.reduce((acc, result) => {
        const status = result.status || 'error';
        acc[status] = (acc[status] || 0) + 1;
        return acc;
      }, {}),
      errors: this.results.filter(r => !r.success).map(r => r.error)
    };
    
    console.log('\n=== Load Test Report ===');
    console.log(`Total Requests: ${report.summary.totalRequests}`);
    console.log(`Successful: ${report.summary.successfulRequests}`);
    console.log(`Failed: ${report.summary.failedRequests}`);
    console.log(`Success Rate: ${report.summary.successRate}`);
    console.log(`Avg Response Time: ${report.summary.avgResponseTime}`);
    console.log(`Min Response Time: ${report.summary.minResponseTime}`);
    console.log(`Max Response Time: ${report.summary.maxResponseTime}`);
    
    if (Object.keys(report.statusCodes).length > 0) {
      console.log('\nStatus Codes:');
      Object.entries(report.statusCodes).forEach(([status, count]) => {
        console.log(`  ${status}: ${count}`);
      });
    }
    
    if (report.errors.length > 0) {
      console.log('\nErrors:');
      report.errors.slice(0, 5).forEach(error => {
        console.log(`  ${error}`);
      });
      if (report.errors.length > 5) {
        console.log(`  ... and ${report.errors.length - 5} more errors`);
      }
    }
    
    return report;
  }

  stop() {
    this.isRunning = false;
    console.log('Load test stopped');
  }
}

// CLI interface
async function main() {
  const args = process.argv.slice(2);
  const baseUrl = args[0] || 'http://localhost:8080';
  const testType = args[1] || 'normal';
  
  const loadTester = new LoadTester(baseUrl);
  
  // Handle graceful shutdown
  process.on('SIGINT', () => {
    console.log('\nReceived SIGINT, stopping load test...');
    loadTester.stop();
    process.exit(0);
  });
  
  try {
    switch (testType) {
      case 'spike':
        await loadTester.simulateSpikeLoad();
        break;
      case 'stress':
        await loadTester.simulateStressTest();
        break;
      case 'resource':
        await loadTester.simulateResourceLoad();
        break;
      case 'normal':
      default:
        await loadTester.simulateLoad();
        break;
    }
  } catch (error) {
    console.error('Load test failed:', error.message);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

module.exports = LoadTester; 