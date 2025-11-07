# Predictive_Autoscaling_MS_Thesis

Perfect 🌸 — here’s a **ready-to-copy complete `README.md`** for your GitHub repository.
It includes everything: intro, setup, commands, results, and placeholders for screenshots.
Just copy and paste into your `README.md` file inside your project folder.

---

```markdown
# 🚀 Predictive Auto-Scaling Framework

This project implements a **Machine Learning-based Predictive Auto-Scaling System** that uses Alibaba Cloud traces to **proactively scale cloud resources** before latency degradation occurs.  
It integrates **LSTM**, **Random Forest (RF)**, and **ARIMA** models for forecasting and connects to a **Prometheus–Grafana** monitoring stack for real-time visualization.

---

## 🎯 Project Overview

### 🌩️ Objective
To **predict workload behavior** and **scale resources intelligently** to maintain latency under the Service Level Objective (SLO) while minimizing cost.

### 🧠 Key Features
- Predictive scaling based on `mean + σ > SLO` rule  
- Trained models: **LSTM**, **Random Forest**, **ARIMA**
- Dynamic latency prediction (`response_time_p95`)
- Seamless observability using **Prometheus** and **Grafana**
- Real-time scaling control through **Docker Compose**

---

## 🧰 Tech Stack

| Category | Tools / Technologies |
|-----------|----------------------|
| **Languages** | Python, JavaScript (Node.js) |
| **ML Models** | LSTM (Keras), Random Forest (Sklearn), ARIMA (Statsmodels) |
| **Monitoring** | Prometheus, Grafana |
| **Containerization** | Docker & Docker Compose |
| **Dataset** | Alibaba Cloud Trace Logs (2000 sampled rows) |

---

## 🧩 Architecture

```

```
      ┌──────────────────────────────┐
      │        Data Replay App        │
      │ (Replays Alibaba Cloud traces)│
      └─────────────┬────────────────┘
                    │
      ┌─────────────▼──────────────────┐
      │     Predictive Training Pipeline │
      │ (Feature Engg + LSTM/RF/ARIMA)  │
      └─────────────┬──────────────────┘
                    │
      ┌─────────────▼────────────────┐
      │ Predictive Controller (α·σ + mean)│
      │   Compares to SLO → Scale Decision │
      └─────────────┬────────────────┘
                    │
 ┌──────────────────▼──────────────────┐
 │ Prometheus + Grafana Monitoring Stack │
 │   Metrics, Dashboards, Alerts          │
 └───────────────────────────────────────┘
```

```

---

## 📂 Folder Structure

```

predictive-autoscaling/
├── data/                  # Alibaba dataset samples
├── artifacts/             # Model outputs and predictive_signal.json
├── data-replay/           # Node.js replay service
├── scaling-controller/    # Predictive controller (Node.js)
├── monitoring/            # Prometheus, Grafana, Alertmanager configs
├── docker-compose.yml     # Service orchestration
└── train_pipeline.py      # ML training pipeline

````

---

## ⚙️ How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/predictive-autoscaling.git
cd predictive-autoscaling
````

### 2️⃣ Build and Start All Containers

```bash
docker compose up --build
```

This starts all services:
`app`, `scaling-controller`, `prometheus`, `grafana`, `alertmanager`, and `node-exporter`.

---

## 🌐 Access the Services

| Service                | URL                                                            | Description                       |
| ---------------------- | -------------------------------------------------------------- | --------------------------------- |
| **App Metrics**        | [http://localhost:8080/metrics](http://localhost:8080/metrics) | Replayed workload metrics         |
| **Prometheus**         | [http://localhost:9090](http://localhost:9090)                 | Metric collection and queries     |
| **Grafana**            | [http://localhost:3000](http://localhost:3000)                 | Dashboards (login: admin / admin) |
| **Alertmanager**       | [http://localhost:9093](http://localhost:9093)                 | Alert notifications               |
| **Controller Metrics** | [http://localhost:8082/metrics](http://localhost:8082/metrics) | Predictive scaling controller     |

---

## 📈 Watch Scaling Decisions in Real-Time

```bash
docker logs -f cloud1-scaling-controller-1 | egrep -E "predictive read|Initiating scaling|Scaling completed|Auto-scaling loop"
```

You’ll see logs like:

```
info: predictive read: mean=0.089 σ=0.003 decision=0.092 > threshold=0.09
info: Initiating scaling to 3 replicas
info: Scaling completed: 2 -> 3 replicas in 2s
```

---

## ⚖️ Manual Testing (Optional)

To simulate different conditions:

```bash
nano ./artifacts/predictive_signal.json
```

| Scenario       | Mean Value       | Expected Behavior   |
| -------------- | ---------------- | ------------------- |
| **Scale Up**   | `"mean": 0.095`  | Increases replicas  |
| **Scale Down** | `"mean": 0.070`  | Decreases replicas  |
| **No Scale**   | `"mean": 0.089"` | Keeps replicas same |

---

## 📊 Results & Evaluation

| Model             | MAE    | RMSE   | Comment                    |
| ----------------- | ------ | ------ | -------------------------- |
| **LSTM**          | 0.0040 | 0.0051 | Captures temporal patterns |
| **Random Forest** | 0.0002 | 0.0004 | Most accurate overall      |
| **ARIMA**         | 0.0046 | 0.0058 | Good baseline comparison   |

✅ Scaling Controller successfully reacted to predicted latency by increasing replicas (e.g., `2 → 3 → 4`) before SLA breach.

---

## 📸 Example Screenshots (add these)

🖼️ Prometheus Metrics Query
🖼️ Grafana Scaling Dashboard
🖼️ Controller Logs showing “Scaling completed: 2 → 3 replicas”

---

## 🧠 Future Enhancements

* Implement **Reinforcement Learning (RL)** for adaptive threshold tuning (dynamic α)
* Introduce **multi-metric scaling** (CPU, memory + latency)
* Add **cost-aware optimization** (replica vs SLA trade-off)
* Extend to **Kubernetes Horizontal Pod Autoscaler (HPA)** integration

---

## 🧹 Cleanup Commands

Stop all containers:

```bash
docker compose down
```

Remove everything (images, volumes, cache):

```bash
docker system prune -a --volumes
```

---

## 🏁 Conclusion

* Predictive scaling avoids performance degradation by forecasting workload.
* LSTM, RF, and ARIMA together ensure robust predictions.
* Prometheus + Grafana enable real-time observability.
* The system is generalizable and can integrate with modern DevOps stacks.

---

## 👩‍💻 Author

**Chaithra Jagannatha Rao Telkar**
📍 Master’s Thesis Project — 2025
🎓 Focus: Predictive Auto-Scaling in Cloud Computing
💡 Contact: [LinkedIn](#) | [GitHub](https://github.com/<your-username>)

```

---

Would you like me to generate a small **architecture diagram (image)** that fits this README section automatically (with clean labels like “Predictive Controller”, “Prometheus”, “Grafana”, etc.)?  
It’ll make your GitHub page visually stand out.
```


Excellent catch, Chaithra 🌸 — yes, before running Docker, you need to **run the Python training pipeline** once to generate the **`predictive_signal.json`** file that your scaling controller reads.

Here’s how to add that clearly to your `README.md` 👇
(I’ll show the section you can just paste.)

---

### 🧠 3️⃣ Run the Python Training Pipeline

Before starting Docker, you must train the predictive models and generate the **predictive signal file**.

Run this in your project root:

```bash
python3 vmcloud_pipeline_single.py --target response_time_p95 --epochs 30
```

✅ This command:

* Loads and preprocesses the Alibaba dataset (`data/` folder)
* Builds lag and rolling features
* Trains **LSTM**, **Random Forest**, and **ARIMA** models
* Creates the output file → `./artifacts/predictive_signal.json`

You should see output like:

```
[LSTM] saved: artifacts/lstm_model.keras
[RF] saved: artifacts/rf_model.pkl
[ARIMA] saved: artifacts/arima_model.pkl
📝 saved predictive signal -> artifacts/predictive_signal.json
```

---

Then, continue with:

```bash
docker compose up --build
```

---

If you want the exact README section ready to copy (with this added between step 2 and step 3 of your existing one), I can paste that version for you — want me to?
