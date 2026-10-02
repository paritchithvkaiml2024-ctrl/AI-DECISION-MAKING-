# ⚡ Automated Decision-Making Core — AI Ranking Engine

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![Ray](https://img.shields.io/badge/Distributed-Ray%20Serve-028CF0.svg)](https://ray.io)
[![XGBoost](https://img.shields.io/badge/GBDT-XGBoost%20%2B%20LightGBM-EB5424.svg)](https://xgboost.readthedocs.io)
[![gRPC](https://img.shields.io/badge/RPC-gRPC%20%2B%20Protobuf-244f5c.svg)](https://grpc.io)
[![Storage](https://img.shields.io/badge/Store-ScyllaDB%20%2B%20Redis%20%2B%20Pinot-DC382D.svg)](https://www.scylladb.com)
[![Observability](https://img.shields.io/badge/Metrics-Prometheus%20%2B%20Grafana-E6522C.svg)](https://prometheus.io)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

An industrial-grade, ultra-low-latency **Automated Decision-Making and AI Personalization Core**. Engineered for multi-stage candidate retrieval, feature crossing, heavy GBDT ranking, exploration–exploitation contextual bandits (LinUCB, Thompson Sampling), and diversity-aware re-ranking (MMR).

---

## 🏛️ System Architecture

```
                                  [ User Request / Context ]
                                               │
                                               ▼
               ┌───────────────────────────────────────────────────────────────┐
               │              Stage 1: Candidate Generation Layer              │
               │      Catalog Retrieval • Two-Tower Filtering • ANN Search     │
               └───────────────────────────────┬───────────────────────────────┘
                                               │
                                               ▼
               ┌───────────────────────────────────────────────────────────────┐
               │             Stage 2: Real-Time Feature Store Layer            │
               │   ScyllaDB User Profiles • Redis L1 Cache • Dynamic Crosses   │
               └───────────────────────────────┬───────────────────────────────┘
                                               │
                                               ▼
               ┌───────────────────────────────────────────────────────────────┐
               │           Stage 3: Heavy GBDT Ranking (XGBoost / LightGBM)    │
               │   Calibrated pCTR Prediction • Pairwise/Listwise Relevancy    │
               └───────────────────────────────┬───────────────────────────────┘
                                               │
                                               ▼
               ┌───────────────────────────────────────────────────────────────┐
               │     Stage 4: Exploration–Exploitation (Contextual Bandits)    │
               │       LinUCB (Ridge Regression) • Thompson Sampling (Bayes)   │
               └───────────────────────────────┬───────────────────────────────┘
                                               │
                                               ▼
               ┌───────────────────────────────────────────────────────────────┐
               │            Stage 5: Re-Ranking & Diversity Engine             │
               │       Maximal Marginal Relevance (MMR) • Business Rules       │
               └───────────────────────────────┬───────────────────────────────┘
                                               │
                                               ▼
                                 [ Final Ranked Decisions ]
                                               │
                         ┌─────────────────────┴─────────────────────┐
                         ▼                                           ▼
          [ gRPC / REST Response ]                      [ Apache Pinot OLAP Logger ]
          (P99 < 15ms Latency)                          (Streaming Event Stream)
```

---

## 🚀 Key Highlights & Capabilities

- **5-Stage Decision Pipeline**:
  1. **Candidate Retrieval**: Slices top-N candidates from catalog based on high-level heuristics and historical engagement.
  2. **Feature Enrichment**: Ultra-low-latency feature joining from ScyllaDB and Redis with memory fallback.
  3. **Heavy ML Ranker**: Ensemble scoring using calibrated **XGBoost** and **LightGBM** ranking models.
  4. **Contextual Bandits**: Online exploration-exploitation preventing popularity bias with **LinUCB** and **Thompson Sampling**.
  5. **Re-Ranking**: **Maximal Marginal Relevance (MMR)** and frequency capping to avoid item and category monotony.
- **Distributed Inference**: Parallel actor workers via **Ray Serve** and distributed **gRPC / Protocol Buffers** interfaces.
- **Sub-Millisecond Feature Cache**: Multi-tier storage with **Redis L1 cache** and **ScyllaDB NoSQL store**.
- **Real-Time OLAP & Feedback Loop**: **Apache Pinot** streaming logger and online bandit parameter updating on user click/buy signals.
- **Full Production Observability**: Embedded **Prometheus metrics** exporter and pre-configured **Grafana** dashboards.
- **Interactive UI Dashboard**: Live visualizer with microsecond latency waterfall, persona selector, and Monte Carlo Bandit A/B simulator.

---

## 📁 Repository Structure

```
.
├── benchmarks/
│   └── latency_benchmark.py       # P50, P90, P95, P99 latency benchmark tool
├── engine/
│   ├── candidate_generator.py     # Stage 1: Candidate generation & catalog retrieval
│   ├── feature_enricher.py        # Stage 2: Feature vector extraction & crosses
│   ├── ranker.py                  # Stage 3: XGBoost & LightGBM heavy rankers
│   ├── contextual_bandits.py      # Stage 4: LinUCB, Thompson Sampling, ε-Greedy
│   ├── reranker.py                # Stage 5: MMR diversity & business constraints
│   └── pipeline.py                # Unified 5-stage pipeline orchestrator
├── observability/
│   ├── metrics.py                 # Prometheus latency histograms & QPS counters
│   └── prometheus.yml             # Prometheus scrape configuration
├── proto/
│   └── ranking_service.proto      # Protobuf gRPC service schema
├── serving/
│   ├── api_server.py              # FastAPI REST & WebSocket gateway
│   ├── grpc_server.py             # High-throughput gRPC servant
│   └── ray_executor.py            # Ray distributed actor pool
├── storage/
│   └── feature_store.py           # ScyllaDB, Redis & Apache Pinot adapters
├── tests/
│   └── test_engine.py             # Pytest unit & integration test suite
├── web/
│   ├── index.html                 # Modern glassmorphic Web Dashboard
│   ├── style.css                  # Dark-mode styling and animations
│   └── app.js                     # Frontend interactive controller & Chart.js
├── docker-compose.yml             # Docker stack for Redis, ScyllaDB, Prometheus, Grafana
├── Dockerfile                     # Container image definition
└── requirements.txt               # Python package dependencies
```

---

## 🛠️ Quickstart Guide

### 1. Local Environment Setup

```bash
# Clone the repository
git clone https://github.com/paritchithvkaiml2024-ctrl/AI-DECISION-MAKING-.git
cd AI-DECISION-MAKING-

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Decision Core API & Dashboard

```bash
python -m uvicorn serving.api_server:app --host 0.0.0.0 --port 8000
```
- 🌐 **Interactive Dashboard**: Open [http://localhost:8000](http://localhost:8000)
- 📖 **FastAPI Swagger Docs**: Open [http://localhost:8000/docs](http://localhost:8000/docs)
- 📊 **Prometheus Metrics**: Open [http://localhost:8000/metrics](http://localhost:8000/metrics)

### 3. Run Latency Benchmarks

```bash
python benchmarks/latency_benchmark.py
```

### 4. Run Unit & Integration Tests

```bash
pytest tests/ -v
```

---

## 🐳 Docker Compose Stack

Run the full distributed infrastructure (FastAPI Core, Redis, ScyllaDB, Prometheus, Grafana) with a single command:

```bash
docker-compose up --build -d
```

| Service | Port | Description |
|---|---|---|
| **Decision Core API** | `8000` | REST API & Dashboard |
| **gRPC RPC Server** | `50051` | High-throughput RPC |
| **Redis Cache** | `6379` | L1 Sub-millisecond Feature Cache |
| **ScyllaDB** | `9042` | High-Throughput Feature Store |
| **Prometheus** | `9090` | Real-time Metrics Scraper |
| **Grafana** | `3000` | Observability Dashboards |

---

## 📡 REST API Specifications

### 1. Multi-Stage Ranking (`POST /api/v1/rank`)
```json
{
  "user_context": {
    "user_id": "usr_tech_enthusiast",
    "past_interacted_categories": ["Electronics & Gaming"]
  },
  "config": {
    "top_k": 6,
    "exploration_policy": "linucb",
    "exploration_factor_alpha": 0.75,
    "diversity_penalty_lambda": 0.20,
    "ranker_model_type": "ensemble"
  }
}
```

### 2. Ingest Feedback & Reinforce Bandit (`POST /api/v1/feedback`)
```json
{
  "user_id": "usr_tech_enthusiast",
  "item_id": "item_ultra_oled_4k",
  "event_type": "conversion",
  "reward_value": 5.0
}
```

### 3. Monte Carlo Bandit Simulation (`POST /api/v1/simulate?steps=100`)
Executes comparative simulation across **Random**, **ε-Greedy**, **LinUCB**, and **Thompson Sampling**, returning cumulative reward and regret vectors.

---

## 📄 License
Distributed under the Apache 2.0 License. See `LICENSE` for more information.
