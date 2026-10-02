"""
Observability & Prometheus Telemetry Exporter
Real-time metrics export for Prometheus and Grafana dashboards.
Tracks latency percentiles (p50, p95, p99), QPS throughput, bandit exploration ratios,
cache performance, and reward feedback rates.
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# 1. Decision Counters
DECISIONS_TOTAL = Counter(
    'ranking_decisions_total',
    'Total ranking decisions served',
    ['policy', 'model_type', 'status']
)

FEEDBACK_EVENTS_TOTAL = Counter(
    'ranking_feedback_events_total',
    'Total user interaction feedback events received',
    ['event_type']
)

# 2. Stage Latency Histograms
LATENCY_TOTAL = Histogram(
    'ranking_latency_total_seconds',
    'End-to-end multi-stage pipeline latency in seconds',
    buckets=[0.001, 0.002, 0.005, 0.010, 0.020, 0.050, 0.100, 0.250]
)

LATENCY_RETRIEVAL = Histogram(
    'ranking_latency_retrieval_seconds',
    'Stage 1 Candidate Retrieval latency in seconds',
    buckets=[0.0005, 0.001, 0.002, 0.005, 0.010]
)

LATENCY_ENRICHMENT = Histogram(
    'ranking_latency_enrichment_seconds',
    'Stage 2 Feature Enrichment latency in seconds',
    buckets=[0.0005, 0.001, 0.002, 0.005, 0.010]
)

LATENCY_RANKING = Histogram(
    'ranking_latency_gbdt_seconds',
    'Stage 3 GBDT Heavy Ranking latency in seconds',
    buckets=[0.001, 0.002, 0.005, 0.010, 0.020, 0.050]
)

LATENCY_BANDIT = Histogram(
    'ranking_latency_bandit_seconds',
    'Stage 4 Contextual Bandit scoring latency in seconds',
    buckets=[0.0005, 0.001, 0.002, 0.005, 0.010]
)

LATENCY_RERANKING = Histogram(
    'ranking_latency_reranking_seconds',
    'Stage 5 MMR Re-ranking latency in seconds',
    buckets=[0.0005, 0.001, 0.002, 0.005, 0.010]
)

# 3. Gauges for state & cache
CACHE_HIT_RATE = Gauge(
    'feature_store_cache_hit_rate',
    'Redis Feature Store hit rate percentage'
)

ACTIVE_BANDIT_ARMS = Gauge(
    'contextual_bandit_active_arms_count',
    'Number of active bandit arm models tracked'
)

CUMULATIVE_REWARD_GAUGE = Gauge(
    'contextual_bandit_cumulative_reward',
    'Cumulative online reward achieved by bandit policy'
)

class PrometheusMetricsManager:
    @staticmethod
    def record_decision(policy: str, model_type: str, latency_breakdown: dict):
        DECISIONS_TOTAL.labels(policy=policy, model_type=model_type, status="success").inc()
        
        total_s = latency_breakdown.get("total_pipeline_ms", 0.0) / 1000.0
        LATENCY_TOTAL.observe(total_s)
        
        LATENCY_RETRIEVAL.observe(latency_breakdown.get("candidate_retrieval_ms", 0.0) / 1000.0)
        LATENCY_ENRICHMENT.observe(latency_breakdown.get("feature_enrichment_ms", 0.0) / 1000.0)
        LATENCY_RANKING.observe(latency_breakdown.get("heavy_ranking_ms", 0.0) / 1000.0)
        LATENCY_BANDIT.observe(latency_breakdown.get("contextual_bandit_ms", 0.0) / 1000.0)
        LATENCY_RERANKING.observe(latency_breakdown.get("reranking_diversity_ms", 0.0) / 1000.0)

    @staticmethod
    def record_feedback(event_type: str):
        FEEDBACK_EVENTS_TOTAL.labels(event_type=event_type).inc()

    @staticmethod
    def update_system_gauges(hit_rate: float, arms_count: int, cum_reward: float):
        CACHE_HIT_RATE.set(hit_rate)
        ACTIVE_BANDIT_ARMS.set(arms_count)
        CUMULATIVE_REWARD_GAUGE.set(cum_reward)

    @staticmethod
    def export_metrics() -> str:
        return generate_latest().decode("utf-8")
