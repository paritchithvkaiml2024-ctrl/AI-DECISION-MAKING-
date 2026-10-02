"""
Ultra-Low Latency & High-Throughput Benchmarking Tool
Profiles Stage 1-5 latency percentiles (P50, P90, P95, P99) and queries per second (QPS).
"""

import sys
import os
import time
import numpy as np

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.pipeline import DecisionPipeline

def run_benchmark(num_iterations: int = 500):
    print("=" * 70)
    print(f"[BENCHMARK] Initializing AI Decision-Making Core Benchmark ({num_iterations} iterations)...")
    print("=" * 70)

    pipeline = DecisionPipeline()
    personas = ["usr_tech_enthusiast", "usr_fashion_shopper", "usr_budget_saver", "usr_vip_power"]
    policies = ["linucb", "thompson_sampling", "epsilon_greedy"]

    # Warmup
    for _ in range(20):
        pipeline.execute_decision({"user_id": "usr_vip_power"})

    total_latencies = []
    stage_latencies = {
        "candidate_retrieval_ms": [],
        "feature_enrichment_ms": [],
        "heavy_ranking_ms": [],
        "contextual_bandit_ms": [],
        "reranking_diversity_ms": []
    }

    start_bench = time.perf_counter()
    for i in range(num_iterations):
        uid = personas[i % len(personas)]
        pol = policies[i % len(policies)]
        
        t0 = time.perf_counter()
        resp = pipeline.execute_decision(
            user_context={"user_id": uid, "past_interacted_categories": ["Electronics & Gaming"]},
            config={"top_k": 6, "exploration_policy": pol}
        )
        t_el = (time.perf_counter() - t0) * 1000.0
        total_latencies.append(t_el)

        bk = resp["latency_breakdown"]
        for k in stage_latencies:
            stage_latencies[k].append(bk[k])

    duration_sec = time.perf_counter() - start_bench
    qps = round(num_iterations / duration_sec, 1)

    p50 = np.percentile(total_latencies, 50)
    p90 = np.percentile(total_latencies, 90)
    p95 = np.percentile(total_latencies, 95)
    p99 = np.percentile(total_latencies, 99)
    avg = np.mean(total_latencies)

    print("\n[STATS] BENCHMARK RESULTS:")
    print("-" * 50)
    print(f"Total Requests Served : {num_iterations}")
    print(f"Benchmark Duration    : {duration_sec:.2f} seconds")
    print(f"Throughput (QPS)      : {qps} requests/sec")
    print("-" * 50)
    print("END-TO-END PIPELINE LATENCY PROFILE:")
    print(f"  * Average Latency   : {avg:.3f} ms")
    print(f"  * P50 (Median)      : {p50:.3f} ms")
    print(f"  * P90 Latency       : {p90:.3f} ms")
    print(f"  * P95 Latency       : {p95:.3f} ms")
    print(f"  * P99 Latency       : {p99:.3f} ms")
    print("-" * 50)
    print("PER-STAGE AVERAGE LATENCY BREAKDOWN:")
    for stage, lats in stage_latencies.items():
        print(f"  * {stage.replace('_', ' ').title():<26} : {np.mean(lats):.3f} ms")
    print("=" * 70)

if __name__ == "__main__":
    run_benchmark(500)
