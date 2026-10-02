"""
FastAPI REST & WebSocket Gateway for Decision-Making Core
Serves ranking endpoints, contextual bandit simulation, Prometheus metrics, and UI dashboard.
"""

import os
import time
import numpy as np
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, PlainTextResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from engine.pipeline import DecisionPipeline
from serving.ray_executor import RayServingCluster
from observability.metrics import PrometheusMetricsManager

# Initialize FastAPI App
app = FastAPI(
    title="Automated Decision-Making Core API",
    description="High-throughput AI Ranking & Personalization Engine with Contextual Bandits, GBDT models, and Ray distributed serving.",
    version="2.4.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Core Pipeline & Cluster Singletons
pipeline = DecisionPipeline()
ray_cluster = RayServingCluster(num_workers=4)

# Static Dashboard Directory
WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
if os.path.exists(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


# Pydantic Request Models
class UserContextModel(BaseModel):
    user_id: str = "usr_tech_enthusiast"
    session_id: Optional[str] = "sess_9921"
    device: Optional[str] = "desktop"
    geo_location: Optional[str] = "US-West"
    past_interacted_categories: Optional[List[str]] = ["Electronics & Gaming"]
    dynamic_features: Optional[Dict[str, float]] = None

class RankingConfigModel(BaseModel):
    top_k: int = 6
    exploration_policy: str = "linucb" # "linucb" | "thompson_sampling" | "epsilon_greedy" | "greedy"
    exploration_factor_alpha: float = 0.75
    diversity_penalty_lambda: float = 0.20
    ranker_model_type: str = "ensemble" # "xgboost" | "lightgbm" | "ensemble"

class RankRequestModel(BaseModel):
    user_context: UserContextModel
    config: Optional[RankingConfigModel] = None
    candidates: Optional[List[Dict[str, Any]]] = None

class FeedbackModel(BaseModel):
    user_id: str
    item_id: str
    event_type: str = "click" # "click", "conversion", "impression"
    reward_value: float = 1.0


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    index_file = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return "<h1>Automated Decision-Making Core API Running</h1><p>Visit /docs for API documentation.</p>"


@app.post("/api/v1/rank")
async def rank_items(req: RankRequestModel):
    user_dict = req.user_context.dict()
    config_dict = req.config.dict() if req.config else {}
    candidates_list = req.candidates
    
    result = pipeline.execute_decision(
        user_context=user_dict,
        candidates_override=candidates_list,
        config=config_dict
    )
    
    # Track Prometheus telemetry
    PrometheusMetricsManager.record_decision(
        policy=config_dict.get("exploration_policy", "linucb"),
        model_type=config_dict.get("ranker_model_type", "ensemble"),
        latency_breakdown=result["latency_breakdown"]
    )
    
    return result


@app.post("/api/v1/feedback")
async def record_feedback(fb: FeedbackModel):
    res = pipeline.ingest_feedback(
        user_id=fb.user_id,
        item_id=fb.item_id,
        event_type=fb.event_type,
        reward_value=fb.reward_value
    )
    PrometheusMetricsManager.record_feedback(event_type=fb.event_type)
    return res


@app.get("/api/v1/health")
async def health_check():
    health = pipeline.get_health_and_metrics()
    health["ray_cluster"] = ray_cluster.get_cluster_status()
    
    # Update Prometheus gauges
    cache_hit_rate = health.get("cache", {}).get("hit_rate_pct", 100.0)
    arms_count = health.get("bandit_telemetry", {}).get("linucb_arms_tracked", 0)
    total_reward = sum(a["total_reward"] for a in health.get("bandit_telemetry", {}).get("arms_summary", []))
    PrometheusMetricsManager.update_system_gauges(cache_hit_rate, arms_count, total_reward)
    
    return health


@app.get("/api/v1/catalog")
async def get_catalog():
    return {
        "catalog_size": len(pipeline.candidate_generator.catalog),
        "items": pipeline.candidate_generator.catalog
    }


@app.get("/api/v1/bandit-stats")
async def get_bandit_stats():
    return pipeline.bandit_engine.get_policy_summary()


@app.post("/api/v1/simulate")
async def run_bandit_simulation(steps: int = 100):
    """
    Monte-Carlo A/B simulation comparing Random, Epsilon-Greedy, LinUCB, and Thompson Sampling.
    """
    steps = min(max(steps, 20), 500)
    
    # Ground truth hidden conversion probabilities for 4 item categories
    true_ctrs = {
        "item_ultra_oled_4k": 0.18,
        "item_wireless_noise_cancelling_anc": 0.15,
        "item_italian_leather_jacket": 0.08,
        "item_titanium_smartwatch_ultra": 0.22,
        "item_smart_ergonomic_chair": 0.06
    }
    
    items = list(true_ctrs.keys())
    
    def simulate_policy(policy_name: str):
        cum_reward = []
        regrets = []
        optimal_ctr = max(true_ctrs.values())
        
        # Local bandit trackers
        counts = {item: 0 for item in items}
        rewards = {item: 0.0 for item in items}
        
        running_total_reward = 0.0
        running_regret = 0.0
        
        for t in range(1, steps + 1):
            if policy_name == "random":
                chosen = np.random.choice(items)
            elif policy_name == "epsilon_greedy":
                if np.random.rand() < 0.20 or t < 5:
                    chosen = np.random.choice(items)
                else:
                    chosen = max(items, key=lambda it: rewards[it] / max(1, counts[it]))
            elif policy_name == "linucb":
                # UCB score
                scores = {}
                for it in items:
                    n_a = counts[it]
                    mean = rewards[it] / max(1, n_a)
                    bonus = 0.8 * np.sqrt(np.log(t + 1) / max(1, n_a)) if n_a > 0 else 2.0
                    scores[it] = mean + bonus
                chosen = max(items, key=lambda it: scores[it])
            elif policy_name == "thompson_sampling":
                scores = {}
                for it in items:
                    a = 1 + rewards[it]
                    b = 1 + (counts[it] - rewards[it])
                    scores[it] = np.random.beta(a, max(1, b))
                chosen = max(items, key=lambda it: scores[it])
            else:
                chosen = np.random.choice(items)
            
            # Simulate Bernoulli reward
            p_win = true_ctrs[chosen]
            reward = 1.0 if np.random.rand() < p_win else 0.0
            
            counts[chosen] += 1
            rewards[chosen] += reward
            running_total_reward += reward
            running_regret += (optimal_ctr - p_win)
            
            cum_reward.append(round(running_total_reward, 1))
            regrets.append(round(running_regret, 2))
            
        return {
            "policy": policy_name,
            "cumulative_rewards": cum_reward,
            "cumulative_regrets": regrets,
            "total_reward": running_total_reward,
            "total_regret": round(running_regret, 2),
            "final_arm_distribution": counts
        }

    return {
        "steps": steps,
        "results": {
            "random": simulate_policy("random"),
            "epsilon_greedy": simulate_policy("epsilon_greedy"),
            "linucb": simulate_policy("linucb"),
            "thompson_sampling": simulate_policy("thompson_sampling")
        }
    }


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    return PrometheusMetricsManager.export_metrics()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serving.api_server:app", host="0.0.0.0", port=8000, reload=False)
