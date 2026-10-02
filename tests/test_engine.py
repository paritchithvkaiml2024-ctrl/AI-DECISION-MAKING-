"""
Unit & Integration Test Suite for Automated Decision-Making Core
"""

import sys
import os
import pytest
import numpy as np

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.pipeline import DecisionPipeline
from engine.candidate_generator import CandidateGenerator
from engine.feature_enricher import FeatureEnricher
from engine.ranker import GBDTRanker
from engine.contextual_bandits import ContextualBanditEngine, LinUCBArm
from engine.reranker import ReRanker
from storage.feature_store import RedisFeatureCache, ScyllaDBFeatureStore

@pytest.fixture
def pipeline():
    return DecisionPipeline()

def test_candidate_retrieval():
    generator = CandidateGenerator()
    user_ctx = {"user_id": "usr_tech_enthusiast", "past_interacted_categories": ["Electronics & Gaming"]}
    candidates = generator.retrieve_candidates(user_ctx, limit=5)
    assert len(candidates) == 5
    assert all("item_id" in c for c in candidates)
    assert all("retrieval_score" in c for c in candidates)

def test_feature_enrichment():
    enricher = FeatureEnricher()
    user_ctx = {"user_id": "usr_fashion_shopper"}
    cand = {"item_id": "item_jacket", "category": "Apparel & Fashion", "price": 200.0, "base_score": 0.8}
    vec, feats = enricher.extract_features(user_ctx, cand)
    
    assert isinstance(vec, np.ndarray)
    assert len(vec) == 9
    assert feats["category_affinity"] > 0.5

def test_gbdt_ranker():
    ranker = GBDTRanker(model_type="ensemble")
    candidates = [
        {"item_id": "item_1", "feature_vector": np.array([0.9, 0.8, 0.95, 0.8, 0.9, 0.4, 0.5, 0.08, 0.9])},
        {"item_id": "item_2", "feature_vector": np.array([0.1, 0.1, 0.05, 0.2, 0.3, 0.1, 0.1, 0.01, 0.2])}
    ]
    scored = ranker.score_candidates(candidates)
    assert len(scored) == 2
    assert scored[0]["ml_rank_score"] >= scored[1]["ml_rank_score"]

def test_linucb_arm_update():
    arm = LinUCBArm("test_arm", feature_dim=9, alpha=1.0)
    vec = np.ones(9, dtype=np.float32)
    
    ucb_before, mean_before, bonus_before = arm.compute_ucb_score(vec)
    arm.update(vec, reward=5.0)
    ucb_after, mean_after, bonus_after = arm.compute_ucb_score(vec)
    
    assert arm.pull_count == 1
    assert arm.total_reward == 5.0
    assert mean_after > mean_before
    # Uncertainty/bonus should decrease as confidence grows
    assert bonus_after < bonus_before

def test_reranker_mmr():
    reranker = ReRanker(diversity_lambda=0.3)
    candidates = [
        {"item_id": "item_a1", "category": "Tech", "bandit_score": 0.95, "feature_vector": np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])},
        {"item_id": "item_a2", "category": "Tech", "bandit_score": 0.94, "feature_vector": np.array([1, 1, 1, 1, 1, 1, 1, 1, 0.99])},
        {"item_id": "item_b1", "category": "Fashion", "bandit_score": 0.88, "feature_vector": np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])}
    ]
    ranked = reranker.rerank_mmr(candidates, top_k=3)
    assert len(ranked) == 3
    assert ranked[0]["item_id"] == "item_a1"
    # Due to MMR diversity penalty on duplicate category, item_b1 gets prioritized over duplicate item_a2
    assert ranked[1]["item_id"] == "item_b1"

def test_end_to_end_decision_pipeline(pipeline):
    user_ctx = {
        "user_id": "usr_vip_power",
        "past_interacted_categories": ["Electronics & Gaming", "Wearables"]
    }
    result = pipeline.execute_decision(
        user_context=user_ctx,
        config={"top_k": 4, "exploration_policy": "linucb"}
    )
    
    assert "request_id" in result
    assert len(result["ranked_items"]) == 4
    assert "latency_breakdown" in result
    assert result["latency_breakdown"]["total_pipeline_ms"] > 0
    assert result["latency_breakdown"]["total_pipeline_ms"] < 200

def test_feedback_loop(pipeline):
    res = pipeline.ingest_feedback(
        user_id="usr_tech_enthusiast",
        item_id="item_ultra_oled_4k",
        event_type="click",
        reward_value=1.0
    )
    assert res["success"] is True
