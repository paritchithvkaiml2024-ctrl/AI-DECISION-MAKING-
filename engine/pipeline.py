"""
Unified Multi-Stage Decision & Ranking Pipeline
Coordinates the complete decision lifecycle with microsecond telemetry:
Retrieval -> Feature Enrichment -> GBDT Heavy Ranker -> Contextual Bandits -> Re-ranking -> Analytics Logging
"""

import time
import uuid
import logging
from typing import Dict, Any, List, Optional

from storage.feature_store import RedisFeatureCache, ScyllaDBFeatureStore, ApachePinotEventLogger
from engine.candidate_generator import CandidateGenerator
from engine.feature_enricher import FeatureEnricher
from engine.ranker import GBDTRanker
from engine.contextual_bandits import ContextualBanditEngine
from engine.reranker import ReRanker

logger = logging.getLogger("DecisionCore.Pipeline")

class DecisionPipeline:
    """
    Automated Decision-Making Core Orchestrator
    """
    def __init__(
        self,
        cache: Optional[RedisFeatureCache] = None,
        feature_store: Optional[ScyllaDBFeatureStore] = None,
        event_logger: Optional[ApachePinotEventLogger] = None
    ):
        self.cache = cache or RedisFeatureCache()
        self.feature_store = feature_store or ScyllaDBFeatureStore()
        self.event_logger = event_logger or ApachePinotEventLogger()

        # Instantiate pipeline stages
        self.candidate_generator = CandidateGenerator()
        self.feature_enricher = FeatureEnricher(self.feature_store, self.cache)
        self.ranker = GBDTRanker(model_type="ensemble")
        self.bandit_engine = ContextualBanditEngine(feature_dim=9, default_alpha=0.75)
        self.reranker = ReRanker(diversity_lambda=0.20)
        
        self.start_time = time.time()
        self.decision_counter = 0

    def execute_decision(
        self,
        user_context: Dict[str, Any],
        candidates_override: Optional[List[Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end multi-stage ranking decision with high-precision stage profiling.
        """
        cfg = config or {}
        top_k = cfg.get("top_k", 6)
        policy = cfg.get("exploration_policy", "linucb")
        alpha = cfg.get("exploration_factor_alpha", 0.75)
        diversity_lambda = cfg.get("diversity_penalty_lambda", 0.20)
        model_type = cfg.get("ranker_model_type", "ensemble")
        
        req_id = str(uuid.uuid4())
        t0 = time.perf_counter()

        # Stage 1: Candidate Generation
        t_stage_0 = time.perf_counter()
        if candidates_override and len(candidates_override) > 0:
            candidates = candidates_override
        else:
            candidates = self.candidate_generator.retrieve_candidates(user_context, limit=20)
        t_retrieval_ms = round((time.perf_counter() - t_stage_0) * 1000, 3)

        # Stage 2: Feature Enrichment
        t_stage_1 = time.perf_counter()
        enriched_candidates = self.feature_enricher.enrich_batch(user_context, candidates)
        t_enrich_ms = round((time.perf_counter() - t_stage_1) * 1000, 3)

        # Stage 3: Heavy GBDT Ranking (XGBoost / LightGBM)
        t_stage_2 = time.perf_counter()
        scored_candidates = self.ranker.score_candidates(enriched_candidates, model_override=model_type)
        t_rank_ms = round((time.perf_counter() - t_stage_2) * 1000, 3)

        # Stage 4: Contextual Bandit Exploration-Exploitation
        t_stage_3 = time.perf_counter()
        bandit_candidates = self.bandit_engine.apply_exploration_policy(
            scored_candidates, 
            policy=policy, 
            alpha=alpha
        )
        t_bandit_ms = round((time.perf_counter() - t_stage_3) * 1000, 3)

        # Stage 5: Re-ranking & Diversity Optimization (MMR)
        t_stage_4 = time.perf_counter()
        final_ranked = self.reranker.rerank_mmr(
            bandit_candidates, 
            top_k=top_k, 
            diversity_lambda=diversity_lambda
        )
        t_rerank_ms = round((time.perf_counter() - t_stage_4) * 1000, 3)

        total_pipeline_ms = round((time.perf_counter() - t0) * 1000, 3)
        self.decision_counter += 1

        # Latency breakdown telemetry
        latency_breakdown = {
            "candidate_retrieval_ms": t_retrieval_ms,
            "feature_enrichment_ms": t_enrich_ms,
            "heavy_ranking_ms": t_rank_ms,
            "contextual_bandit_ms": t_bandit_ms,
            "reranking_diversity_ms": t_rerank_ms,
            "total_pipeline_ms": total_pipeline_ms
        }

        # Format items for clean response
        clean_ranked_items = []
        for item in final_ranked:
            clean_ranked_items.append({
                "item_id": item.get("item_id"),
                "title": item.get("title", item.get("item_id")),
                "category": item.get("category"),
                "price": item.get("price"),
                "rank_position": item.get("rank_position"),
                "final_score": item.get("final_score"),
                "ml_rank_score": item.get("ml_rank_score"),
                "bandit_exploration_bonus": item.get("bandit_bonus", 0.0),
                "diversity_penalty": item.get("diversity_penalty", 0.0),
                "was_explored": item.get("was_explored", False),
                "tags": item.get("tags", [])
            })

        response = {
            "request_id": req_id,
            "user_id": user_context.get("user_id", "anonymous"),
            "ranked_items": clean_ranked_items,
            "latency_breakdown": latency_breakdown,
            "decision_policy_used": policy,
            "model_version": f"gbdt-{model_type}-v2.4",
            "generated_at_ms": int(time.time() * 1000)
        }

        # Log decision event to Apache Pinot OLAP stream
        self.event_logger.log_decision_event(response)

        return response

    def ingest_feedback(
        self, 
        user_id: str, 
        item_id: str, 
        event_type: str, 
        reward_value: float,
        request_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Processes online feedback and updates Contextual Bandit parameters.
        """
        # Re-extract or estimate feature vector for this user/item pair
        ctx = request_context or {"user_id": user_id}
        cand = {"item_id": item_id, "category": "Electronics & Gaming", "price": 200.0, "item_features": {}}
        vec, _ = self.feature_enricher.extract_features(ctx, cand)
        
        self.bandit_engine.update_feedback(item_id, vec, reward_value)

        return {
            "success": True,
            "message": f"Updated arm {item_id} with reward {reward_value}",
            "updated_arm_cumulative_reward": self.bandit_engine.linucb_arms.get(item_id, None).total_reward if item_id in self.bandit_engine.linucb_arms else reward_value
        }

    def get_health_and_metrics(self) -> Dict[str, Any]:
        uptime = round(time.time() - self.start_time, 1)
        cache_stats = self.cache.get_stats()
        olap_stats = self.event_logger.query_aggregate_stats()
        bandit_stats = self.bandit_engine.get_policy_summary()

        return {
            "status": "SERVING",
            "uptime_seconds": uptime,
            "total_decisions_served": self.decision_counter,
            "cache": cache_stats,
            "analytics_olap": olap_stats,
            "bandit_telemetry": bandit_stats
        }
