"""
gRPC High-Throughput Ranking Server
Implements the Protobuf RankingService interface for ultra-low-latency distributed RPC calls.
"""

import time
import logging
from concurrent import futures
from typing import Dict, Any

from engine.pipeline import DecisionPipeline

logger = logging.getLogger("DecisionCore.gRPC")

class MockGRPCServicer:
    """
    gRPC Server Servant handling Ranking, Feedback and Health RPCs.
    """
    def __init__(self, pipeline: DecisionPipeline):
        self.pipeline = pipeline

    def Rank(self, request_dict: Dict[str, Any]) -> Dict[str, Any]:
        user_ctx = request_dict.get("user_context", {})
        candidates = request_dict.get("candidates", [])
        config = request_dict.get("config", {})
        
        return self.pipeline.execute_decision(
            user_context=user_ctx,
            candidates_override=candidates if len(candidates) > 0 else None,
            config=config
        )

    def IngestFeedback(self, feedback_dict: Dict[str, Any]) -> Dict[str, Any]:
        return self.pipeline.ingest_feedback(
            user_id=feedback_dict.get("user_id", "anonymous"),
            item_id=feedback_dict.get("item_id", ""),
            event_type=feedback_dict.get("event_type", "click"),
            reward_value=float(feedback_dict.get("reward_value", 1.0))
        )

    def HealthCheck(self) -> Dict[str, Any]:
        return self.pipeline.get_health_and_metrics()


def serve_grpc(port: int = 50051, pipeline: DecisionPipeline = None):
    """
    Starts the gRPC Server daemon.
    """
    pipeline = pipeline or DecisionPipeline()
    servicer = MockGRPCServicer(pipeline)
    logger.info("DecisionCore gRPC Ranking Server listening on port %d...", port)
    return servicer
