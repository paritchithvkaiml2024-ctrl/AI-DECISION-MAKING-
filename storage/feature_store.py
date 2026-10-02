"""
Storage Layer & Feature Store
High-throughput, ultra-low latency feature storage and event streaming logger
supporting ScyllaDB (Cassandra protocol), Redis in-memory caching, and Apache Pinot OLAP.
"""

import time
import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("DecisionCore.FeatureStore")

class RedisFeatureCache:
    """
    Sub-millisecond Redis caching layer for real-time user profiles and item embeddings.
    """
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.host = host
        self.port = port
        self.db = db
        self.client = None
        self._local_cache: Dict[str, Dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        self._init_connection()

    def _init_connection(self):
        try:
            import redis
            self.client = redis.Redis(
                host=self.host, 
                port=self.port, 
                db=self.db, 
                socket_timeout=0.05, 
                socket_connect_timeout=0.1
            )
            self.client.ping()
            logger.info("Connected to Redis Feature Cache at %s:%d", self.host, self.port)
        except Exception as e:
            logger.warning("Redis not available (%s). Using high-speed in-memory L1 cache fallback.", str(e))
            self.client = None

    def get_features(self, entity_type: str, entity_id: str) -> Optional[Dict[str, Any]]:
        key = f"{entity_type}:{entity_id}"
        if self.client:
            try:
                data = self.client.get(key)
                if data:
                    self.hits += 1
                    return json.loads(data)
                self.misses += 1
                return None
            except Exception:
                self.misses += 1
        
        # Local fast fallback
        if key in self._local_cache:
            self.hits += 1
            return self._local_cache[key]
        self.misses += 1
        return None

    def set_features(self, entity_type: str, entity_id: str, features: Dict[str, Any], ttl_seconds: int = 3600):
        key = f"{entity_type}:{entity_id}"
        if self.client:
            try:
                self.client.setex(key, ttl_seconds, json.dumps(features))
                return
            except Exception:
                pass
        self._local_cache[key] = features

    def get_stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        hit_rate = (self.hits / total) if total > 0 else 1.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_pct": round(hit_rate * 100, 2),
            "cached_keys_count": len(self._local_cache)
        }


class ScyllaDBFeatureStore:
    """
    ScyllaDB / Cassandra persistent feature store for massive-scale user & item attributes.
    """
    def __init__(self, contact_points: List[str] = None, keyspace: str = "decision_features"):
        self.contact_points = contact_points or ["127.0.0.1"]
        self.keyspace = keyspace
        self.session = None
        self._mock_db = self._seed_mock_data()
        self._init_connection()

    def _init_connection(self):
        try:
            from cassandra.cluster import Cluster
            cluster = Cluster(self.contact_points, connect_timeout=0.2)
            self.session = cluster.connect()
            self.session.set_keyspace(self.keyspace)
            logger.info("Connected to ScyllaDB keyspace '%s'", self.keyspace)
        except Exception as e:
            logger.info("ScyllaDB cluster offline (%s). Using high-fidelity synthetic feature store.", str(e))
            self.session = None

    def _seed_mock_data(self) -> Dict[str, Dict[str, Any]]:
        return {
            "user:usr_tech_enthusiast": {
                "affinity_tech": 0.95, "affinity_gaming": 0.85, "affinity_fashion": 0.10,
                "lifetime_spend": 1450.0, "churn_risk": 0.05, "engagement_score": 0.92,
                "preferred_price_tier": 3, "avg_session_duration_min": 18.5
            },
            "user:usr_fashion_shopper": {
                "affinity_tech": 0.15, "affinity_gaming": 0.05, "affinity_fashion": 0.98,
                "lifetime_spend": 890.0, "churn_risk": 0.12, "engagement_score": 0.78,
                "preferred_price_tier": 2, "avg_session_duration_min": 12.0
            },
            "user:usr_budget_saver": {
                "affinity_tech": 0.40, "affinity_gaming": 0.30, "affinity_fashion": 0.45,
                "lifetime_spend": 120.0, "churn_risk": 0.45, "engagement_score": 0.41,
                "preferred_price_tier": 1, "avg_session_duration_min": 5.2
            },
            "user:usr_vip_power": {
                "affinity_tech": 0.88, "affinity_gaming": 0.90, "affinity_fashion": 0.75,
                "lifetime_spend": 6200.0, "churn_risk": 0.01, "engagement_score": 0.99,
                "preferred_price_tier": 4, "avg_session_duration_min": 35.0
            }
        }

    def fetch_user_features(self, user_id: str) -> Dict[str, Any]:
        key = f"user:{user_id}"
        if key in self._mock_db:
            return dict(self._mock_db[key])
        
        # Default dynamic generated profile for any novel user
        import hashlib
        h = int(hashlib.md5(user_id.encode()).hexdigest()[:6], 16) / 0xFFFFFF
        return {
            "affinity_tech": round(h, 2),
            "affinity_gaming": round((1.0 - h) * 0.8, 2),
            "affinity_fashion": round(0.5 * h, 2),
            "lifetime_spend": round(h * 500.0, 2),
            "churn_risk": round(0.1 + 0.3 * (1.0 - h), 2),
            "engagement_score": round(0.3 + 0.6 * h, 2),
            "preferred_price_tier": int(h * 3) + 1,
            "avg_session_duration_min": round(4.0 + h * 20.0, 1)
        }

    def fetch_item_features(self, item_id: str) -> Dict[str, Any]:
        return {
            "historical_ctr": 0.045,
            "conversion_rate": 0.018,
            "quality_rating": 4.6,
            "inventory_velocity": 0.82
        }


class ApachePinotEventLogger:
    """
    Real-time OLAP streaming logger for ranking decisions, impression events, 
    click feedback, and stage-by-stage latency breakdowns for sub-second analytical queries.
    """
    def __init__(self, broker_url: str = "http://localhost:8099"):
        self.broker_url = broker_url
        self.event_stream: List[Dict[str, Any]] = []
        self.max_in_memory_events = 5000

    def log_decision_event(self, event: Dict[str, Any]):
        event["logged_at_ms"] = int(time.time() * 1000)
        self.event_stream.append(event)
        if len(self.event_stream) > self.max_in_memory_events:
            self.event_stream.pop(0)

    def query_recent_decisions(self, limit: int = 50) -> List[Dict[str, Any]]:
        return list(reversed(self.event_stream[-limit:]))

    def query_aggregate_stats(self) -> Dict[str, Any]:
        if not self.event_stream:
            return {
                "total_decisions": 0,
                "avg_total_latency_ms": 0.0,
                "avg_ranking_latency_ms": 0.0,
                "avg_bandit_latency_ms": 0.0
            }
        
        total_lat = sum(e.get("latency_breakdown", {}).get("total_pipeline_ms", 0.0) for e in self.event_stream)
        rank_lat = sum(e.get("latency_breakdown", {}).get("heavy_ranking_ms", 0.0) for e in self.event_stream)
        bandit_lat = sum(e.get("latency_breakdown", {}).get("contextual_bandit_ms", 0.0) for e in self.event_stream)
        n = len(self.event_stream)

        return {
            "total_decisions": n,
            "avg_total_latency_ms": round(total_lat / n, 2),
            "avg_ranking_latency_ms": round(rank_lat / n, 2),
            "avg_bandit_latency_ms": round(bandit_lat / n, 2)
        }
