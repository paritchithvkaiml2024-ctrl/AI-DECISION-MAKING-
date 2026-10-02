"""
Real-Time Feature Enrichment Engine
Joins online user context, cached ScyllaDB/Redis profiles, and candidate item metadata
into dense numerical feature vectors for GBDT ranking models and Contextual Bandits.
"""

from typing import Dict, Any, List, Tuple
import numpy as np

class FeatureEnricher:
    """
    Stage 2: Real-time Feature Enrichment
    Constructs normalized interaction feature vectors combining:
    - User dynamic features (affinities, lifetime spend, engagement)
    - Item features (CTR prior, margin, novelty, popularity)
    - Contextual crosses (category alignment, price sensitivity vs item price)
    """
    def __init__(self, feature_store=None, cache=None):
        self.feature_store = feature_store
        self.cache = cache

    def extract_features(
        self, 
        user_context: Dict[str, Any], 
        candidate: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        user_id = user_context.get("user_id", "anonymous")
        
        # 1. Fetch user features from cache or feature store
        user_feats = {}
        if self.cache:
            user_feats = self.cache.get_features("user", user_id) or {}
        if not user_feats and self.feature_store:
            user_feats = self.feature_store.fetch_user_features(user_id)
            if self.cache and user_feats:
                self.cache.set_features("user", user_id, user_feats)
        
        # Merge with any dynamic session features passed directly in the request
        req_dyn = user_context.get("dynamic_features", {})
        user_feats.update(req_dyn)

        # 2. Extract item features
        item_feats = candidate.get("item_features", {})
        price = candidate.get("price", 100.0)
        base_score = candidate.get("base_score", 0.5)
        category = candidate.get("category", "")
        
        # 3. Compute domain crosses
        aff_tech = user_feats.get("affinity_tech", 0.5)
        aff_fashion = user_feats.get("affinity_fashion", 0.5)
        aff_gaming = user_feats.get("affinity_gaming", 0.5)
        
        cat_affinity = 0.5
        if "Gaming" in category or "Electronics" in category:
            cat_affinity = 0.6 * aff_tech + 0.4 * aff_gaming
        elif "Fashion" in category or "Apparel" in category:
            cat_affinity = aff_fashion
        elif "Smart" in category or "Wearables" in category:
            cat_affinity = 0.7 * aff_tech + 0.3 * aff_fashion

        price_tier = user_feats.get("preferred_price_tier", 2)
        price_norm = min(price / 1000.0, 1.0)
        # Price match penalty/affinity
        expected_price = price_tier * 250.0
        price_alignment = max(0.0, 1.0 - abs(price - expected_price) / 1000.0)

        feature_dict = {
            "user_engagement": float(user_feats.get("engagement_score", 0.5)),
            "user_lifetime_spend_norm": min(float(user_feats.get("lifetime_spend", 100.0)) / 2000.0, 1.0),
            "category_affinity": float(cat_affinity),
            "price_alignment": float(price_alignment),
            "item_popularity": float(item_feats.get("popularity", 0.5)),
            "item_margin": float(item_feats.get("margin", 0.3)),
            "item_novelty": float(item_feats.get("novelty", 0.5)),
            "item_ctr_prior": float(item_feats.get("ctr_prior", 0.05)),
            "base_score": float(base_score)
        }

        # Fixed-dimension feature vector for model input
        vec = np.array([
            feature_dict["user_engagement"],
            feature_dict["user_lifetime_spend_norm"],
            feature_dict["category_affinity"],
            feature_dict["price_alignment"],
            feature_dict["item_popularity"],
            feature_dict["item_margin"],
            feature_dict["item_novelty"],
            feature_dict["item_ctr_prior"],
            feature_dict["base_score"]
        ], dtype=np.float32)

        return vec, feature_dict

    def enrich_batch(
        self, 
        user_context: Dict[str, Any], 
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        enriched = []
        for cand in candidates:
            vec, feats = self.extract_features(user_context, cand)
            c = dict(cand)
            c["feature_vector"] = vec
            c["enriched_features"] = feats
            enriched.append(c)
        return enriched
