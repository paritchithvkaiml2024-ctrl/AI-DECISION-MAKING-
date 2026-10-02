"""
Candidate Generation & Retrieval Layer
Fast retrieval of top candidate items from an inventory catalog of thousands/millions of items.
Uses category indexing, tag matching, and base popularity filtering.
"""

from typing import List, Dict, Any, Optional

DEFAULT_CATALOG: List[Dict[str, Any]] = [
    {
        "item_id": "item_ultra_oled_4k",
        "title": "Ultra OLED 4K Gaming Monitor 144Hz",
        "category": "Electronics & Gaming",
        "base_score": 0.88,
        "price": 699.99,
        "tags": ["gaming", "tech", "4k", "high_refresh"],
        "item_features": {"popularity": 0.94, "margin": 0.28, "novelty": 0.60, "ctr_prior": 0.082}
    },
    {
        "item_id": "item_mech_keyboard_pro",
        "title": "Tactile Mechanical Keyboard RGB",
        "category": "Electronics & Gaming",
        "base_score": 0.79,
        "price": 149.50,
        "tags": ["gaming", "tech", "accessories"],
        "item_features": {"popularity": 0.86, "margin": 0.45, "novelty": 0.35, "ctr_prior": 0.065}
    },
    {
        "item_id": "item_wireless_noise_cancelling_anc",
        "title": "Studio Pro ANC Wireless Headphones",
        "category": "Electronics & Gaming",
        "base_score": 0.92,
        "price": 349.00,
        "tags": ["audio", "tech", "premium"],
        "item_features": {"popularity": 0.96, "margin": 0.38, "novelty": 0.40, "ctr_prior": 0.091}
    },
    {
        "item_id": "item_smart_ergonomic_chair",
        "title": "Ergonomic Lumbar Executive Chair",
        "category": "Home & Office",
        "base_score": 0.74,
        "price": 420.00,
        "tags": ["office", "furniture", "comfort"],
        "item_features": {"popularity": 0.72, "margin": 0.50, "novelty": 0.20, "ctr_prior": 0.042}
    },
    {
        "item_id": "item_italian_leather_jacket",
        "title": "Artisan Full-Grain Italian Leather Jacket",
        "category": "Apparel & Fashion",
        "base_score": 0.81,
        "price": 550.00,
        "tags": ["fashion", "luxury", "apparel"],
        "item_features": {"popularity": 0.68, "margin": 0.55, "novelty": 0.75, "ctr_prior": 0.038}
    },
    {
        "item_id": "item_organic_cotton_hoodie",
        "title": "Minimalist Organic Heavyweight Hoodie",
        "category": "Apparel & Fashion",
        "base_score": 0.69,
        "price": 85.00,
        "tags": ["fashion", "casual", "streetwear"],
        "item_features": {"popularity": 0.89, "margin": 0.42, "novelty": 0.50, "ctr_prior": 0.055}
    },
    {
        "item_id": "item_smart_home_hub_ai",
        "title": "Voice Assistant AI Smart Home Controller",
        "category": "Smart Home",
        "base_score": 0.83,
        "price": 129.99,
        "tags": ["smart_home", "iot", "tech"],
        "item_features": {"popularity": 0.78, "margin": 0.35, "novelty": 0.80, "ctr_prior": 0.061}
    },
    {
        "item_id": "item_cold_brew_nitro_maker",
        "title": "Precision Nitro Cold Brew Dispenser",
        "category": "Kitchen & Dining",
        "base_score": 0.71,
        "price": 179.00,
        "tags": ["coffee", "kitchen", "lifestyle"],
        "item_features": {"popularity": 0.64, "margin": 0.48, "novelty": 0.65, "ctr_prior": 0.048}
    },
    {
        "item_id": "item_mirrorless_vlog_camera",
        "title": "4K60p Compact Cinema Vlogging Camera",
        "category": "Electronics & Gaming",
        "base_score": 0.89,
        "price": 999.00,
        "tags": ["photography", "tech", "video", "creator"],
        "item_features": {"popularity": 0.84, "margin": 0.22, "novelty": 0.85, "ctr_prior": 0.073}
    },
    {
        "item_id": "item_titanium_smartwatch_ultra",
        "title": "Titanium GPS Cellular Adventure Watch",
        "category": "Wearables",
        "base_score": 0.94,
        "price": 799.00,
        "tags": ["fitness", "tech", "wearables", "gps"],
        "item_features": {"popularity": 0.92, "margin": 0.34, "novelty": 0.70, "ctr_prior": 0.088}
    },
    {
        "item_id": "item_portable_power_station",
        "title": "1000W Solar Backup Portable Generator",
        "category": "Outdoor & Emergency",
        "base_score": 0.77,
        "price": 849.00,
        "tags": ["outdoor", "energy", "solar", "emergency"],
        "item_features": {"popularity": 0.70, "margin": 0.30, "novelty": 0.45, "ctr_prior": 0.035}
    },
    {
        "item_id": "item_sustainable_sneakers_pro",
        "title": "Zero-Carbon High Performance Running Shoes",
        "category": "Apparel & Fashion",
        "base_score": 0.85,
        "price": 160.00,
        "tags": ["fitness", "fashion", "eco", "footwear"],
        "item_features": {"popularity": 0.91, "margin": 0.40, "novelty": 0.60, "ctr_prior": 0.068}
    }
]

class CandidateGenerator:
    """
    Stage 1: Candidate Generation
    Filters catalog items down to a relevant candidate pool (e.g. 50-200 items)
    prior to computationally heavy ranking.
    """
    def __init__(self, catalog: Optional[List[Dict[str, Any]]] = None):
        self.catalog = catalog or DEFAULT_CATALOG

    def retrieve_candidates(
        self, 
        user_context: Dict[str, Any], 
        limit: int = 20,
        category_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        candidates = []
        user_past_cats = user_context.get("past_interacted_categories", [])
        
        for item in self.catalog:
            if category_filter and item.get("category") != category_filter:
                continue
            
            # Simple matching score based on user context
            match_bonus = 0.0
            if item.get("category") in user_past_cats:
                match_bonus += 0.15
            
            cand = dict(item)
            cand["retrieval_score"] = round(item.get("base_score", 0.5) + match_bonus, 3)
            candidates.append(cand)
            
        # Sort by initial retrieval heuristic and slice top candidates
        candidates.sort(key=lambda x: x["retrieval_score"], reverse=True)
        return candidates[:limit]
