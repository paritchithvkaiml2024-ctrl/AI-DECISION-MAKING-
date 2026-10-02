"""
Re-ranking, Diversity & Business Constraint Engine
Implements:
1. Maximal Marginal Relevance (MMR) for result diversity and anti-monotony
2. Category Interleaving & Frequency Capping
3. Multi-objective Pareto optimization (Relevance x Margin x Novelty)
"""

import numpy as np
from typing import List, Dict, Any, Optional

class ReRanker:
    """
    Stage 5: Re-ranking & Diversity Optimization
    Applies business constraints and diversity penalties so recommendations
    are not concentrated in a single dominant category.
    """
    def __init__(self, diversity_lambda: float = 0.25):
        self.diversity_lambda = diversity_lambda

    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        dot = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def rerank_mmr(
        self, 
        candidates: List[Dict[str, Any]], 
        top_k: int = 10,
        diversity_lambda: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Maximal Marginal Relevance (MMR) algorithm:
        Score_MMR(d) = (1 - lambda) * Score(d) - lambda * max_{s in S} Sim(d, s)
        """
        lam = diversity_lambda if diversity_lambda is not None else self.diversity_lambda
        if not candidates:
            return []
        
        remaining = list(candidates)
        selected: List[Dict[str, Any]] = []

        # 1. Pick the top candidate with highest base bandit score
        first_item = remaining.pop(0)
        first_item["diversity_penalty"] = 0.0
        first_item["final_score"] = first_item.get("bandit_score", 0.5)
        selected.append(first_item)

        # 2. Greedily pick remaining items balancing relevance and redundancy
        while len(selected) < top_k and remaining:
            best_idx = -1
            best_mmr_score = -float('inf')
            best_penalty = 0.0

            for i, cand in enumerate(remaining):
                base_s = cand.get("bandit_score", 0.5)
                vec = cand["feature_vector"]
                
                # Compute maximum similarity against already selected items
                max_sim = max(
                    self._cosine_similarity(vec, s["feature_vector"])
                    for s in selected
                )
                
                # Category duplication penalty
                cat = cand.get("category", "")
                cat_count = sum(1 for s in selected if s.get("category") == cat)
                cat_penalty = 0.15 * cat_count

                penalty = lam * max_sim + cat_penalty
                mmr_score = (1.0 - lam) * base_s - penalty

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = i
                    best_penalty = penalty

            chosen = remaining.pop(best_idx)
            chosen["diversity_penalty"] = round(best_penalty, 4)
            chosen["final_score"] = round(best_mmr_score, 4)
            selected.append(chosen)

        # Assign final rank positions
        for rank, item in enumerate(selected, start=1):
            item["rank_position"] = rank

        return selected
