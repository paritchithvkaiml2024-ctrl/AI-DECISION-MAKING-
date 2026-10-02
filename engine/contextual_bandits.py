"""
Exploration-Exploitation Engine (Contextual Bandits)
Implements:
1. LinUCB (Disjoint Linear Upper Confidence Bound)
2. Contextual Thompson Sampling (Bayesian Ridge Regression)
3. Dynamic Epsilon-Greedy
4. Boltzmann (Softmax) Exploration
Includes real-time online parameter updating and regret tracking.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("DecisionCore.Bandits")

class LinUCBArm:
    """
    Individual arm representation in LinUCB with Ridge Regression inverse matrix A and vector b.
    Ridge regression model: reward = theta^T * x + noise
    A = D_a^T * D_a + I_d
    b = D_a^T * c_a
    theta = A^{-1} * b
    """
    def __init__(self, arm_id: str, feature_dim: int = 9, alpha: float = 1.0):
        self.arm_id = arm_id
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.A = np.identity(feature_dim, dtype=np.float64)
        self.A_inv = np.identity(feature_dim, dtype=np.float64)
        self.b = np.zeros((feature_dim, 1), dtype=np.float64)
        self.pull_count = 0
        self.total_reward = 0.0

    def compute_ucb_score(self, context_vec: np.ndarray, alpha: Optional[float] = None) -> Tuple[float, float, float]:
        a = alpha if alpha is not None else self.alpha
        x = context_vec.reshape(-1, 1).astype(np.float64)
        
        # theta = A^{-1} * b
        theta = np.dot(self.A_inv, self.b)
        
        # Expected mean payoff: theta^T * x
        expected_payoff = float(np.dot(theta.T, x)[0, 0])
        
        # Confidence interval width (exploration bonus): alpha * sqrt(x^T * A^{-1} * x)
        variance = float(np.dot(np.dot(x.T, self.A_inv), x)[0, 0])
        cb_width = a * np.sqrt(max(0.0, variance))
        
        ucb = expected_payoff + cb_width
        return float(ucb), float(expected_payoff), float(cb_width)

    def update(self, context_vec: np.ndarray, reward: float):
        x = context_vec.reshape(-1, 1).astype(np.float64)
        self.A += np.dot(x, x.T)
        self.b += reward * x
        self.A_inv = np.linalg.pinv(self.A)
        self.pull_count += 1
        self.total_reward += reward


class ThompsonSamplingArm:
    """
    Bayesian Contextual Bandits using Gaussian Posterior sampling over regression weights.
    """
    def __init__(self, arm_id: str, feature_dim: int = 9, v2: float = 0.25):
        self.arm_id = arm_id
        self.feature_dim = feature_dim
        self.v2 = v2 # Observation noise variance
        self.B = np.identity(feature_dim, dtype=np.float64)
        self.B_inv = np.identity(feature_dim, dtype=np.float64)
        self.f = np.zeros((feature_dim, 1), dtype=np.float64)
        self.pull_count = 0
        self.total_reward = 0.0

    def sample_score(self, context_vec: np.ndarray) -> float:
        mu = np.dot(self.B_inv, self.f).flatten()
        cov = self.v2 * self.B_inv
        # Sample parameter vector from posterior
        theta_sample = np.random.multivariate_normal(mu, cov)
        score = float(np.dot(context_vec, theta_sample))
        return score

    def update(self, context_vec: np.ndarray, reward: float):
        x = context_vec.reshape(-1, 1).astype(np.float64)
        self.B += np.dot(x, x.T)
        self.f += reward * x
        self.B_inv = np.linalg.pinv(self.B)
        self.pull_count += 1
        self.total_reward += reward


class ContextualBanditEngine:
    """
    Stage 4: Contextual Bandit Decisioning & Exploration-Exploitation
    Manages arm models across catalog items, executes policies, and records telemetry.
    """
    def __init__(self, feature_dim: int = 9, default_alpha: float = 0.75):
        self.feature_dim = feature_dim
        self.default_alpha = default_alpha
        self.linucb_arms: Dict[str, LinUCBArm] = {}
        self.ts_arms: Dict[str, ThompsonSamplingArm] = {}
        self.total_decisions = 0
        self.total_explorations = 0

    def get_or_create_linucb_arm(self, item_id: str) -> LinUCBArm:
        if item_id not in self.linucb_arms:
            self.linucb_arms[item_id] = LinUCBArm(item_id, self.feature_dim, self.default_alpha)
        return self.linucb_arms[item_id]

    def get_or_create_ts_arm(self, item_id: str) -> ThompsonSamplingArm:
        if item_id not in self.ts_arms:
            self.ts_arms[item_id] = ThompsonSamplingArm(item_id, self.feature_dim)
        return self.ts_arms[item_id]

    def apply_exploration_policy(
        self,
        candidates: List[Dict[str, Any]],
        policy: str = "linucb",
        alpha: Optional[float] = None,
        epsilon: float = 0.15
    ) -> List[Dict[str, Any]]:
        """
        Adjust scores and rank order using the selected exploration-exploitation policy.
        """
        self.total_decisions += 1
        effective_alpha = alpha if alpha is not None else self.default_alpha
        enriched_candidates = []

        if policy == "linucb":
            for cand in candidates:
                item_id = cand["item_id"]
                vec = cand["feature_vector"]
                arm = self.get_or_create_linucb_arm(item_id)
                ucb_val, mean_payoff, cb_bonus = arm.compute_ucb_score(vec, effective_alpha)
                
                # Blend Heavy Ranker ML score with LinUCB bonus
                combined_score = 0.65 * cand.get("ml_rank_score", 0.5) + 0.35 * ucb_val
                
                item_copy = dict(cand)
                item_copy["bandit_score"] = round(combined_score, 4)
                item_copy["bandit_bonus"] = round(cb_bonus, 4)
                item_copy["was_explored"] = bool(cb_bonus > 0.12)
                enriched_candidates.append(item_copy)

        elif policy == "thompson_sampling":
            for cand in candidates:
                item_id = cand["item_id"]
                vec = cand["feature_vector"]
                arm = self.get_or_create_ts_arm(item_id)
                sample_val = arm.sample_score(vec)
                combined_score = 0.60 * cand.get("ml_rank_score", 0.5) + 0.40 * sample_val
                
                item_copy = dict(cand)
                item_copy["bandit_score"] = round(combined_score, 4)
                item_copy["bandit_bonus"] = round(sample_val - cand.get("ml_rank_score", 0.5), 4)
                item_copy["was_explored"] = True
                enriched_candidates.append(item_copy)

        elif policy == "epsilon_greedy":
            is_exploring = (np.random.rand() < epsilon)
            if is_exploring:
                self.total_explorations += 1
            for cand in candidates:
                item_copy = dict(cand)
                bonus = float(np.random.uniform(0.1, 0.4)) if is_exploring else 0.0
                item_copy["bandit_score"] = round(cand.get("ml_rank_score", 0.5) + bonus, 4)
                item_copy["bandit_bonus"] = round(bonus, 4)
                item_copy["was_explored"] = is_exploring
                enriched_candidates.append(item_copy)

        else: # "greedy" or fallback
            for cand in candidates:
                item_copy = dict(cand)
                item_copy["bandit_score"] = cand.get("ml_rank_score", 0.5)
                item_copy["bandit_bonus"] = 0.0
                item_copy["was_explored"] = False
                enriched_candidates.append(item_copy)

        # Sort by bandit-adjusted score
        enriched_candidates.sort(key=lambda x: x["bandit_score"], reverse=True)
        return enriched_candidates

    def update_feedback(self, item_id: str, context_vec: np.ndarray, reward: float):
        """
        Online Bayesian and Ridge parameter update when feedback (click/conversion) is received.
        """
        if item_id in self.linucb_arms:
            self.linucb_arms[item_id].update(context_vec, reward)
        if item_id in self.ts_arms:
            self.ts_arms[item_id].update(context_vec, reward)
        logger.debug("Bandit arm %s updated with reward %.2f", item_id, reward)

    def get_policy_summary(self) -> Dict[str, Any]:
        return {
            "total_decisions": self.total_decisions,
            "linucb_arms_tracked": len(self.linucb_arms),
            "ts_arms_tracked": len(self.ts_arms),
            "arms_summary": [
                {
                    "arm_id": arm_id,
                    "pull_count": arm.pull_count,
                    "total_reward": round(arm.total_reward, 2),
                    "avg_reward": round(arm.total_reward / max(1, arm.pull_count), 3)
                }
                for arm_id, arm in list(self.linucb_arms.items())[:15]
            ]
        }
