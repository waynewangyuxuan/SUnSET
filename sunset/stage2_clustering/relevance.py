"""
Relevance Score Computation

Implements Rel(ς, d) = β · P(ς, d) · R(count(ς_d)) from paper Equation 1.
"""

import math
import numpy as np
from typing import Dict
from .global_stats import GlobalStats


def compute_reward(count_s_d: int) -> float:
    """
    Compute reward score R(x) = tanh(x/10).

    From paper Equation 3.

    Args:
        count_s_d: Stakeholder count in current topic

    Returns:
        Reward score in [0, 1)
    """
    return math.tanh(count_s_d / 10.0)


def compute_penalty(
    stakeholder: str,
    topic: str,
    global_stats: GlobalStats
) -> float:
    """
    Compute penalty score P(ς, d).

    From paper Equation 2:
    P(ς, d) = (σ_D / x̄) · √|D| · (count(ς_d) / count(ς_D))

    Where σ_D/x̄ is the coefficient of variation (CV).

    Args:
        stakeholder: Stakeholder name
        topic: Current topic name
        global_stats: Global statistics across all topics

    Returns:
        Penalty score in [0, 1]
    """
    # Get counts across all topics
    counts = global_stats.get_stakeholder_counts(stakeholder)

    if not counts:
        return 0.0

    # Compute mean and std
    mean_val = np.mean(counts)
    std_val = np.std(counts)

    if mean_val == 0:
        return 0.0

    # Coefficient of variation
    cv = std_val / mean_val

    # Topic-specific ratio
    count_s_d = global_stats.get_topic_count(stakeholder, topic)
    count_s_D = global_stats.get_total_count(stakeholder)

    if count_s_D == 0:
        return 0.0

    topic_ratio = count_s_d / count_s_D

    # Number of topics
    num_topics = len(global_stats.topics)

    # Penalty formula
    P = cv * math.sqrt(num_topics) * topic_ratio

    # Clamp to [0, 1] as proven in Appendix C
    P = max(0.0, min(1.0, P))

    return P


def compute_relevance(
    stakeholder: str,
    topic: str,
    global_stats: GlobalStats,
    beta: float = 1.0
) -> float:
    """
    Compute relevance score Rel(ς, d) = β · P(ς, d) · R(count(ς_d)).

    From paper Equation 1.

    Args:
        stakeholder: Stakeholder name
        topic: Current topic name
        global_stats: Global statistics across all topics
        beta: Scaling hyperparameter (default 1.0)

    Returns:
        Relevance score
    """
    count_s_d = global_stats.get_topic_count(stakeholder, topic)
    P = compute_penalty(stakeholder, topic, global_stats)
    R = compute_reward(count_s_d)

    return beta * P * R


def compute_stakeholder_relevances(
    stakeholders: set,
    topic: str,
    global_stats: GlobalStats,
    beta: float = 1.0
) -> Dict[str, float]:
    """
    Compute relevance for all stakeholders in a set.

    Args:
        stakeholders: Set of stakeholder names
        topic: Current topic name
        global_stats: Global statistics
        beta: Scaling hyperparameter

    Returns:
        Dict mapping stakeholder -> relevance score
    """
    return {
        s: compute_relevance(s, topic, global_stats, beta)
        for s in stakeholders
    }
