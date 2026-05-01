from typing import List

import numpy as np


def _cosine_similarity(vec_a, vec_b) -> float:
    a = np.array(vec_a, dtype=float)
    b = np.array(vec_b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def mmr_rerank(
    query_vector,
    candidates: List[dict],
    top_k: int = 6,
    lambda_mult: float = 0.7,
):
    if not candidates:
        return []

    remaining = candidates.copy()
    selected = []

    while remaining and len(selected) < top_k:
        best_item = None
        best_score = float("-inf")

        for item in remaining:
            relevance = _cosine_similarity(query_vector, item["vector"])
            if not selected:
                mmr_score = relevance
            else:
                diversity_penalty = max(
                    _cosine_similarity(item["vector"], chosen["vector"])
                    for chosen in selected
                )
                mmr_score = lambda_mult * relevance - (1 - lambda_mult) * diversity_penalty

            if mmr_score > best_score:
                best_score = mmr_score
                best_item = item

        selected.append(best_item)
        remaining.remove(best_item)

    return selected
