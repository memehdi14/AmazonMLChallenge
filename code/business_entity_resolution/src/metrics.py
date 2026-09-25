"""
ML Challenge 2026 - Evaluation Metrics Module
Implements exact competition F_0.5 macro-averaged score per Source 1 entity,
and candidate generation recall metrics.
"""

from typing import Dict, Set, Iterable


def compute_entity_f05(true_matches: Set[str], pred_matches: Set[str]) -> float:
    """
    Compute F_0.5 for a single Source 1 entity.
    
    Formula: F_0.5 = (1.25 * P * R) / (0.25 * P + R)
    
    Singleton logic per competition rules:
    - If true_matches is empty (singleton):
        - If pred_matches is empty -> 1.0 (correctly identified singleton)
        - If pred_matches is non-empty -> 0.0 (false merge on singleton)
    - If true_matches is non-empty:
        - If pred_matches is empty -> 0.0
        - Otherwise calculate standard precision, recall, F_0.5
    """
    if len(true_matches) == 0:
        return 1.0 if len(pred_matches) == 0 else 0.0

    if len(pred_matches) == 0:
        return 0.0

    tp = len(true_matches.intersection(pred_matches))
    if tp == 0:
        return 0.0

    precision = tp / len(pred_matches)
    recall = tp / len(true_matches)

    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0

    return (1.25 * precision * recall) / denom


def evaluate_macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]]
) -> Dict[str, float]:
    """
    Computes macro-averaged F_0.5 across all Source 1 entities in ground_truth.
    
    Returns a dict with:
    - 'macro_f05': Average F_0.5 score
    - 'singleton_accuracy': Fraction of singletons scored 1.0
    - 'non_singleton_f05': Average F_0.5 score on non-singletons
    - 'precision_macro': Average precision across non-singletons with predictions
    - 'recall_macro': Average recall across non-singletons
    """
    total_entities = len(ground_truth)
    if total_entities == 0:
        return {"macro_f05": 0.0}

    f05_scores = []
    singleton_scores = []
    non_singleton_scores = []

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        score = compute_entity_f05(true_set, pred_set)
        f05_scores.append(score)

        if len(true_set) == 0:
            singleton_scores.append(score)
        else:
            non_singleton_scores.append(score)

    macro_f05 = sum(f05_scores) / total_entities
    singleton_acc = sum(singleton_scores) / len(singleton_scores) if singleton_scores else 0.0
    non_singleton_f05 = sum(non_singleton_scores) / len(non_singleton_scores) if non_singleton_scores else 0.0

    return {
        "macro_f05": macro_f05,
        "singleton_accuracy": singleton_acc,
        "non_singleton_f05": non_singleton_f05,
        "num_entities": total_entities,
        "num_singletons": len(singleton_scores)
    }


def evaluate_blocking_recall(
    ground_truth: Dict[str, Set[str]],
    candidates: Dict[str, Set[str]]
) -> Dict[str, float]:
    """
    Evaluate candidate generation / blocking recall ceiling:
    Recall = (Total true matches found in candidates) / (Total true matches in ground truth)
    
    Also calculates:
    - Fraction of entities whose true matches were 100% captured
    - Average number of candidates per entity
    """
    total_true_matches = 0
    captured_matches = 0
    fully_covered_entities = 0
    non_singleton_entities = 0
    candidate_counts = []

    for s1_id, true_set in ground_truth.items():
        cand_set = candidates.get(s1_id, set())
        candidate_counts.append(len(cand_set))

        if len(true_set) > 0:
            non_singleton_entities += 1
            total_true_matches += len(true_set)
            hits = len(true_set.intersection(cand_set))
            captured_matches += hits
            if hits == len(true_set):
                fully_covered_entities += 1

    overall_recall = captured_matches / total_true_matches if total_true_matches > 0 else 1.0
    full_coverage_rate = fully_covered_entities / non_singleton_entities if non_singleton_entities > 0 else 1.0
    avg_candidates = sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0.0

    return {
        "recall_ceiling": overall_recall,
        "full_coverage_rate": full_coverage_rate,
        "captured_matches": captured_matches,
        "total_true_matches": total_true_matches,
        "avg_candidates_per_entity": avg_candidates,
        "max_candidates_for_entity": max(candidate_counts) if candidate_counts else 0
    }
