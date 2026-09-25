"""
ML Challenge 2026 - LightGBM Matching Model & Threshold Tuning
Trains a pairwise gradient boosted classifier and sweeps decision thresholds
to strictly maximize competition macro-averaged F_0.5.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Set, Tuple
from metrics import evaluate_macro_f05


def train_matching_gbm(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "label",
    scale_pos_weight: float = 10.0,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    num_leaves: int = 31,
) -> lgb.LGBMClassifier:
    """
    Trains a LightGBM classifier on pairwise features with class weighting.
    """
    X_train = train_df[feature_cols]
    y_train = train_df[label_col].values

    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        importance_type="gain",
        verbose=-1
    )
    model.fit(X_train, y_train)
    return model


def sweep_macro_f05_threshold(
    val_pairs_df: pd.DataFrame,
    val_probs: np.ndarray,
    ground_truth: Dict[str, Set[str]],
    thresholds: np.ndarray = np.arange(0.30, 0.95, 0.02),
    top_k_cap: int = 10
) -> Tuple[float, float, Dict[str, float]]:
    """
    Sweeps the decision threshold on validation candidate pairs to directly maximize
    macro per-entity F_0.5.
    
    val_pairs_df must have columns: ['source1_entity_id', 'candidate_id']
    ground_truth: dict mapping source1_entity_id -> set of true matched IDs
    """
    df = val_pairs_df[["source1_entity_id", "candidate_id"]].copy()
    df["prob"] = val_probs

    best_thresh = 0.50
    best_score = -1.0
    best_details = {}

    # Sort df by prob descending so top_k_cap takes highest probabilities first
    df = df.sort_values(by="prob", ascending=False)

    for thresh in thresholds:
        # Filter pairs exceeding threshold
        matched_pairs = df[df["prob"] >= thresh]

        # Group by source1_entity_id up to top_k_cap
        predictions = {s1: set() for s1 in ground_truth.keys()}
        for s1_id, group in matched_pairs.groupby("source1_entity_id"):
            if s1_id in predictions:
                cands = group["candidate_id"].head(top_k_cap).tolist()
                predictions[s1_id] = set(cands)

        # Evaluate macro F_0.5
        eval_res = evaluate_macro_f05(ground_truth, predictions)
        score = eval_res["macro_f05"]

        if score > best_score:
            best_score = score
            best_thresh = thresh
            best_details = eval_res

    return best_thresh, best_score, best_details
