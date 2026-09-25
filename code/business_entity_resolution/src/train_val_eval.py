"""
Phase 2 & Phase 3: Feature Extraction, LightGBM Training & Macro F_0.5 Threshold Sweep
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Set, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import evaluate_macro_f05, compute_entity_f05
from features import compute_pairwise_features
from train_gbm import train_matching_gbm, sweep_macro_f05_threshold


def evaluate_and_train_gbm(
    val_candidates: Dict[str, Set[str]],
    s1_records: Dict[str, Dict] = None,
    target_records: Dict[str, Dict] = None,
    ground_truth: Dict[str, Set[str]] = None,
    test_ratio: float = 0.3,
    df_features: Optional[pd.DataFrame] = None
):
    """
    Builds pairwise features on candidate pairs, splits into train/val folds,
    trains LightGBM, and performs threshold optimization for macro F_0.5.
    """
    if df_features is not None:
        df = df_features
        print(f"Using precomputed feature dataframe with {len(df):,} pairs.")
    else:
        print("\n" + "="*50)
        print("=== PHASE 2: BUILDING PAIRWISE FEATURES ===")
        print("="*50)
        
        rows = []
        t0 = time.time()
        
        for s1_id, cands in val_candidates.items():
            s1 = s1_records[s1_id]
            true_set = ground_truth.get(s1_id, set()) if ground_truth else set()
            
            for cid in cands:
                if cid not in target_records:
                    continue
                cand = target_records[cid]
                
                feats = compute_pairwise_features(
                    s1_name_clean=s1["clean_name"],
                    s1_name_stripped=s1["stripped_name"],
                    s1_name_tokens=s1["name_tokens"],
                    s1_addr_clean=s1["clean_addr"],
                    s1_addr_tokens=s1["addr_tokens"],
                    s1_num_tokens=s1["num_tokens"],
                    s1_country=s1["country"],
                    s1_has_landmark=s1.get("has_landmark", 0),
                    cand_id=cid,
                    cand_name_clean=cand["clean_name"],
                    cand_name_stripped=cand["stripped_name"],
                    cand_name_tokens=cand["name_tokens"],
                    cand_addr_clean=cand["clean_addr"],
                    cand_addr_tokens=cand["addr_tokens"],
                    cand_num_tokens=cand["num_tokens"],
                    cand_country=cand["country"],
                    cand_has_landmark=cand.get("has_landmark", 0)
                )
                feats["source1_entity_id"] = s1_id
                feats["candidate_id"] = cid
                feats["label"] = 1 if cid in true_set else 0
                rows.append(feats)

        df = pd.DataFrame(rows)
        print(f"Constructed {len(df):,} candidate pairs in {time.time() - t0:.2f}s.")
    pos_count = (df["label"] == 1).sum()
    neg_count = (df["label"] == 0).sum()
    print(f"Class Balance: {pos_count:,} positives (true matches) vs {neg_count:,} negatives ({pos_count/len(df)*100:.2f}% positive).")

    # Feature columns (exclude IDs and label)
    feature_cols = [c for c in df.columns if c not in ("source1_entity_id", "candidate_id", "label")]
    print(f"Total Features ({len(feature_cols)}): {feature_cols}")

    # Split by Source 1 entity_id to prevent data leakage!
    all_s1 = list(val_candidates.keys())
    np.random.seed(42)
    np.random.shuffle(all_s1)
    split_idx = int(len(all_s1) * (1.0 - test_ratio))
    train_s1 = set(all_s1[:split_idx])
    test_s1 = set(all_s1[split_idx:])

    train_df = df[df["source1_entity_id"].isin(train_s1)].copy()
    test_df = df[df["source1_entity_id"].isin(test_s1)].copy()
    test_gt = {s1: ground_truth.get(s1, set()) for s1 in test_s1}

    print(f"\nTraining on {len(train_df):,} pairs ({len(train_s1)} entities), Evaluating on {len(test_df):,} pairs ({len(test_s1)} entities)...")

    # Calculate class weight ratio
    scale_pos = max(neg_count / max(pos_count, 1), 1.0)
    print(f"Class weighting scale_pos_weight: {scale_pos:.2f}")

    print("\n" + "="*50)
    print("=== PHASE 3: TRAINING LIGHTGBM MATCHING MODEL ===")
    print("="*50)

    model = train_matching_gbm(
        train_df=train_df,
        feature_cols=feature_cols,
        scale_pos_weight=min(scale_pos, 15.0), # Capped to avoid extreme false positives
        n_estimators=400,
        learning_rate=0.04,
        max_depth=6,
        num_leaves=31
    )

    # Feature Importance
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop 10 Feature Importances:")
    print(importances.head(10))

    # Evaluate on held-out test_df
    X_test = test_df[feature_cols]
    test_probs = model.predict_proba(X_test)[:, 1]

    print("\n" + "="*50)
    print("=== SWEEPING DECISION THRESHOLD FOR MACRO F_0.5 ===")
    print("="*50)

    best_thresh, best_f05, details = sweep_macro_f05_threshold(
        val_pairs_df=test_df,
        val_probs=test_probs,
        ground_truth=test_gt,
        thresholds=np.arange(0.35, 0.96, 0.02),
        top_k_cap=8
    )

    print(f"\nOPTIMAL THRESHOLD: {best_thresh:.2f}")
    print(f"VALIDATION MACRO F_0.5: {best_f05:.4f}")
    print(f"Singleton Accuracy:     {details.get('singleton_accuracy', 0.0):.4f}")
    print(f"Non-Singleton F_0.5:    {details.get('non_singleton_f05', 0.0):.4f}")
    print("="*50)

    return model, best_thresh, best_f05, feature_cols
