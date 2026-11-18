# member3_train_model.py
"""
Member 3 — Enhanced Model Training & Evaluation Script
======================================================
Trains the JobShield AI 4-Branch Hybrid Ensemble Model:
  1. Sublinear TF-IDF Unigram + Bigram Vectorizer (75,000 n-grams, L2 normalized)
  2. Structured + N-Gram NBSVM L2-Regularized Logistic Regression
  3. Pure-Text N-Gram NBSVM L2-Regularized Logistic Regression
  4. Sparse TF-IDF Cosine k-NN Memory Bank + Domain Risk Calibrator

Saves:
  - fraud_model.pkl
  - tfidf_vectorizer.pkl
  - model_metrics.json
"""

import json
import time
import numpy as np
import pandas as pd
from scipy import sparse

from jobshield_model import (
    JobShieldTfidfVectorizer,
    JobShieldHybridModel,
    URGENCY_PHRASES,
    HIGH_RISK_SCAM_PHRASES,
    MEDIUM_RISK_SCAM_PHRASES,
    normalize_job_text,
    save_artifact,
)


def compute_roc_auc(y_true, y_prob):
    """Exact ROC-AUC calculation via rank statistics (Mann-Whitney U)."""
    y_true = np.asarray(y_true, dtype=np.int32)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    order = np.argsort(y_prob)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_true) + 1)
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    pos_ranks_sum = np.sum(ranks[y_true == 1])
    auc = (pos_ranks_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def build_numeric_feature_matrix(sub_df: pd.DataFrame) -> np.ndarray:
    ft = sub_df["combined_text"].fillna("")
    urg = np.array([sum(t.count(w) for w in URGENCY_PHRASES) for t in ft], dtype=np.float32)
    scam_terms = HIGH_RISK_SCAM_PHRASES + MEDIUM_RISK_SCAM_PHRASES + [
        "__email_token__", "__phone_token__", "__free_email_token__"
    ]
    scam = np.array([sum(t.count(w) for w in scam_terms) for t in ft], dtype=np.float32)
    vag = (
        (sub_df["company_profile"].str.strip() == "").astype(np.float32)
        + (sub_df["required_education"].str.strip() == "").astype(np.float32)
        + (sub_df["industry"].str.strip() == "").astype(np.float32)
        + (sub_df["benefits"].str.strip() == "").astype(np.float32)
    ).values
    tlen = np.log1p(ft.str.split().str.len().values.astype(np.float32))
    clen = np.log1p(sub_df["company_profile"].str.len().values.astype(np.float32))
    rlen = np.log1p(sub_df["requirements"].str.len().values.astype(np.float32))
    blen = np.log1p(sub_df["benefits"].str.len().values.astype(np.float32))
    has_sal = (sub_df["salary_range"].str.strip() != "").astype(np.float32).values
    tele = sub_df["telecommuting"].values.astype(np.float32)
    logo = sub_df["has_company_logo"].values.astype(np.float32)
    ques = sub_df["has_questions"].values.astype(np.float32)
    return np.column_stack([urg, scam, vag, tlen, clen, rlen, blen, has_sal, tele, logo, ques])


def main():
    t0 = time.time()
    print("=" * 72)
    print("🛡️  JobShield AI — High-Accuracy Hybrid Ensemble Training Pipeline")
    print("=" * 72)

    df = pd.read_csv("fake_job_postings.csv")
    text_cols = [
        "title", "location", "department", "salary_range", "company_profile",
        "description", "requirements", "benefits", "employment_type",
        "required_experience", "required_education", "industry", "function",
    ]
    for c in text_cols:
        df[c] = df[c].fillna("").astype(str)
    for c in ["telecommuting", "has_company_logo", "has_questions", "fraudulent"]:
        df[c] = df[c].fillna(0).astype(int)

    df["combined_text"] = (
        df["title"] + " " + df["title"] + " "
        + df["location"] + " " + df["department"] + " " + df["salary_range"] + " "
        + df["company_profile"] + " " + df["description"] + " "
        + df["requirements"] + " " + df["benefits"] + " "
        + df["employment_type"] + " " + df["required_experience"] + " "
        + df["required_education"] + " " + df["industry"] + " " + df["function"]
    ).apply(normalize_job_text)

    # Stratified 80/20 Train/Test Split (random_state=42)
    np.random.seed(42)
    pos_idx = np.where(df["fraudulent"].values == 1)[0]
    neg_idx = np.where(df["fraudulent"].values == 0)[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)

    n_pos_test = int(round(len(pos_idx) * 0.2))
    n_neg_test = int(round(len(neg_idx) * 0.2))
    test_idx = np.concatenate([pos_idx[:n_pos_test], neg_idx[:n_neg_test]])
    train_idx = np.concatenate([pos_idx[n_pos_test:], neg_idx[n_neg_test:]])

    print(f"[1/4] Dataset loaded: {len(df):,} total postings ({len(pos_idx):,} Fake, {len(neg_idx):,} Real)")
    print(f"      Stratified Split -> Train: {len(train_idx):,} | Test: {len(test_idx):,}")

    # Vectorize with 75,000 sublinear TF-IDF unigrams & bigrams
    print("[2/4] Fitting 75,000-feature Sublinear TF-IDF Unigram+Bigram Vectorizer...")
    vectorizer = JobShieldTfidfVectorizer(max_features=75000, min_df=2)
    X_train_text = vectorizer.fit_transform(df["combined_text"].iloc[train_idx].tolist())
    X_test_text = vectorizer.transform(df["combined_text"].iloc[test_idx].tolist())
    print(f"      Vocabulary size: {len(vectorizer.vocab_):,} n-grams")

    # Numeric clue matrix
    print("[3/4] Engineering 11 structural, urgency, scam-lure, and vagueness features...")
    F_train = build_numeric_feature_matrix(df.iloc[train_idx])
    F_test = build_numeric_feature_matrix(df.iloc[test_idx])
    y_train = df["fraudulent"].iloc[train_idx].values
    y_test = df["fraudulent"].iloc[test_idx].values

    # Train 4-branch Hybrid Ensemble
    print("[4/4] Training 4-Branch Hybrid Ensemble (NBSVM-Struct + NBSVM-Text + Cosine k-NN + Calibrator)...")
    model = JobShieldHybridModel(threshold=0.40, knn_weight=0.25)
    model.fit(
        X_train_text,
        F_train,
        y_train,
        train_titles=df["title"].iloc[train_idx].tolist(),
        vocab_items=vectorizer.vocab_items_,
    )

    # Evaluate on unseen test set (3,576 postings)
    X_test_input = sparse.hstack([X_test_text, F_test], format="csr")
    y_prob = model.predict_proba(X_test_input)[:, 1]
    y_pred = (y_prob >= model.threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_test == 1)))
    fp = int(np.sum((y_pred == 1) & (y_test == 0)))
    fn = int(np.sum((y_pred == 0) & (y_test == 1)))
    tn = int(np.sum((y_pred == 0) & (y_test == 0)))

    prec_fake = tp / (tp + fp)
    rec_fake = tp / (tp + fn)
    f1_fake = 2 * prec_fake * rec_fake / (prec_fake + rec_fake)

    prec_real = tn / (tn + fn)
    rec_real = tn / (tn + fp)
    f1_real = 2 * prec_real * rec_real / (prec_real + rec_real)

    acc = (tp + tn) / len(y_test)
    roc_auc = compute_roc_auc(y_test, y_prob)

    metrics = {
        "dataset_total": int(len(df)),
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "vocab_size": int(len(vectorizer.vocab_)),
        "threshold": float(model.threshold),
        "accuracy": round(acc * 100, 2),
        "roc_auc": round(roc_auc * 100, 2),
        "fake_precision": round(prec_fake * 100, 2),
        "fake_recall": round(rec_fake * 100, 2),
        "fake_f1": round(f1_fake * 100, 2),
        "real_precision": round(prec_real * 100, 2),
        "real_recall": round(rec_real * 100, 2),
        "real_f1": round(f1_real * 100, 2),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "baseline_comparison": {
            "baseline_accuracy": 98.13,
            "baseline_fake_precision": 93.44,
            "baseline_fake_recall": 65.90,
            "baseline_fake_f1": 77.29,
        },
        "training_seconds": round(time.time() - t0, 2),
    }
    model.metrics_summary_ = metrics

    print("\n" + "=" * 72)
    print("📊 EVALUATION ON 3,576 UNSEEN TEST POSTINGS (STRATIFIED 80/20 SPLIT)")
    print("=" * 72)
    print(f"Overall Accuracy : {acc*100:.2f}%  (Baseline: 98.13%)")
    print(f"ROC-AUC Score    : {roc_auc*100:.2f}%")
    print("-" * 72)
    print(f"{'Class':<15} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>10}")
    print("-" * 72)
    print(f"{'Real (0)':<15} {prec_real:12.4f} {rec_real:12.4f} {f1_real:12.4f} {tn+fp:10d}")
    print(f"{'Fake (1)':<15} {prec_fake:12.4f} {rec_fake:12.4f} {f1_fake:12.4f} {tp+fn:10d}")
    print("-" * 72)
    print("Confusion Matrix:")
    print(f"  Actually Real -> Predicted Real (TN): {tn:4d} | Predicted Fake (FP): {fp:4d}")
    print(f"  Actually Fake -> Predicted Fake (TP): {tp:4d} | Predicted Real (FN): {fn:4d}")
    print("=" * 72)

    save_artifact(model, "fraud_model.pkl")
    save_artifact(vectorizer, "tfidf_vectorizer.pkl")
    with open("model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Saved: fraud_model.pkl, tfidf_vectorizer.pkl, model_metrics.json in {time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()
