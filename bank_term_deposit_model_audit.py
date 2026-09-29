from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DATA = PROJECT_DIR / "train.csv"
OUT = PROJECT_DIR / "model_audit_results.json"


def stratified_indices(y: np.ndarray, seed: int = 42):
    rng = np.random.default_rng(seed)
    train, valid, test = [], [], []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        n = len(idx)
        a, b = int(0.60 * n), int(0.80 * n)
        train.extend(idx[:a])
        valid.extend(idx[a:b])
        test.extend(idx[b:])
    for values in (train, valid, test):
        rng.shuffle(values)
    return np.array(train), np.array(valid), np.array(test)


def transform(train: pd.DataFrame, others: list[pd.DataFrame], include_duration: bool):
    numeric = ["age", "balance", "day", "campaign", "pdays", "previous"]
    if include_duration:
        numeric.append("duration")
    categorical = [
        "job", "marital", "education", "default", "housing", "loan",
        "contact", "month", "poutcome",
    ]
    train_num = train[numeric].astype(float)
    means = train_num.mean()
    stds = train_num.std().replace(0, 1)
    train_num = (train_num - means) / stds
    train_cat = pd.get_dummies(train[categorical], drop_first=False, dtype=float)
    columns = list(train_cat.columns)

    result = [np.column_stack([train_num.to_numpy(), train_cat.to_numpy()])]
    for frame in others:
        num = (frame[numeric].astype(float) - means) / stds
        cat = pd.get_dummies(frame[categorical], drop_first=False, dtype=float)
        cat = cat.reindex(columns=columns, fill_value=0)
        result.append(np.column_stack([num.to_numpy(), cat.to_numpy()]))
    return result


def sigmoid(z):
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logistic(x, y, class_weight=True, l2=0.2, steps=3500, lr=0.035):
    x = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(x.shape[1])
    m = np.zeros_like(beta)
    v = np.zeros_like(beta)
    if class_weight:
        count0, count1 = np.bincount(y)
        weights = np.where(y == 1, len(y) / (2 * count1), len(y) / (2 * count0))
    else:
        weights = np.ones(len(y))
    for t in range(1, steps + 1):
        p = sigmoid(x @ beta)
        grad = (x.T @ ((p - y) * weights)) / weights.sum()
        grad[1:] += l2 * beta[1:] / len(y)
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * (grad * grad)
        mh = m / (1 - 0.9**t)
        vh = v / (1 - 0.999**t)
        beta -= lr * mh / (np.sqrt(vh) + 1e-8)
    return beta


def predict_proba(beta, x):
    return sigmoid(np.column_stack([np.ones(len(x)), x]) @ beta)


def auc_roc(y, p):
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": round(float(threshold), 4), "accuracy": (tp + tn) / len(y),
        "precision_yes": precision, "recall_yes": recall, "f1_yes": f1,
        "roc_auc": auc_roc(y, p), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def run_variant(df, train_idx, valid_idx, test_idx, include_duration):
    train, valid, test = df.iloc[train_idx], df.iloc[valid_idx], df.iloc[test_idx]
    x_train, x_valid, x_test = transform(train, [valid, test], include_duration)
    y_train = (train["y"] == "yes").astype(int).to_numpy()
    y_valid = (valid["y"] == "yes").astype(int).to_numpy()
    y_test = (test["y"] == "yes").astype(int).to_numpy()
    beta = fit_logistic(x_train, y_train, class_weight=True)
    p_valid = predict_proba(beta, x_valid)
    thresholds = np.linspace(0.05, 0.90, 172)
    scores = [metrics(y_valid, p_valid, t) for t in thresholds]
    best = max(scores, key=lambda m: m["f1_yes"])
    p_test = predict_proba(beta, x_test)
    return {
        "validation_selected_threshold": best,
        "test_at_0_5": metrics(y_test, p_test, 0.5),
        "test_at_selected_threshold": metrics(y_test, p_test, best["threshold"]),
    }


def main():
    df = pd.read_csv(DATA)
    y = (df["y"] == "yes").astype(int).to_numpy()
    train_idx, valid_idx, test_idx = stratified_indices(y)
    report = {
        "rows": len(df), "positive_rate": float(y.mean()),
        "split_rows": {"train": len(train_idx), "validation": len(valid_idx), "test": len(test_idx)},
        "pre_call_model_without_duration": run_variant(df, train_idx, valid_idx, test_idx, False),
        "diagnostic_model_with_duration": run_variant(df, train_idx, valid_idx, test_idx, True),
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
