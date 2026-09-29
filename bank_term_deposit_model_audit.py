from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent
DATA = PROJECT_DIR / "train.csv"
OUT = PROJECT_DIR / "model_audit_results.json"
PLOTS_DIR = PROJECT_DIR / "images"

COLORS = {
    "navy": "#243B6B",
    "blue": "#4F6FBA",
    "teal": "#19A79B",
    "gold": "#F2C14E",
    "light": "#EAF0F8",
    "text": "#172033",
}


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


def style_axes(ax, title):
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", color=COLORS["text"], pad=14)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D7DFEA", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def add_bar_labels(ax, bars, formatter=lambda value: f"{value:.3f}"):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(ax.get_ylim()[1] * 0.015, 0.01),
            formatter(height),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=COLORS["text"],
        )


def plot_class_distribution(df):
    counts = df["y"].value_counts().reindex(["no", "yes"])
    labels = ["No subscription", "Subscription"]
    colors = [COLORS["navy"], COLORS["teal"]]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(labels, counts.to_numpy(), color=colors, width=0.58)
    style_axes(ax, "Target class distribution")
    ax.set_ylabel("Customers")
    ax.set_ylim(0, counts.max() * 1.18)
    total = counts.sum()
    add_bar_labels(ax, bars, lambda value: f"{int(value):,}\n({value / total:.1%})")
    fig.text(
        0.01,
        0.01,
        "The 88/12 class imbalance makes accuracy alone an unsuitable model-selection metric.",
        fontsize=10,
        color="#475467",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(PLOTS_DIR / "01_target_class_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(report):
    pre_call = report["pre_call_model_without_duration"]["test_at_selected_threshold"]
    diagnostic = report["diagnostic_model_with_duration"]["test_at_selected_threshold"]
    metric_keys = ["roc_auc", "precision_yes", "recall_yes", "f1_yes"]
    metric_labels = ["ROC AUC", "Precision", "Recall", "F1"]
    x = np.arange(len(metric_keys))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10, 5.8))
    pre_bars = ax.bar(
        x - width / 2,
        [pre_call[key] for key in metric_keys],
        width,
        label="Pre-call model (without duration)",
        color=COLORS["blue"],
    )
    diagnostic_bars = ax.bar(
        x + width / 2,
        [diagnostic[key] for key in metric_keys],
        width,
        label="Post-call diagnostic (with duration)",
        color=COLORS["teal"],
    )
    style_axes(ax, "Leakage-aware model comparison on the untouched test split")
    ax.set_xticks(x, metric_labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.legend(frameon=False, loc="upper center", ncols=2)
    add_bar_labels(ax, pre_bars)
    add_bar_labels(ax, diagnostic_bars)
    fig.text(
        0.01,
        0.01,
        "The diagnostic model is not deployable for pre-call targeting because duration is known only after a call.",
        fontsize=10,
        color="#475467",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(PLOTS_DIR / "02_model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(report):
    result = report["pre_call_model_without_duration"]["test_at_selected_threshold"]
    matrix = np.array([[result["tn"], result["fp"]], [result["fn"], result["tp"]]])

    fig, ax = plt.subplots(figsize=(7.5, 6))
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            value = matrix[row, column]
            color = "white" if value > matrix.max() * 0.52 else COLORS["text"]
            ax.text(column, row, f"{value:,}", ha="center", va="center", fontsize=18, fontweight="bold", color=color)
    ax.set_xticks([0, 1], ["Predicted No", "Predicted Yes"])
    ax.set_yticks([0, 1], ["Actual No", "Actual Yes"])
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Actual outcome")
    ax.set_title(
        f"Pre-call model confusion matrix (threshold = {result['threshold']:.4f})",
        fontsize=15,
        fontweight="bold",
        color=COLORS["text"],
        pad=16,
    )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Customers")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "03_pre_call_confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_tradeoff(report):
    model = report["pre_call_model_without_duration"]
    baseline = model["test_at_0_5"]
    selected = model["test_at_selected_threshold"]
    metric_keys = ["precision_yes", "recall_yes", "f1_yes"]
    metric_labels = ["Precision", "Recall", "F1"]
    x = np.arange(len(metric_keys))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9, 5.6))
    baseline_bars = ax.bar(
        x - width / 2,
        [baseline[key] for key in metric_keys],
        width,
        label="Default threshold (0.5000)",
        color=COLORS["gold"],
    )
    selected_bars = ax.bar(
        x + width / 2,
        [selected[key] for key in metric_keys],
        width,
        label=f"Validation-selected threshold ({selected['threshold']:.4f})",
        color=COLORS["navy"],
    )
    style_axes(ax, "Pre-call threshold trade-off")
    ax.set_xticks(x, metric_labels)
    ax.set_ylim(0, 0.72)
    ax.set_ylabel("Positive-class score")
    ax.legend(frameon=False, loc="upper center", ncols=2)
    add_bar_labels(ax, baseline_bars)
    add_bar_labels(ax, selected_bars)
    fig.text(
        0.01,
        0.01,
        "Raising the threshold improves precision and F1 while reducing recall; the final threshold should reflect campaign costs.",
        fontsize=10,
        color="#475467",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(PLOTS_DIR / "04_threshold_tradeoff.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_visualizations(df, report):
    PLOTS_DIR.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.labelcolor": COLORS["text"],
        "xtick.color": COLORS["text"],
        "ytick.color": COLORS["text"],
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    plot_class_distribution(df)
    plot_model_comparison(report)
    plot_confusion_matrix(report)
    plot_threshold_tradeoff(report)


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
    create_visualizations(df, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
