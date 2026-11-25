"""Final evaluation on the test split (used once, after every choice was made on validation).

    python -m src.evaluate

Writes reports/test_results.json, test_policies.csv, fairness.csv, feature_importance.csv,
test_scores.csv and figures.
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score)

from src.data import load
from src.explain import Explainer
from src.threshold import assumptions, curve, evaluate_policy, policies
from src.train import FEATURES

ROOT = Path(__file__).resolve().parents[1]
MODELS, REPORTS, FIG = ROOT / "models", ROOT / "reports", ROOT / "reports" / "figures"
NAVY, ROSE, PINK, SKY, STEEL, BEIGE = "#2F4156", "#D9738C", "#F7C9D4", "#C8D9E6", "#8EA8C0", "#F5EFEB"


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=11, color=NAVY, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    model, base = joblib.load(MODELS / "churn_model.joblib"), joblib.load(MODELS / "baseline_logreg.joblib")
    t = json.loads((MODELS / "threshold.json").read_text())["threshold"]
    train, test = load("train"), load("test")
    y = test["churn"].values
    s, sb = model.predict_proba(test[FEATURES])[:, 1], base.predict_proba(test[FEATURES])[:, 1]
    a = assumptions()

    res = {"threshold": t,
           "improved": {"pr_auc": average_precision_score(y, s), "roc_auc": roc_auc_score(y, s)},
           "baseline": {"pr_auc": average_precision_score(y, sb), "roc_auc": roc_auc_score(y, sb)},
           "at_threshold": evaluate_policy(test, s, t, a),
           "confusion_matrix": confusion_matrix(y, s >= t).tolist()}
    pol = policies(test, s, a, t)
    pol.to_csv(REPORTS / "test_policies.csv", index=False)

    # paired bootstrap: improved - baseline PR-AUC
    rng, diffs = np.random.default_rng(0), []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum():
            diffs.append(average_precision_score(y[i], s[i]) - average_precision_score(y[i], sb[i]))
    res["pr_auc_diff"] = {"diff": res["improved"]["pr_auc"] - res["baseline"]["pr_auc"],
                          "ci95": list(np.percentile(diffs, [2.5, 97.5]))}

    # lift: share of churners found when contacting the top x% riskiest
    order = np.argsort(-s)
    cum = np.cumsum(y[order]) / y.sum()
    res["gains"] = {f"top_{p}%": float(cum[int(len(y) * p / 100) - 1]) for p in [10, 20, 30, 40, 50]}

    # fairness: does the model treat groups similarly at the chosen threshold?
    rows = []
    test = test.assign(score=s, flagged=(s >= t).astype(int))
    for col, labels in [("gender", None), ("SeniorCitizen", {0: "Not senior", 1: "Senior"}),
                        ("Partner", None), ("Dependents", None)]:
        for v, g in test.groupby(col):
            churners = g[g.churn == 1]
            rows.append({"attribute": col, "group": labels[v] if labels else v, "customers": len(g),
                         "churn_rate": g.churn.mean(), "flagged_rate": g.flagged.mean(),
                         "recall": churners.flagged.mean(),
                         "precision": g[g.flagged == 1].churn.mean() if g.flagged.sum() else np.nan})
    fair = pd.DataFrame(rows)
    fair.to_csv(REPORTS / "fairness.csv", index=False)

    ex = Explainer(model, train)
    imp = ex.importance(test)
    imp.rename("mean_abs_contribution").to_csv(REPORTS / "feature_importance.csv")
    test[["customerID", "churn", "score", "flagged"]].to_csv(REPORTS / "test_scores.csv", index=False)
    (REPORTS / "test_results.json").write_text(json.dumps(res, indent=2, default=float))

    # ---------- figures
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for p, label, color in [(sb, "Baseline (raw columns)", STEEL), (s, "Improved (+ engineered features)", NAVY)]:
        pr, rc, _ = precision_recall_curve(y, p)
        axes[0].plot(rc, pr, color=color, lw=2, label=f"{label}  PR-AUC {average_precision_score(y, p):.3f}")
    m = evaluate_policy(test, s, t, a)
    axes[0].scatter([m["recall"]], [m["precision"]], color=ROSE, s=70, zorder=5, label=f"chosen threshold {t}")
    axes[0].axhline(y.mean(), color=STEEL, ls=":", lw=1)
    axes[0].set_xlabel("recall (share of churners found)")
    axes[0].set_ylabel("precision (share of contacts who churn)")
    axes[0].legend(frameon=False, fontsize=8)
    style(axes[0], "Precision vs. recall (test)")

    c = curve(test, s, a)
    axes[1].plot(c["threshold"], c["net_value"] / 1000, color=NAVY, lw=2)
    axes[1].axvline(t, color=ROSE, lw=2, label=f"chosen {t}")
    axes[1].axvline(0.5, color=STEEL, ls=":", label="default 0.5")
    axes[1].set_xlabel("threshold")
    axes[1].set_ylabel("net value ($K, test customers)")
    axes[1].legend(frameon=False, fontsize=8)
    style(axes[1], "Net value of the retention campaign")

    xs = np.arange(1, len(y) + 1) / len(y) * 100
    axes[2].plot(xs, cum * 100, color=NAVY, lw=2, label="model")
    axes[2].plot([0, 100], [0, 100], color=STEEL, ls=":", label="random")
    axes[2].set_xlabel("% of customers contacted (riskiest first)")
    axes[2].set_ylabel("% of churners reached")
    axes[2].legend(frameon=False, fontsize=8)
    style(axes[2], "Gains: who to call first")
    fig.tight_layout()
    fig.savefig(FIG / "test_evaluation.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.2))
    top = imp.head(12)[::-1]
    ax.barh(top.index, top.values, color=[ROSE if i >= len(top) - 3 else STEEL for i in range(len(top))])
    ax.set_xlabel("average impact on churn score (|log-odds|)")
    style(ax, "What drives churn risk")
    fig.tight_layout()
    fig.savefig(FIG / "feature_importance.png", dpi=130)
    plt.close(fig)

    cm = np.array(res["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    from matplotlib.colors import LinearSegmentedColormap
    ax.imshow(cm / cm.sum(1, keepdims=True), cmap=LinearSegmentedColormap.from_list("c", [SKY, STEEL, NAVY]), vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n{cm[i, j] / cm[i].sum():.0%}", ha="center", va="center", color="white", fontsize=12)
    ax.set_xticks([0, 1], ["not contacted", "contacted"])
    ax.set_yticks([0, 1], ["stayed", "churned"])
    ax.set_title(f"Test confusion matrix at threshold {t}", color=NAVY, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "confusion_matrix.png", dpi=130)
    plt.close(fig)

    print(json.dumps({k: v for k, v in res.items() if k != "at_threshold"}, indent=1, default=float))
    print(pd.Series(res["at_threshold"]).round(3).to_string())
    print(pol.round(3).to_string(index=False))
    print(fair.round(3).to_string(index=False))
    print(imp.round(3).head(12).to_string())
    for _, r in test.sort_values("score", ascending=False).head(3).iterrows():
        print(r["customerID"], round(r["score"], 2), ex.reasons(r), ex.protective(r))


if __name__ == "__main__":
    main()
