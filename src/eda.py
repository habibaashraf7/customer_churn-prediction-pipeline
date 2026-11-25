"""Exploratory analysis: churn rate by the main customer attributes.

    python -m src.eda       # figures in reports/figures/, numbers in reports/eda_summary.csv

Uses the training split only, so nothing about validation/test leaks into our choices.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data import load

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
NAVY, PINK, ROSE, SKY, BEIGE = "#2F4156", "#F7C9D4", "#D9738C", "#C8D9E6", "#F5EFEB"


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=11, color=NAVY, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.4, linestyle="--")
    ax.set_axisbelow(True)


def rate_by(d, col, order=None):
    r = d.groupby(col)["churn"].agg(["mean", "size"]).rename(columns={"mean": "churn_rate", "size": "customers"})
    return r.reindex(order) if order else r.sort_values("churn_rate")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    d = load("train")
    overall = d["churn"].mean()
    d["tenure_group"] = pd.cut(d["tenure"], [-1, 6, 12, 24, 48, 72], labels=["0-6 m", "7-12 m", "13-24 m", "25-48 m", "49-72 m"])
    d["charges_group"] = pd.cut(d["MonthlyCharges"], [0, 35, 70, 90, 200], labels=["< $35", "$35-70", "$70-90", "> $90"])

    panels = [("Contract", ["Two year", "One year", "Month-to-month"]),
              ("tenure_group", ["49-72 m", "25-48 m", "13-24 m", "7-12 m", "0-6 m"]),
              ("InternetService", ["No", "DSL", "Fiber optic"]),
              ("PaymentMethod", None),
              ("charges_group", ["< $35", "$35-70", "$70-90", "> $90"]),
              ("TechSupport", ["Yes", "No internet service", "No"])]
    titles = {"tenure_group": "Tenure", "charges_group": "Monthly charges", "InternetService": "Internet service",
              "PaymentMethod": "Payment method", "TechSupport": "Tech support", "Contract": "Contract"}
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    for ax, (col, order) in zip(axes.flat, panels):
        r = rate_by(d, col, order)
        colors = [ROSE if v > overall else SKY for v in r["churn_rate"]]
        ax.barh(r.index.astype(str), r["churn_rate"], color=colors)
        for y, (v, n) in enumerate(zip(r["churn_rate"], r["customers"])):
            ax.text(v + 0.01, y, f"{v:.0%}  (n={n})", va="center", fontsize=9, color=NAVY)
        ax.axvline(overall, color=NAVY, lw=1, ls=":")
        ax.set_xlim(0, 0.75)
        style(ax, f"Churn rate by {titles[col].lower()}")
        rows += [{"feature": titles[col], "value": str(k), "churn_rate": v, "customers": n}
                 for k, v, n in zip(r.index, r["churn_rate"], r["customers"])]
    fig.suptitle(f"Who cancels? (training data, overall churn {overall:.1%}, dotted line)", x=0.01, ha="left",
                 fontsize=13, color=NAVY, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "eda_churn_rates.png", dpi=130)
    plt.close(fig)

    # tenure and charges distributions
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, col, label in [(axes[0], "tenure", "tenure (months)"), (axes[1], "MonthlyCharges", "monthly charges ($)")]:
        ax.hist(d.loc[d.churn == 0, col], bins=30, alpha=0.85, color=SKY, label="stayed")
        ax.hist(d.loc[d.churn == 1, col], bins=30, alpha=0.85, color=ROSE, label="churned")
        ax.set_xlabel(label)
        style(ax, f"Distribution of {label}")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "eda_distributions.png", dpi=130)
    plt.close(fig)

    summary = pd.DataFrame(rows)
    summary.to_csv(ROOT / "reports" / "eda_summary.csv", index=False)
    print(f"overall churn {overall:.3f}  n={len(d)}")
    print(summary.round(3).to_string(index=False))
    print(d.groupby("churn")[["tenure", "MonthlyCharges", "TotalCharges", "n_services", "charge_vs_avg"]].median())


if __name__ == "__main__":
    main()
