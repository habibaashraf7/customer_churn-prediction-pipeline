"""Pick the decision threshold from a simple retention business case.

    python -m src.threshold

Every customer whose churn score is above the threshold gets a retention offer.

Assumptions (all editable in the app):
  * contacting a customer costs CONTACT_COST (agent time)
  * the offer is a discount: OFFER_DISCOUNT off the bill for OFFER_MONTHS months.
    Everyone who is contacted can accept it, churner or not (that is the cost of a false alarm)
  * an offer stops a real churner from leaving with probability SAVE_RATE
  * a saved customer is worth VALUE_MONTHS more months of their monthly bill

Net value of a threshold = (saved churners x their value) - (contact + discount for everyone contacted).
The threshold is chosen on validation data, then checked once on test.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from src.data import load
from src.train import FEATURES

ROOT = Path(__file__).resolve().parents[1]
MODELS, REPORTS = ROOT / "models", ROOT / "reports"

CONTACT_COST = 5.0       # $ per customer contacted
OFFER_DISCOUNT = 0.20    # 20% off ...
OFFER_MONTHS = 3         # ... for 3 months
SAVE_RATE = 0.30         # share of real churners the offer keeps
VALUE_MONTHS = 12        # months of revenue a saved customer is worth


def assumptions(**kw):
    a = dict(contact_cost=CONTACT_COST, offer_discount=OFFER_DISCOUNT, offer_months=OFFER_MONTHS,
             save_rate=SAVE_RATE, value_months=VALUE_MONTHS)
    a.update(kw)
    return a


def customer_economics(df: pd.DataFrame, a: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["offer_cost"] = a["contact_cost"] + a["offer_discount"] * a["offer_months"] * df["MonthlyCharges"]
    out["save_value"] = a["save_rate"] * a["value_months"] * df["MonthlyCharges"]
    return out


def evaluate_policy(df, scores, threshold, a):
    """Outcome of contacting every customer with score >= threshold."""
    flag = scores >= threshold
    eco = customer_economics(df, a)
    y = df["churn"].values.astype(bool)
    gain = eco["save_value"][flag & y].sum()
    cost = eco["offer_cost"][flag].sum()
    return {"threshold": threshold, "contacted": int(flag.sum()), "contacted_share": flag.mean(),
            "precision": precision_score(y, flag, zero_division=0), "recall": recall_score(y, flag),
            "f1": f1_score(y, flag), "churners_caught": int((flag & y).sum()),
            "expected_saved": (flag & y).sum() * a["save_rate"],
            "value_saved": gain, "campaign_cost": cost, "net_value": gain - cost}


def curve(df, scores, a, grid=None):
    grid = np.round(np.arange(0.02, 0.96, 0.01), 2) if grid is None else grid
    return pd.DataFrame([evaluate_policy(df, scores, t, a) for t in grid])


NEAR_BEST = 0.98   # accept up to 2% less value for a smaller, more precise campaign


def best_threshold(df, scores, a):
    c = curve(df, scores, a)
    return float(c.loc[c["net_value"].idxmax(), "threshold"]), c


def recommended_threshold(df, scores, a):
    """The net-value curve is flat near its peak, so take the HIGHEST threshold that still earns
    at least 98% of the best net value: nearly the same money, fewer customers contacted."""
    c = curve(df, scores, a)
    peak = c["net_value"].max()
    if peak <= 0:
        return 1.01, c
    return float(c.loc[c["net_value"] >= NEAR_BEST * peak, "threshold"].max()), c


def policies(df, scores, a, chosen):
    c = curve(df, scores, a)
    peak_t = float(c.loc[c["net_value"].idxmax(), "threshold"])
    best_f1 = float(c.loc[c["f1"].idxmax(), "threshold"])
    top20 = float(np.quantile(scores, 0.80))
    rows = [("Contact nobody", 1.01), ("Contact everyone", 0.0), ("Default threshold 0.5", 0.5),
            ("Best F1", best_f1), ("Top 20% riskiest (capacity)", top20), ("Highest net value", peak_t),
            ("Recommended (98% of best value, fewest contacts)", chosen)]
    return pd.DataFrame([{"policy": n, **evaluate_policy(df, scores, t, a)} for n, t in rows])


def sensitivity(df, scores):
    rows = []
    for save_rate in [0.15, 0.30, 0.50]:
        for value_months in [6, 12, 24]:
            for discount in [0.10, 0.20, 0.30]:
                a = assumptions(save_rate=save_rate, value_months=value_months, offer_discount=discount)
                t, c = recommended_threshold(df, scores, a)
                r = c.set_index("threshold").loc[t] if t <= 1 else pd.Series({"contacted_share": 0, "recall": 0, "net_value": 0})
                rows.append({"save_rate": save_rate, "value_months": value_months, "discount": discount,
                             "best_threshold": t, "contacted_share": r["contacted_share"],
                             "recall": r["recall"], "net_value": r["net_value"]})
    return pd.DataFrame(rows)


def main():
    model = joblib.load(MODELS / "churn_model.joblib")
    val = load("val")
    s = model.predict_proba(val[FEATURES])[:, 1]
    a = assumptions()
    t, c = recommended_threshold(val, s, a)
    c.to_csv(REPORTS / "threshold_curve_val.csv", index=False)
    pol = policies(val, s, a, t)
    pol.to_csv(REPORTS / "threshold_policies_val.csv", index=False)
    sens = sensitivity(val, s)
    sens.to_csv(REPORTS / "threshold_sensitivity.csv", index=False)
    (MODELS / "threshold.json").write_text(json.dumps({"threshold": t, "assumptions": a}, indent=2))
    print(pol.round(3).to_string(index=False))
    print(f"\nchosen threshold {t}")
    print("sensitivity: best threshold range", sens["best_threshold"].min(), "-", sens["best_threshold"].max())
    print(sens.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
