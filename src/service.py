"""Small Python API used by the Streamlit app, the FastAPI endpoint and the notebook.

    from src.service import ChurnService
    svc = ChurnService()
    svc.score_customer({...})          # one customer as a dict of raw columns
"""
import json
from functools import cached_property
from pathlib import Path

import joblib
import pandas as pd

from src.data import add_features, load
from src.explain import Explainer
from src.threshold import assumptions, customer_economics
from src.train import FEATURES

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def risk_band(p: float, threshold: float) -> str:
    if p >= max(0.6, threshold + 0.25):
        return "High"
    if p >= threshold:
        return "Medium"
    return "Low"


class ChurnService:
    @cached_property
    def model(self):
        return joblib.load(MODELS / "churn_model.joblib")

    @cached_property
    def baseline(self):
        return joblib.load(MODELS / "baseline_logreg.joblib")

    @cached_property
    def config(self):
        return json.loads((MODELS / "threshold.json").read_text())

    @property
    def threshold(self) -> float:
        return self.config["threshold"]

    @cached_property
    def explainer(self):
        return Explainer(self.model, load("train"))

    @cached_property
    def customers(self) -> pd.DataFrame:
        """Validation + test customers, scored: they play the role of 'this month's customers' in the demo."""
        d = pd.concat([load("val"), load("test")], ignore_index=True)
        d["score"] = self.model.predict_proba(d[FEATURES])[:, 1]
        return d.sort_values("score", ascending=False).reset_index(drop=True)

    def score(self, df: pd.DataFrame):
        return self.model.predict_proba(df[FEATURES])[:, 1]

    def score_customer(self, raw: dict, threshold: float | None = None) -> dict:
        t = self.threshold if threshold is None else threshold
        row = add_features(pd.DataFrame([raw])).iloc[0]
        p = float(self.score(row.to_frame().T)[0])
        eco = customer_economics(row.to_frame().T, assumptions()).iloc[0]
        return {"churn_probability": round(p, 4), "risk_band": risk_band(p, t), "contact": p >= t,
                "threshold": t, "reasons": self.explainer.reasons(row), "protective": self.explainer.protective(row),
                "offer_cost": round(float(eco["offer_cost"]), 2),
                "expected_value_if_contacted": round(float(p * eco["save_value"] - eco["offer_cost"]), 2)}
