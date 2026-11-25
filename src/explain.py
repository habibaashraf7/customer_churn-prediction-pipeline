"""Explain the churn model: global feature importance and per-customer reasons.

The chosen model is a logistic regression, so its SHAP values are exact and simple:
contribution of feature j = coefficient_j x (value_j - average value_j)   (in log-odds).
One-hot columns are summed back to the original column (e.g. all Contract_* -> "Contract"),
so each customer gets readable reasons like "Month-to-month contract".
"""
import numpy as np
import pandas as pd

from src.train import CATS, FEATURES, NUMS

PRETTY = {"Contract": "Contract", "tenure": "Tenure", "log_tenure": "Tenure", "new_customer": "Tenure",
          "InternetService": "Internet service", "MonthlyCharges": "Monthly charges", "TotalCharges": "Total charges",
          "PaymentMethod": "Payment method", "auto_pay": "Payment method", "TechSupport": "Tech support",
          "OnlineSecurity": "Online security", "OnlineBackup": "Online backup", "DeviceProtection": "Device protection",
          "PaperlessBilling": "Paperless billing", "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming movies",
          "MultipleLines": "Multiple lines", "PhoneService": "Phone service", "SeniorCitizen": "Senior citizen",
          "Partner": "Partner", "Dependents": "Dependents", "n_services": "Number of services",
          "charge_vs_avg": "Recent price change"}


class Explainer:
    def __init__(self, model, background: pd.DataFrame):
        self.model = model
        self.prep = model.named_steps["prep"]
        self.coef = model.named_steps["clf"].coef_[0]
        self.names = self.prep.get_feature_names_out()
        self.mean = np.asarray(self.prep.transform(background[FEATURES]).mean(axis=0)).ravel()
        self.groups = np.array([self._group(n) for n in self.names])

    @staticmethod
    def _group(name):
        kind, col = name.split("__", 1)
        if kind == "cat":
            for c in sorted(CATS, key=len, reverse=True):
                if col.startswith(c + "_"):
                    return PRETTY.get(c, c)
        return PRETTY.get(col, col)

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Rows = customers, columns = readable feature groups, values = log-odds contributions."""
        Z = self.prep.transform(X[FEATURES])
        Z = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)
        raw = (Z - self.mean) * self.coef
        return pd.DataFrame(raw, columns=self.groups, index=X.index).T.groupby(level=0).sum().T

    def importance(self, X: pd.DataFrame) -> pd.Series:
        return self.contributions(X).abs().mean().sort_values(ascending=False)

    def reasons(self, row: pd.Series, top=3):
        """Plain-language reasons that push this customer's risk UP, strongest first."""
        c = self.contributions(row.to_frame().T).iloc[0].sort_values(ascending=False)
        out = []
        for group, v in c.items():
            if v <= 0.05 or len(out) >= top:
                break
            out.append(describe(group, row))
        return out

    def protective(self, row: pd.Series, top=2):
        c = self.contributions(row.to_frame().T).iloc[0].sort_values()
        return [describe(g, row) for g, v in c.items() if v < -0.05][:top]


def describe(group, r):
    if group == "Contract":
        return f"{r['Contract']} contract"
    if group == "Tenure":
        return f"Customer for {int(r['tenure'])} month{'s' if r['tenure'] != 1 else ''}"
    if group == "Internet service":
        return f"Internet: {r['InternetService']}"
    if group == "Monthly charges":
        return f"Pays ${r['MonthlyCharges']:.0f} a month"
    if group == "Total charges":
        return f"Total paid so far ${r['TotalCharges']:,.0f}"
    if group == "Payment method":
        return f"Pays by {r['PaymentMethod'].lower()}"
    if group in ("Tech support", "Online security", "Online backup", "Device protection"):
        col = {"Tech support": "TechSupport", "Online security": "OnlineSecurity", "Online backup": "OnlineBackup",
               "Device protection": "DeviceProtection"}[group]
        return f"{group}: {r[col].lower()}"
    if group == "Paperless billing":
        return f"Paperless billing: {r['PaperlessBilling'].lower()}"
    if group == "Number of services":
        return f"Uses {int(r['n_services'])} services"
    if group == "Recent price change":
        return f"Current bill is {r['charge_vs_avg']:.2f}x their average"
    if group == "Senior citizen":
        return "Senior citizen" if r["SeniorCitizen"] else "Not a senior citizen"
    col = {"Streaming TV": "StreamingTV", "Streaming movies": "StreamingMovies", "Multiple lines": "MultipleLines",
           "Phone service": "PhoneService", "Partner": "Partner", "Dependents": "Dependents"}.get(group)
    return f"{group}: {str(r[col]).lower()}" if col else group
