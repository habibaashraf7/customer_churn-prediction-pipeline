"""Baseline vs. improved churn models, with and without class-imbalance handling.

    python -m src.train

Every candidate is trained on the training split and compared on the validation split.
Model selection metric: 5-fold cross-validated PR-AUC (average precision), because only 26.5% of customers churn
and the retention team cares about how clean the top of the risk list is.
The chosen model is refit on train and saved to models/churn_model.joblib.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.data import CATEGORICAL, ENGINEERED, NUMERIC, load

ROOT = Path(__file__).resolve().parents[1]
MODELS, REPORTS = ROOT / "models", ROOT / "reports"
SEED = 42
FEATURES = CATEGORICAL + NUMERIC + ENGINEERED
# gender is kept out of the model on purpose (fairness); it is only used to check the model afterwards
FEATURES = [f for f in FEATURES if f != "gender"]
CATS = [c for c in CATEGORICAL if c != "gender"]
NUMS = NUMERIC + ENGINEERED


def linear_prep(nums=None):
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), CATS),
                              ("num", StandardScaler(), nums or NUMS)])


def tree_prep():
    return ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATS),
                              ("num", "passthrough", NUMS)], verbose_feature_names_out=False)


def hgb(**kw):
    return HistGradientBoostingClassifier(categorical_features=list(range(len(CATS))), random_state=SEED, **kw)


class RuleModel:
    """What a retention team might do without ML: call month-to-month customers in their first year."""
    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        s = ((X["Contract"] == "Month-to-month").astype(float) * 0.6 + (X["tenure"] <= 12).astype(float) * 0.3
             + (X["InternetService"] == "Fiber optic").astype(float) * 0.1)
        return np.c_[1 - s, s]


def candidates():
    """(name, family, imbalance handling, model)"""
    yield "Always 'stays'", "no model", "-", DummyClassifier(strategy="most_frequent")
    yield "Business rule", "no model", "-", RuleModel()
    # baseline: logistic regression on the raw columns only
    for imb, cw in [("none", None), ("class weights", "balanced")]:
        yield "Logistic regression, raw columns", "baseline", imb, Pipeline([
            ("prep", linear_prep(NUMERIC)), ("clf", LogisticRegression(max_iter=3000, class_weight=cw))])
    # improved: same model + engineered features (n_services, auto_pay, charge_vs_avg, new_customer, log_tenure)
    for imb, cw in [("none", None), ("class weights", "balanced")]:
        yield "Logistic regression + engineered features", "improved", imb, Pipeline([
            ("prep", linear_prep()), ("clf", LogisticRegression(max_iter=3000, class_weight=cw))])
    # resampling: SMOTE creates synthetic churners, undersampling drops some loyal customers (training folds only)
    yield "Logistic regression + engineered features", "improved", "SMOTE", ImbPipeline([
        ("prep", linear_prep()), ("smote", SMOTE(random_state=SEED)), ("clf", LogisticRegression(max_iter=3000))])
    yield "Logistic regression + engineered features", "improved", "undersampling", ImbPipeline([
        ("prep", linear_prep()), ("under", RandomUnderSampler(random_state=SEED)), ("clf", LogisticRegression(max_iter=3000))])
    yield "Random forest", "improved", "class weights", Pipeline([("prep", tree_prep()), ("clf", RandomForestClassifier(
        n_estimators=500, min_samples_leaf=10, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1))])
    for imb, kw in [("none", {}), ("class weights", {"class_weight": "balanced"})]:
        yield "Gradient boosting (tuned)", "improved", imb, Pipeline([("prep", tree_prep()), ("clf", hgb(
            learning_rate=0.03, max_iter=400, max_leaf_nodes=4, min_samples_leaf=50, l2_regularization=1.0,
            early_stopping=False, **kw))])


def scores(y, p, threshold=0.5):
    pred = (p >= threshold).astype(int)
    return {"pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p) if len(set(p)) > 1 else 0.5,
            "accuracy": accuracy_score(y, pred), "precision@0.5": precision_score(y, pred, zero_division=0),
            "recall@0.5": recall_score(y, pred), "f1@0.5": f1_score(y, pred)}


def main():
    MODELS.mkdir(exist_ok=True)
    train, val = load("train"), load("val")
    Xtr, ytr, Xva, yva = train[FEATURES], train["churn"], val[FEATURES], val["churn"]
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows, fitted = [], {}
    for name, family, imb, model in candidates():
        cv_pr = (cross_val_score(model, Xtr, ytr, cv=cv, scoring="average_precision").mean()
                 if family != "no model" else np.nan)
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xva)[:, 1]
        row = {"model": name, "family": family, "imbalance": imb, "cv_pr_auc": cv_pr, **scores(yva, p)}
        rows.append(row)
        fitted[(name, imb)] = model
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})

    res = pd.DataFrame(rows)
    res.to_csv(REPORTS / "model_comparison.csv", index=False)

    # choose on 5-fold CV PR-AUC (more stable than one 1,409-row validation set), validation as a check
    chosen = res[res["family"] != "no model"].sort_values("cv_pr_auc", ascending=False).iloc[0]
    model = fitted[(chosen["model"], chosen["imbalance"])]
    joblib.dump(model, MODELS / "churn_model.joblib")
    joblib.dump(fitted[("Logistic regression, raw columns", "none")], MODELS / "baseline_logreg.joblib")
    info = {"model": chosen["model"], "imbalance": chosen["imbalance"], "val_pr_auc": chosen["pr_auc"],
            "val_roc_auc": chosen["roc_auc"], "features": FEATURES}
    (MODELS / "model_info.json").write_text(json.dumps(info, indent=2))
    print("\nselected:", chosen["model"], "/", chosen["imbalance"], f"val PR-AUC {chosen['pr_auc']:.3f}")


if __name__ == "__main__":
    main()
