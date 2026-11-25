"""Load, clean and split the IBM Telco Customer Churn data.

    python -m src.data      # downloads the CSV if needed, writes data/processed/{train,val,test}.parquet

Split: 60% train / 20% validation / 20% test, stratified on churn, fixed seed.
The dataset is a single snapshot (no dates), so a random stratified split is the best we can do;
validation is used for every choice (model, threshold), test only once at the end.
"""
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "telco_churn.csv"
OUT = ROOT / "data" / "processed"
URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
SEED = 42

SERVICES = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"]
CATEGORICAL = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
               "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
               "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]
NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
ENGINEERED = ["n_services", "auto_pay", "charge_vs_avg", "new_customer", "log_tenure"]


def load_raw() -> pd.DataFrame:
    if not RAW.exists():
        RAW.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(URL, RAW)
    return pd.read_csv(RAW)


def clean(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    # 11 customers with tenure 0 have a blank TotalCharges: they haven't been billed yet, so 0 is correct
    d["TotalCharges"] = pd.to_numeric(d["TotalCharges"].astype(str).str.strip().replace("", "0"))
    d["churn"] = (d.pop("Churn") == "Yes").astype(int)
    return add_features(d)


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    """Simple, explainable features. All use information the company has today."""
    d = d.copy()
    d["n_services"] = sum((d[c] == "Yes").astype(int) for c in SERVICES) + (d["InternetService"] != "No").astype(int)
    d["auto_pay"] = d["PaymentMethod"].str.contains("automatic").astype(int)
    # current bill vs. the customer's average bill so far (>1 means the price went up recently)
    avg = np.where(d["tenure"] > 0, d["TotalCharges"] / d["tenure"].clip(lower=1), d["MonthlyCharges"])
    d["charge_vs_avg"] = (d["MonthlyCharges"] / np.maximum(avg, 1)).round(3)
    d["new_customer"] = (d["tenure"] <= 6).astype(int)
    d["log_tenure"] = np.log1p(d["tenure"])   # churn risk falls fastest in the first months
    return d


def split(d: pd.DataFrame):
    train, rest = train_test_split(d, test_size=0.4, stratify=d["churn"], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["churn"], random_state=SEED)
    return train, val, test


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    d = clean(load_raw())
    for name, part in zip(["train", "val", "test"], split(d)):
        part.reset_index(drop=True).to_parquet(OUT / f"{name}.parquet", index=False)
        print(f"{name:5} {len(part):5} rows, churn {part['churn'].mean():.1%}")
    d.to_parquet(OUT / "all.parquet", index=False)


def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(OUT / f"{name}.parquet")


if __name__ == "__main__":
    build()
