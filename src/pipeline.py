"""Run the whole pipeline end to end with one command.

    python -m src.pipeline

1. data       download + clean + engineered features + stratified 60/20/20 split
2. eda        churn-rate charts (training data only)
3. train      baseline vs. improved models, imbalance strategies, pick on cross-validation
4. threshold  business-value threshold on validation + sensitivity analysis
5. evaluate   one-time test evaluation, fairness check, explanations, figures

Fixed seeds everywhere, so a rerun gives the same numbers.
"""
import time

from src import data, eda, evaluate, threshold, train

STEPS = [("data", data.build), ("eda", eda.main), ("train", train.main),
         ("threshold", threshold.main), ("evaluate", evaluate.main)]


def main():
    t0 = time.time()
    for name, step in STEPS:
        t = time.time()
        print(f"\n===== {name} =====")
        step()
        print(f"----- {name} done in {time.time() - t:.0f}s")
    print(f"\nPipeline finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
