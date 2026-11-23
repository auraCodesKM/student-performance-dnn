"""Compare the tuned DNN against baselines with the same cross-validation, then score the locked test set once.

Run:  python -m src.evaluate        (after python -m src.tune)
Writes results/RESULTS.md, results/metrics.json, figures, and the two app models in models/.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # avoids a libomp crash when xgboost and torch share a process on macOS

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, balanced_accuracy_score,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data import SEED, load_data, make_preprocessor, split
from src.tune import build_dnn

FIG = "results/figures"
SCORING = {"accuracy": "accuracy", "balanced_acc": "balanced_accuracy", "f1_macro": "f1_macro",
           "recall_at_risk": "recall", "roc_auc": "roc_auc"}


class G2Rule(ClassifierMixin, BaseEstimator):
    """Naive baseline from Cortez & Silva (2008): predict 'at risk' if the 2nd-period grade G2 < 10.
    The late-term DNN is only worth having if it beats this one-line rule."""

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = (20 - X["G2"].to_numpy()) / 20  # lower G2 -> higher risk score (used for ROC-AUC)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (X["G2"].to_numpy() < 10).astype(int)


def tuned_baselines(scenario, train):
    """Traditional models, each given a small grid search with the same CV and metric as the DNN got, for a fair fight."""
    def pipe(model):
        return Pipeline(make_preprocessor(scenario) + [("model", model)])

    grids = {
        "Logistic regression": (pipe(LogisticRegression(max_iter=2000)),
                                {"model__C": [0.01, 0.1, 1, 10], "model__class_weight": [None, "balanced"],
                                 "select__k": [10, 20, "all"]}),
        "Random forest": (pipe(RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)),
                          {"model__max_depth": [None, 6], "model__min_samples_leaf": [1, 5],
                           "model__class_weight": [None, "balanced_subsample"]}),
        "XGBoost": (pipe(XGBClassifier(learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=SEED)),
                    {"model__max_depth": [2, 3, 4], "model__n_estimators": [200, 400],
                     "model__scale_pos_weight": [1, 5.5]}),  # 5.5 = #pass / #fail
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    models = {"Majority class": DummyClassifier(strategy="most_frequent")}
    if scenario == "late":
        models["Naive rule (G2 < 10)"] = G2Rule()
    for name, (estimator, grid) in grids.items():
        search = GridSearchCV(estimator, grid, cv=cv, scoring="f1_macro", n_jobs=-1).fit(train, train["at_risk"])
        models[name] = search.best_estimator_
        print(f"  {name}: {search.best_params_}")
    return models


def test_metrics(model, test):
    y, p = test["at_risk"], model.predict_proba(test)[:, 1]
    yhat = model.predict(test)
    return {"accuracy": accuracy_score(y, yhat), "balanced_acc": balanced_accuracy_score(y, yhat),
            "precision_at_risk": precision_score(y, yhat, zero_division=0), "recall_at_risk": recall_score(y, yhat),
            "f1_at_risk": f1_score(y, yhat), "f1_macro": f1_score(y, yhat, average="macro"), "roc_auc": roc_auc_score(y, p)}


def bootstrap_ci(model, test, n=1000):
    """95% confidence interval for test accuracy and macro-F1: resample the 130 test students with replacement."""
    rng, y, yhat = np.random.default_rng(SEED), test["at_risk"].to_numpy(), model.predict(test)
    stats = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        stats.append([accuracy_score(y[i], yhat[i]), f1_score(y[i], yhat[i], average="macro")])
    lo, hi = np.percentile(stats, [2.5, 97.5], axis=0)
    return {"accuracy": [lo[0], hi[0]], "f1_macro": [lo[1], hi[1]]}


def evaluate_scenario(scenario, params, train, test):
    print(f"\n=== {scenario} scenario ===")
    models = tuned_baselines(scenario, train)
    models["DNN (MLP)"] = build_dnn(params, scenario)

    # Same repeated CV for every model (different seed from tuning, so scores are not the tuning scores re-used).
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=7)
    rows = {}
    for name, model in models.items():
        s = cross_validate(model, train, train["at_risk"], cv=cv, scoring=SCORING, n_jobs=-1)
        rows[name] = {f"cv_{k}": s[f"test_{k}"].mean() for k in SCORING} | {"cv_f1_macro_std": s["test_f1_macro"].std()}
        models[name] = model.fit(train, train["at_risk"])  # refit on all training rows
        rows[name] |= {f"test_{k}": v for k, v in test_metrics(models[name], test).items()}
        print(f"  {name:22s} CV macro-F1 {rows[name]['cv_f1_macro']:.3f} | test acc {rows[name]['test_accuracy']:.3f}")

    # The DNN depends on its random initialisation, so report test scores over 5 seeds as well.
    seeds = [test_metrics(build_dnn(params, scenario).set_params(model__seed=100 * s).fit(train, train["at_risk"]), test)
             for s in range(5)]
    dnn = models["DNN (MLP)"]
    extra = {"seed_mean": pd.DataFrame(seeds).mean().to_dict(), "seed_std": pd.DataFrame(seeds).std().to_dict(),
             "bootstrap_95ci": bootstrap_ci(dnn, test), "best_epoch": dnn.named_steps["model"].best_epoch_,
             "selected_features": list(dnn.named_steps["select"].get_feature_names_out()), "params": params}
    joblib.dump(dnn, f"models/dnn_{scenario}.joblib")
    plots(scenario, models, test)
    return rows, extra


def plots(scenario, models, test):
    dnn = models["DNN (MLP)"]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(dnn, test, test["at_risk"], display_labels=["Pass", "At risk"], cmap="Blues", ax=ax)
    ax.set_title(f"DNN confusion matrix ({scenario}, test set)")
    fig.tight_layout(); fig.savefig(f"{FIG}/confusion_{scenario}.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, model in models.items():
        if name != "Majority class":
            RocCurveDisplay.from_estimator(model, test, test["at_risk"], name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "--", color="grey"); ax.set_title(f"ROC curves ({scenario}, test set)")
    fig.tight_layout(); fig.savefig(f"{FIG}/roc_{scenario}.png", dpi=150); plt.close(fig)

    h = dnn.named_steps["model"].history_
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(h["train"], label="train loss"); ax.plot(h["val"], label="validation loss")
    ax.axvline(dnn.named_steps["model"].best_epoch_ - 1, color="grey", ls="--", label="best epoch (weights kept)")
    ax.set(xlabel="Epoch", ylabel="Binary cross-entropy loss", title=f"DNN learning curve ({scenario})"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{FIG}/learning_curve_{scenario}.png", dpi=150); plt.close(fig)


def write_report(results):
    lines = ["# Results", "",
             "Portuguese dataset, target = at risk (G3 < 10). 519 training students, 130 locked test students.",
             "CV = 5-fold x 3 repeats on the training set. Test = scored once.", ""]
    for scenario, (rows, extra) in results.items():
        label = "Late-term (with G1, G2)" if scenario == "late" else "Early-warning (no grades)"
        lines += [f"## {label}", "",
                  "| Model | CV macro-F1 | CV accuracy | CV at-risk recall | Test accuracy | Test macro-F1 | Test at-risk recall | Test ROC-AUC |",
                  "|---|---|---|---|---|---|---|---|"]
        for name, r in rows.items():
            lines.append(f"| {name} | {r['cv_f1_macro']:.3f} ± {r['cv_f1_macro_std']:.3f} | {r['cv_accuracy']:.3f} | "
                         f"{r['cv_recall_at_risk']:.3f} | {r['test_accuracy']:.3f} | {r['test_f1_macro']:.3f} | "
                         f"{r['test_recall_at_risk']:.3f} | {r['test_roc_auc']:.3f} |")
        m, s, ci = extra["seed_mean"], extra["seed_std"], extra["bootstrap_95ci"]
        lines += ["", f"DNN over 5 seeds (test): accuracy {m['accuracy']:.3f} ± {s['accuracy']:.3f}, "
                      f"macro-F1 {m['f1_macro']:.3f} ± {s['f1_macro']:.3f}.",
                  f"Bootstrap 95% CI (saved model): accuracy {ci['accuracy'][0]:.3f}–{ci['accuracy'][1]:.3f}, "
                  f"macro-F1 {ci['f1_macro'][0]:.3f}–{ci['f1_macro'][1]:.3f}.",
                  f"Selected features ({len(extra['selected_features'])}): {', '.join(extra['selected_features'])}.",
                  f"Hyperparameters: `{extra['params']}`; best epoch {extra['best_epoch']}.", "",
                  f"![confusion](figures/confusion_{scenario}.png) ![roc](figures/roc_{scenario}.png)",
                  f"![learning curve](figures/learning_curve_{scenario}.png)", ""]
    with open("results/RESULTS.md", "w") as f:
        f.write("\n".join(lines))


def main():
    train, test = split(load_data())
    with open("results/best_params.json") as f:
        best = json.load(f)
    results = {s: evaluate_scenario(s, best[s], train, test) for s in ["late", "early"]}
    with open("results/metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    write_report(results)
    print("\nWrote results/RESULTS.md and models/dnn_late.joblib, models/dnn_early.joblib")


if __name__ == "__main__":
    main()
