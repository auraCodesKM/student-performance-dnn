"""Compare the tuned DNN against traditional baselines with the same cross-validation, then score the locked test set once.

Run:  python -m src.evaluate        (after python -m src.tune; about 1 minute)
Writes results/RESULTS.md, results/metrics.json, the figures, and models/dnn.joblib for the app.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # avoids a libomp crash when xgboost and torch share a process on macOS

import json
from math import comb, sqrt

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
                             classification_report, confusion_matrix, f1_score, precision_score, recall_score,
                             make_scorer, roc_auc_score)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data import SEED, load_data, make_preprocessor, split
from src.tune import build_dnn

FIG = "results/figures"
DNN = "DNN (MLP)"
SCORING = {"accuracy": "accuracy", "balanced_acc": "balanced_accuracy",
           "precision_at_risk": make_scorer(precision_score, zero_division=0),  # majority class never predicts "at risk": 0/0 -> 0
           "recall_at_risk": "recall", "f1_macro": "f1_macro", "roc_auc": "roc_auc"}


class G2Rule(ClassifierMixin, BaseEstimator):
    """Naive baseline from Cortez & Silva (2008): predict 'at risk' if the 2nd-period grade G2 < 10.
    The DNN is only worth having if it can match or beat this one-line rule."""

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = (20 - X["G2"].to_numpy()) / 20  # lower G2 -> higher risk score (used for ROC-AUC)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (X["G2"].to_numpy() < 10).astype(int)


def tuned_baselines(train):
    """Traditional models, each given a grid search with the same CV and metric the DNN was tuned with, for a fair fight."""
    def pipe(model):
        return Pipeline(make_preprocessor() + [("model", model)])

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
    models = {"Majority class": DummyClassifier(strategy="most_frequent"), "Naive rule (G2 < 10)": G2Rule()}
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
    """95% interval for test accuracy and macro-F1: resample the 130 test students with replacement 1000 times."""
    rng, y, yhat = np.random.default_rng(SEED), test["at_risk"].to_numpy(), model.predict(test)
    stats = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        stats.append([accuracy_score(y[i], yhat[i]), f1_score(y[i], yhat[i], average="macro")])
    lo, hi = np.percentile(stats, [2.5, 97.5], axis=0)
    return {"accuracy": [lo[0], hi[0]], "f1_macro": [lo[1], hi[1]]}


def wilson_ci(correct, n, z=1.96):
    """95% Wilson score interval for a proportion (better than p ± 1.96·SE when n is small)."""
    p = correct / n
    centre, half = (p + z**2 / (2 * n)) / (1 + z**2 / n), z * sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return [centre - half, centre + half]


def mcnemar(y, pred_a, pred_b):
    """Exact McNemar test: do two models make significantly different numbers of errors on the same students?
    Only students where exactly one model is right count. Under 'no difference' each such student is a coin flip."""
    a_only = int(((pred_a == y) & (pred_b != y)).sum())
    b_only = int(((pred_a != y) & (pred_b == y)).sum())
    n = a_only + b_only
    p = min(1.0, 2 * sum(comb(n, k) for k in range(min(a_only, b_only) + 1)) / 2**n) if n else 1.0
    return {"dnn_right_only": a_only, "other_right_only": b_only, "p_value": p}


def evaluate(params, train, test):
    models = tuned_baselines(train)
    models[DNN] = build_dnn(params)

    # Same repeated CV for every model (different seed from tuning, so scores are not the tuning scores re-used).
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=7)
    rows = {}
    for name, model in models.items():
        s = cross_validate(model, train, train["at_risk"], cv=cv, scoring=SCORING, n_jobs=-1)
        rows[name] = ({f"cv_{k}": s[f"test_{k}"].mean() for k in SCORING}
                      | {"cv_accuracy_std": s["test_accuracy"].std(), "cv_f1_macro_std": s["test_f1_macro"].std()})
        models[name] = model.fit(train, train["at_risk"])  # refit on all 519 training rows
        rows[name] |= {f"test_{k}": v for k, v in test_metrics(models[name], test).items()}
        print(f"  {name:22s} CV macro-F1 {rows[name]['cv_f1_macro']:.3f} | test acc {rows[name]['test_accuracy']:.3f}")

    # The DNN depends on its random initialisation, so also report test scores over 5 different seeds.
    seeds = [test_metrics(build_dnn(params).set_params(model__seed=100 * s).fit(train, train["at_risk"]), test)
             for s in range(5)]
    dnn, y = models[DNN], test["at_risk"].to_numpy()
    yhat = dnn.predict(test)
    extra = {"seed_mean": pd.DataFrame(seeds).mean().to_dict(), "seed_std": pd.DataFrame(seeds).std().to_dict(),
             "seed_runs": seeds, "bootstrap_95ci": bootstrap_ci(dnn, test),
             "wilson_95ci_accuracy": wilson_ci(int((yhat == y).sum()), len(y)),
             "confusion_matrix": confusion_matrix(y, yhat).tolist(),  # [[TN, FP], [FN, TP]]
             "classification_report": classification_report(y, yhat, target_names=["Pass", "At risk"], digits=3),
             "mcnemar_vs": {name: mcnemar(y, yhat, m.predict(test)) for name, m in models.items() if name != DNN},
             "best_epoch": dnn.named_steps["model"].best_epoch_,
             "selected_features": list(dnn.named_steps["select"].get_feature_names_out()), "params": params}
    joblib.dump(dnn, "models/dnn.joblib")
    plots(models, test)
    comparison_plot(rows)
    return rows, extra


def plots(models, test):
    dnn = models[DNN]
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ConfusionMatrixDisplay.from_estimator(dnn, test, test["at_risk"], display_labels=["Pass", "At risk"], cmap="Greens",
                                          colorbar=False, ax=ax)
    ax.set_title("DNN confusion matrix (test set)")
    fig.tight_layout(); fig.savefig(f"{FIG}/confusion.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, model in models.items():
        if name != "Majority class":
            RocCurveDisplay.from_estimator(model, test, test["at_risk"], name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "--", color="grey"); ax.set_title("ROC curves (locked test set)")
    fig.tight_layout(); fig.savefig(f"{FIG}/roc.png", dpi=150); plt.close(fig)

    h = dnn.named_steps["model"].history_
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(h["train"], label="train loss"); ax.plot(h["val"], label="validation loss")
    ax.axvline(dnn.named_steps["model"].best_epoch_ - 1, color="grey", ls="--", label="best epoch (weights kept)")
    ax.set(xlabel="Epoch", ylabel="Binary cross-entropy loss", title="DNN learning curve (first network of the ensemble)")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{FIG}/learning_curve.png", dpi=150); plt.close(fig)


def comparison_plot(rows):
    """DNN vs traditional ML: CV accuracy and CV macro-F1 (mean ± std over 15 folds), plus locked-test accuracy."""
    names = list(rows)
    colours = ["#16a34a" if n == DNN else "#a1a1aa" for n in names]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    panels = [("cv_accuracy", "cv_accuracy_std", "CV accuracy (mean ± std)"),
              ("cv_f1_macro", "cv_f1_macro_std", "CV macro-F1 (mean ± std)"), ("test_accuracy", None, "Locked-test accuracy")]
    for ax, (key, std, title) in zip(axes, panels):  # dot plot: the axis doesn't start at 0, so bars would exaggerate gaps
        for y, n in enumerate(names):
            v, e = rows[n][key], rows[n][std] if std else 0
            ax.errorbar(v, y, xerr=e, fmt="o", color=colours[y], ecolor=colours[y], capsize=3, ms=7)
            ax.text(v + e + 0.012, y, f"{v:.3f}", va="center", fontsize=8)
        ax.set(title=title, xlim=(0.4, 1.08), yticks=range(len(names)), yticklabels=names)
        ax.grid(axis="x", color="#e4e4e7"); ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.suptitle("DNN (green) vs traditional machine learning, same preprocessing and cross-validation")
    fig.tight_layout(); fig.savefig(f"{FIG}/comparison.png", dpi=150); plt.close(fig)


def write_report(rows, extra):
    m, s, ci, w = extra["seed_mean"], extra["seed_std"], extra["bootstrap_95ci"], extra["wilson_95ci_accuracy"]
    (tn, fp), (fn, tp) = extra["confusion_matrix"]
    lines = ["# Results", "",
             "Portuguese dataset, 649 students. Target: at risk = G3 < 10. 519 training students, 130 locked test students.",
             "**CV** = 5-fold x 3 repeats on the 519 training students. **Test** = the 130 locked students, scored once.", "",
             "| Model | CV accuracy | CV macro-F1 | CV at-risk recall | CV ROC-AUC | Test accuracy | Test precision | Test recall | Test macro-F1 | Test ROC-AUC |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in rows.items():
        lines.append(f"| {name} | {r['cv_accuracy']:.3f} ± {r['cv_accuracy_std']:.3f} | {r['cv_f1_macro']:.3f} ± {r['cv_f1_macro_std']:.3f} | "
                     f"{r['cv_recall_at_risk']:.3f} | {r['cv_roc_auc']:.3f} | {r['test_accuracy']:.3f} | "
                     f"{r['test_precision_at_risk']:.3f} | {r['test_recall_at_risk']:.3f} | {r['test_f1_macro']:.3f} | {r['test_roc_auc']:.3f} |")
    lines += ["", "## DNN on the locked test set", "",
              f"- Saved model (used by the app): accuracy {rows[DNN]['test_accuracy']:.3f}, "
              f"Wilson 95% CI {w[0]:.3f}–{w[1]:.3f}, bootstrap 95% CI {ci['accuracy'][0]:.3f}–{ci['accuracy'][1]:.3f}.",
              f"- 5 retrained copies (seeds 0, 100, 200, 300, 400): accuracy {m['accuracy']:.3f} ± {s['accuracy']:.3f}, "
              f"macro-F1 {m['f1_macro']:.3f} ± {s['f1_macro']:.3f}. This spread measures only the random start, not the small test set.",
              f"- Confusion matrix: {tn} true pass, {fp} false alarms, {fn} missed at-risk, {tp} caught at-risk.", "",
              "```", extra["classification_report"].rstrip(), "```", "",
              "McNemar exact test, DNN vs each model on the same 130 students (p < 0.05 would mean a real difference):", "",
              "| Compared with | Only DNN right | Only other right | p-value |", "|---|---|---|---|"]
    lines += [f"| {k} | {v['dnn_right_only']} | {v['other_right_only']} | {v['p_value']:.3f} |" for k, v in extra["mcnemar_vs"].items()]
    lines += ["", f"Selected features ({len(extra['selected_features'])}): {', '.join(extra['selected_features'])}.",
              f"Hyperparameters: `{extra['params']}`; best epoch {extra['best_epoch']}.", "",
              "![comparison](figures/comparison.png)",
              "![confusion](figures/confusion.png) ![roc](figures/roc.png)", "![learning curve](figures/learning_curve.png)", ""]
    with open("results/RESULTS.md", "w") as f:
        f.write("\n".join(lines))


def main():
    train, test = split(load_data())
    with open("results/best_params.json") as f:
        params = json.load(f)
    rows, extra = evaluate(params, train, test)
    with open("results/metrics.json", "w") as f:
        json.dump({"models": rows, "dnn": extra}, f, indent=2, default=float)
    write_report(rows, extra)
    print("\nWrote results/RESULTS.md, results/metrics.json and models/dnn.joblib")


if __name__ == "__main__":
    main()
