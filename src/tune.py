"""Hyperparameter search for the DNN with Optuna, using cross-validation on the training set only.

Run:  python -m src.tune            (about 10 minutes on a laptop CPU)
Writes results/best_params.json and logs every trial to results/experiments.csv.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # tiny data: one thread per process is fastest and avoids a libomp clash on macOS

import json
from datetime import datetime

import optuna
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.data import SEED, load_data, make_preprocessor, split
from src.model import DNNClassifier

N_TRIALS = 40


def build_dnn(params, n_models=5):
    """Turn a flat dict of hyperparameters into preprocessing + DNN pipeline."""
    hidden = tuple(max(8, params["width"] // 2**i) for i in range(params["n_layers"]))  # funnel shape, e.g. 64 -> 32 -> 16
    dnn = DNNClassifier(hidden=hidden, dropout=params["dropout"], batchnorm=params["batchnorm"], lr=params["lr"],
                        weight_decay=params["weight_decay"], batch_size=params["batch_size"],
                        class_weight=params["class_weight"], n_models=n_models)
    return Pipeline(make_preprocessor(params["k"]) + [("model", dnn)])


def objective(trial, train):
    params = {
        "n_layers": trial.suggest_int("n_layers", 1, 3),
        "width": trial.suggest_categorical("width", [16, 32, 64, 128]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "batchnorm": trial.suggest_categorical("batchnorm", [True, False]),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "class_weight": trial.suggest_categorical("class_weight", [True, False]),
        "k": trial.suggest_categorical("k", [10, 15, 20, "all"]),
    }
    # 5-fold x 2 repeats: averaging 10 folds makes each trial's score less dependent on one lucky split.
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
    # Macro-F1 averages F1 of both classes, so ignoring the rare "at risk" class is punished (accuracy would not punish it).
    # Tuning trains 1 network per fold (5x faster); the final model averages 5 (see DNNClassifier).
    scores = cross_val_score(build_dnn(params, n_models=1), train, train["at_risk"], cv=cv, scoring="f1_macro", n_jobs=-1)
    log = {"time": datetime.now().isoformat(timespec="seconds"), "trial": trial.number,
           **params, "cv_f1_macro": scores.mean(), "cv_f1_std": scores.std()}
    pd.DataFrame([log]).to_csv("results/experiments.csv", mode="a", index=False,
                               header=not os.path.exists("results/experiments.csv"))
    return scores.mean()


def main():
    train, _ = split(load_data())  # the test set is not used here at all
    if os.path.exists("results/experiments.csv"):
        os.remove("results/experiments.csv")  # the log describes the current study only (older logs are in git history)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(lambda t: objective(t, train), n_trials=N_TRIALS)
    print(f"best CV macro-F1 {study.best_value:.3f} with {study.best_params}")
    with open("results/best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)


if __name__ == "__main__":
    main()
