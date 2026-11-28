"""Step 3: make every choice with cross-validation, train the final DNN, compare it with traditional ML, save everything.

Run:  python -m src.train        (about 8 minutes on a laptop)

How choices are made: 5-fold cross-validation on the 519 training students (train on 4/5, check on 1/5, five times).
The 130 test students are used only once, at the very end, to report the final scores.
"""
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.data import SEED, load_data, make_preprocessor, split
from src.model import build_model, predict_at_risk, train_model

FOLDS = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)  # the same 5 folds for every experiment
START = {"hidden": (16, 8), "dropout": 0.3, "learning_rate": 0.001, "activation": "relu", "optimizer": "adam"}
FIG = "results/figures"
GREEN, GREY = "#16a34a", "#a1a1aa"


def scores(y_true, y_pred):
    return {"accuracy": accuracy_score(y_true, y_pred), "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred), "f1": f1_score(y_true, y_pred)}


def cv_dnn(train, k, settings):
    """5-fold cross-validation of the DNN. Scaling and feature selection are re-learned inside each fold (no leakage)."""
    folds = []
    for fit_rows, check_rows in FOLDS.split(train, train.at_risk):
        part, check = train.iloc[fit_rows], train.iloc[check_rows]
        prep = make_preprocessor(k).fit(part, part.at_risk)
        model = build_model(prep.transform(part).shape[1], **settings)
        train_model(model, prep.transform(part), part.at_risk.to_numpy())
        folds.append(scores(check.at_risk, predict_at_risk(model, prep.transform(check)) >= 0.5))
    result = {m: float(np.mean([f[m] for f in folds])) for m in folds[0]}
    print(f"  k={k!s:>3} {settings} -> CV accuracy {result['accuracy']:.3f}, F1 {result['f1']:.3f}", flush=True)
    return result


def best_by_f1(options):
    return max(options, key=lambda name: options[name]["f1"])


def main():
    train, test = split(load_data())
    y_train, y_test = train.at_risk.to_numpy(), test.at_risk.to_numpy()

    # A. How many input columns to keep?
    print("A. number of features")
    features = {str(k): cv_dnn(train, k, START) for k in [5, 10, 20, "all"]}
    k = best_by_f1(features)
    k = k if k == "all" else int(k)

    # B. Grid search: 2 layer sizes x 2 dropout rates x 2 learning rates = 8 combinations
    print("B. grid search")
    grid = {f"{h[0]}-{h[1]} | dropout {d} | lr {lr}": START | {"hidden": h, "dropout": d, "learning_rate": lr}
            for h in [(16, 8), (32, 16)] for d in [0.2, 0.4] for lr in [0.001, 0.01]}
    grid_scores = {name: cv_dnn(train, k, s) for name, s in grid.items()}
    settings = grid[best_by_f1(grid_scores)]

    # C. Which activation function?  D. Which optimizer?
    print("C. activation"); activation = {a: cv_dnn(train, k, settings | {"activation": a}) for a in ["relu", "tanh", "sigmoid"]}
    settings = settings | {"activation": best_by_f1(activation)}
    print("D. optimizer"); optimizer = {o: cv_dnn(train, k, settings | {"optimizer": o}) for o in ["adam", "sgd"]}
    settings = settings | {"optimizer": best_by_f1(optimizer)}
    print("final settings:", k, settings)

    # E. Train the final DNN on all 519 training students, then test once on the 130 test students
    prep = make_preprocessor(k).fit(train, y_train)
    X_train, X_test = prep.transform(train), prep.transform(test)
    dnn = build_model(X_train.shape[1], **settings)
    history = train_model(dnn, X_train, y_train)
    dnn_pred = (predict_at_risk(dnn, X_test) >= 0.5).astype(int)
    # The last experiment (D) already cross-validated exactly these final settings, so that is the DNN's CV score.
    results = {"DNN (neural network)": {"cv": optimizer[settings["optimizer"]], "test": scores(y_test, dnn_pred)}}

    # F. Traditional machine learning: same inputs, same folds, each with a small grid search of its main setting
    traditional = {
        "Logistic regression": (LogisticRegression(max_iter=2000), {"model__C": [0.01, 0.1, 1, 10]}),
        "Decision tree": (DecisionTreeClassifier(random_state=SEED), {"model__max_depth": [2, 3, 5, 8]}),
        "Random forest": (RandomForestClassifier(n_estimators=300, random_state=SEED), {"model__max_depth": [3, 6, None]}),
    }
    for name, (model, grid_ml) in traditional.items():
        search = GridSearchCV(Pipeline([("prep", make_preprocessor(k)), ("model", model)]), grid_ml, cv=FOLDS,
                              scoring=["accuracy", "precision", "recall", "f1"], refit="f1").fit(train, y_train)
        i = search.best_index_
        results[name] = {"cv": {m: float(search.cv_results_[f"mean_test_{m}"][i]) for m in ["accuracy", "precision", "recall", "f1"]},
                         "test": scores(y_test, search.predict(test)), "setting": str(search.best_params_)}
    results["Always predict 'pass'"] = {"cv": None, "test": scores(y_test, np.zeros_like(y_test))}

    # G. Save the model, the results and the graphs
    os.makedirs("models", exist_ok=True)
    joblib.dump(prep, "models/preprocessor.joblib")
    dnn.save("models/dnn.keras")
    kept = list(prep.named_steps["select"].get_feature_names_out())
    info = dict(zip(prep.named_steps["select"].feature_names_in_, prep.named_steps["select"].scores_))
    report = {"k": k, "settings": settings, "kept_columns": kept, "info_scores": info,
              "experiments": {"features": features, "grid": grid_scores, "activation": activation, "optimizer": optimizer},
              "models": results, "confusion_matrix": confusion_matrix(y_test, dnn_pred).tolist(),
              "epochs_trained": len(history.history["loss"]), "best_epoch": int(np.argmin(history.history["val_loss"])) + 1}
    with open("results/results.json", "w") as f:
        json.dump(report, f, indent=2, default=float)
    draw_graphs(report, history, y_test, dnn_pred)
    print("test scores:", {n: round(r["test"]["accuracy"], 3) for n, r in results.items()})


def draw_graphs(report, history, y_test, dnn_pred):
    # 1. Confusion matrix of the DNN on the test students
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, dnn_pred, display_labels=["Pass", "At risk"], cmap="Greens", colorbar=False, ax=ax)
    ax.set_title("DNN on the 130 test students"); fig.tight_layout(); fig.savefig(f"{FIG}/confusion.png", dpi=150); plt.close(fig)

    # 2. Learning curve: loss per epoch
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(history.history["loss"], color=GREY, label="training loss")
    ax.plot(history.history["val_loss"], color=GREEN, label="validation loss")
    ax.axvline(report["best_epoch"] - 1, color="black", ls="--", lw=1, label=f"best epoch ({report['best_epoch']}), weights kept")
    ax.set(xlabel="epoch", ylabel="binary cross-entropy loss", title="How the DNN learned"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{FIG}/learning_curve.png", dpi=150); plt.close(fig)

    # 3. DNN vs traditional ML on the test students
    names = list(report["models"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, metric, title in [(axes[0], "accuracy", "Test accuracy"), (axes[1], "f1", "Test F1 (finding at-risk students)")]:
        values = [report["models"][n]["test"][metric] for n in names]
        ax.barh(names, values, color=[GREEN if n.startswith("DNN") else GREY for n in names])
        for y, v in enumerate(values):
            ax.text(v + 0.01, y, f"{v:.3f}", va="center", fontsize=9)
        ax.set(xlim=(0, 1.1), title=title)
    axes[0].invert_yaxis(); fig.tight_layout(); fig.savefig(f"{FIG}/comparison.png", dpi=150); plt.close(fig)

    # 4. Why these choices: number of features, activation, optimizer (cross-validation F1)
    exp = report["experiments"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
    for ax, key, title, chosen in [(axes[0], "features", "Number of features kept", str(report["k"])),
                                   (axes[1], "activation", "Activation function", report["settings"]["activation"]),
                                   (axes[2], "optimizer", "Optimizer", report["settings"]["optimizer"])]:
        names = list(exp[key]); values = [exp[key][n]["f1"] for n in names]
        ax.bar(names, values, color=[GREEN if n == chosen else GREY for n in names])
        for x, v in enumerate(values):
            ax.text(x, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
        ax.set(ylim=(0, 1), title=title, ylabel="cross-validation F1")
    fig.suptitle("Why these choices (green = chosen, best cross-validation F1)"); fig.tight_layout()
    fig.savefig(f"{FIG}/choices.png", dpi=150); plt.close(fig)

    # 5. Information score of every column (top 20); green = kept
    info = sorted(report["info_scores"].items(), key=lambda kv: kv[1])[-20:]
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.barh([c for c, _ in info], [v for _, v in info], color=[GREEN if c in report["kept_columns"] else GREY for c, _ in info])
    ax.set(xlabel="information score (mutual information with pass / at risk)", title="Which columns tell us most (green = kept)")
    fig.tight_layout(); fig.savefig(f"{FIG}/feature_scores.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
