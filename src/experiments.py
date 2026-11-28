"""Supplementary experiments for the course practicals (CHO labs 7, 8, 10, 11), on training data only.

Every run trains one copy of the final architecture on the same 85/15 train/validation split of the 519
training students, for a fixed number of epochs and with no early stopping, so the curves can be compared
directly. The locked test set is never used here, and none of these runs changes the final model.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data import SEED, make_preprocessor
from src.model import MLP

ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}


def prepare(train, k=10):
    """Split the training students 85/15 and preprocess (fitted on the 85% part only)."""
    fit_part, val_part = train_test_split(train, test_size=0.15, stratify=train["at_risk"], random_state=SEED)
    pre = Pipeline(make_preprocessor(k)).fit(fit_part, fit_part["at_risk"])
    as_tensor = lambda a: torch.tensor(np.asarray(a, dtype=np.float32))
    return (as_tensor(pre.transform(fit_part)), as_tensor(fit_part["at_risk"]),
            as_tensor(pre.transform(val_part)), as_tensor(val_part["at_risk"]))


def make_optimizer(name, params, lr):
    return {"SGD": lambda: torch.optim.SGD(params, lr=lr),
            "SGD + momentum": lambda: torch.optim.SGD(params, lr=lr, momentum=0.9),
            "RMSprop": lambda: torch.optim.RMSprop(params, lr=lr),
            "Adam": lambda: torch.optim.Adam(params, lr=lr),
            "AdamW": lambda: torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)}[name]()


def run(data, optimizer="AdamW", lr=1e-3, batch_size=64, init="he", activation="relu", batchnorm=True,
        schedule=None, epochs=100, seed=SEED):
    """Train one network and return its per-epoch history plus wall-clock time."""
    X_tr, y_tr, X_val, y_val = data
    torch.manual_seed(seed)
    model = MLP(X_tr.shape[1], hidden=(64, 32, 16), dropout=0.16, batchnorm=batchnorm)
    for i, layer in enumerate(model.net):
        if isinstance(layer, nn.ReLU):
            model.net[i] = ACTIVATIONS[activation]()
        if isinstance(layer, nn.Linear) and init != "he":  # MLP already uses He; re-initialise for the comparison
            {"xavier": nn.init.xavier_normal_, "zeros": nn.init.zeros_}[init](layer.weight)

    # PyTorch Dataset/DataLoader: wraps the tensors and yields shuffled mini-batches each epoch.
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), drop_last=batchnorm and batch_size > 1)
    opt = make_optimizer(optimizer, model.parameters(), lr)
    scheduler = {None: None,
                 "step": lambda: torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.5),     # halve every 30 epochs
                 "exponential": lambda: torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.97)}[schedule]
    scheduler = scheduler() if scheduler else None
    loss_fn = nn.BCEWithLogitsLoss()

    history, start = {"train": [], "val": [], "val_acc": [], "lr": []}, time.perf_counter()
    for _ in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        history["lr"].append(opt.param_groups[0]["lr"])
        if scheduler:
            scheduler.step()
        model.eval()
        with torch.no_grad():  # validation loop
            logits = model(X_val)
            history["val"].append(loss_fn(logits, y_val).item())
            history["val_acc"].append(((logits > 0).float() == y_val).float().mean().item())
        history["train"].append(total / len(loader.dataset))
    history["seconds"] = time.perf_counter() - start
    return history


def summary(results):
    """One row per run: best validation loss, the epoch it happened, and validation accuracy at that epoch."""
    rows = []
    for name, h in results.items():
        best = int(np.argmin(h["val"]))
        rows.append({"run": name, "best_val_loss": h["val"][best], "best_epoch": best + 1,
                     "val_acc_at_best": h["val_acc"][best], "final_val_loss": h["val"][-1], "seconds": h["seconds"]})
    return rows


def feature_ablation(train, params, n_models=5):
    """Paired CV test for every input, one at a time (training students only).

    Baseline: the DNN on the 10 encoded columns of the final model. For each of the 10 used inputs we remove it,
    and for each of the 22 unused inputs we add it back, then compare on the same 15 CV folds (5-fold x 3).
    A change smaller than about 2 standard errors is noise. The baseline uses a fixed column list chosen on the
    training set, so its CV score differs slightly from the per-fold-selection score in results/RESULTS.md.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

    from src.data import FEATURES
    from src.tune import build_dnn

    encode = make_preprocessor()[0][1]
    encoded = list(Pipeline([("encode", encode)]).fit(train).transform(train).columns)
    final = list(Pipeline(make_preprocessor(params["k"])).fit(train, train["at_risk"])[-1].get_feature_names_out())
    columns_of = {f: [c for c in encoded if c == f or c.startswith(f + "_")] for f in FEATURES}
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=7)

    def scores(cols, seed=42):
        dnn = build_dnn(params, n_models=n_models).steps[-1][1].set_params(seed=seed)
        model = Pipeline([("encode", encode), ("keep", ColumnTransformer([("keep", "passthrough", cols)])), ("model", dnn)])
        s = cross_validate(model, train, train["at_risk"], cv=cv, scoring=["accuracy", "f1_macro"], n_jobs=-1)
        return s["test_accuracy"], s["test_f1_macro"]

    base_acc, base_f1 = scores(final)
    # Noise floor: the same 10 inputs retrained with 4 other seeds. Any feature change smaller than this spread
    # is indistinguishable from simply re-running training.
    seed_f1 = [base_f1.mean()] + [scores(final, seed)[1].mean() for seed in (100, 200, 300, 400)]
    rows = []
    for f in FEATURES:
        used = any(c in final for c in columns_of[f])
        cols = [c for c in final if c not in columns_of[f]] if used else final + columns_of[f]
        acc, f1 = scores(cols)
        d_acc, d_f1 = acc - base_acc, f1 - base_f1  # paired: same folds, so fold-to-fold noise cancels
        rows.append({"feature": f, "in_model": used, "test": "removed" if used else "added",
                     "delta_cv_accuracy": d_acc.mean(), "delta_cv_f1_macro": d_f1.mean(),
                     "se_f1": d_f1.std(ddof=1) / np.sqrt(len(d_f1))})
        print(f"  {f:11s} {'removed' if used else 'added  '}  Δacc {d_acc.mean():+.3f}  ΔF1 {d_f1.mean():+.3f} ± {rows[-1]['se_f1']:.3f}", flush=True)
    out = pd.DataFrame(rows)
    out.attrs["baseline"] = {"cv_accuracy": base_acc.mean(), "cv_f1_macro": base_f1.mean(), "seed_cv_f1_macro": seed_f1,
                             "seed_f1_range": max(seed_f1) - min(seed_f1)}
    return out
