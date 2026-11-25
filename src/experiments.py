"""Supplementary experiments for the course practicals (CHO labs 7, 8, 10, 11), on training data only.

Every run trains one copy of the final architecture on the same 85/15 train/validation split of the 519
training students, for a fixed number of epochs and with no early stopping, so the curves can be compared
directly. The locked test set is never used here, and none of these runs changes the final model.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import time

import numpy as np
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
