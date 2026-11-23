"""The deep neural network: a multilayer perceptron in PyTorch, wrapped in the scikit-learn API.

Wrapping it as a scikit-learn classifier means the DNN plugs into the same Pipeline,
cross-validation and metrics code as the traditional models, so the comparison is fair.
"""
import copy

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from torch import nn


class MLP(nn.Module):
    """Linear -> [BatchNorm] -> ReLU -> Dropout, repeated per hidden layer, then one output logit."""

    def __init__(self, n_inputs, hidden, dropout, batchnorm):
        super().__init__()
        layers, width = [], n_inputs
        for units in hidden:
            linear = nn.Linear(width, units)
            nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")  # He init: keeps activation variance stable with ReLU
            nn.init.zeros_(linear.bias)
            layers += [linear, nn.BatchNorm1d(units) if batchnorm else nn.Identity(), nn.ReLU(), nn.Dropout(dropout)]
            width = units
        layers.append(nn.Linear(width, 1))  # one logit; sigmoid(logit) = P(at risk)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DNNClassifier(ClassifierMixin, BaseEstimator):
    """Trains the MLP with mini-batch AdamW, binary cross-entropy and early stopping.

    n_models > 1 trains that many networks from different random seeds and averages their
    probabilities (a seed ensemble). One small network on ~440 rows is sensitive to its random
    initialisation; averaging several cuts that variance without changing the architecture.
    """

    def __init__(self, hidden=(32, 16), dropout=0.2, batchnorm=True, lr=1e-3, weight_decay=1e-3,
                 batch_size=32, class_weight=True, max_epochs=300, patience=30, n_models=5, seed=42):
        self.hidden, self.dropout, self.batchnorm = hidden, dropout, batchnorm
        self.lr, self.weight_decay, self.batch_size = lr, weight_decay, batch_size
        self.class_weight, self.max_epochs, self.patience = class_weight, max_epochs, patience
        self.n_models, self.seed = n_models, seed

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)
        self.classes_ = np.array([0, 1])
        runs = [self._train_one(X, y, self.seed + i) for i in range(self.n_models)]
        self.models_ = [model for model, _ in runs]
        self.history_ = runs[0][1]  # learning curve of the first network, for plotting
        self.best_epoch_ = int(np.argmin(self.history_["val"])) + 1
        return self

    def _train_one(self, X, y, seed):
        torch.manual_seed(seed)
        # Hold out 15% of the training rows to decide when to stop (never the test set).
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, stratify=y, random_state=seed)
        X_tr, y_tr, X_val, y_val = map(torch.from_numpy, (X_tr, y_tr, X_val, y_val))

        model = MLP(X.shape[1], self.hidden, self.dropout, self.batchnorm)
        # AdamW = Adam with decoupled weight decay (L2-style regularisation).
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # Only ~15% of students fail. pos_weight = (#pass / #fail) makes each missed failure cost as much
        # as all the passes it is outnumbered by, so the model can't ignore the minority class.
        pos_weight = (len(y_tr) - y_tr.sum()) / y_tr.sum() if self.class_weight else torch.tensor(1.0)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # sigmoid + binary cross-entropy, numerically stable

        best_loss, bad_epochs, history = float("inf"), 0, {"train": [], "val": []}
        best_state = copy.deepcopy(model.state_dict())
        for epoch in range(self.max_epochs):
            model.train()  # dropout on, batchnorm uses batch statistics
            order, total = torch.randperm(len(X_tr)), 0.0
            for start in range(0, len(X_tr), self.batch_size):
                idx = order[start:start + self.batch_size]
                if len(idx) < 2:  # BatchNorm needs at least 2 rows
                    continue
                optimizer.zero_grad()
                loss = loss_fn(model(X_tr[idx]), y_tr[idx])  # forward pass
                loss.backward()                               # backpropagation
                optimizer.step()                              # weight update
                total += loss.item() * len(idx)

            model.eval()  # dropout off, batchnorm uses running statistics
            with torch.no_grad():
                val_loss = loss_fn(model(X_val), y_val).item()
            history["train"].append(total / len(X_tr))
            history["val"].append(val_loss)

            # Early stopping: keep the weights from the best validation epoch, stop after `patience` epochs without improvement.
            if val_loss < best_loss:
                best_loss, bad_epochs, best_state = val_loss, 0, copy.deepcopy(model.state_dict())
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break
        model.load_state_dict(best_state)
        model.eval()  # dropout off, batchnorm uses running statistics
        return model, history

    def predict_proba(self, X):
        X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            p = np.mean([torch.sigmoid(m(X)).numpy() for m in self.models_], axis=0)  # average the ensemble
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
