"""A neural network written from scratch with NumPy only (CHO lab 6): forward pass, loss, backpropagation, updates.

This is a teaching implementation, not the production model. It shows exactly what PyTorch's autograd and
optimizer do for us, and gradient_check() proves the hand-written gradients are right (CHO lab 2).
"""
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class NumpyMLP:
    """inputs -> hidden (ReLU) -> 1 output (sigmoid), trained with binary cross-entropy and mini-batch gradient descent."""

    def __init__(self, n_inputs, n_hidden=16, lr=0.1, epochs=200, batch_size=32, seed=42):
        rng = np.random.default_rng(seed)
        # He initialisation: weights ~ N(0, 2 / fan_in), biases start at 0
        self.params = {"W1": rng.normal(0, np.sqrt(2 / n_inputs), (n_inputs, n_hidden)), "b1": np.zeros(n_hidden),
                       "W2": rng.normal(0, np.sqrt(2 / n_hidden), (n_hidden, 1)), "b2": np.zeros(1)}
        self.lr, self.epochs, self.batch_size, self.rng = lr, epochs, batch_size, rng

    def forward(self, X):
        P = self.params
        z1 = X @ P["W1"] + P["b1"]       # (n, hidden): weighted sums of the inputs
        a1 = np.maximum(z1, 0)           # ReLU
        z2 = a1 @ P["W2"] + P["b2"]      # (n, 1): the logit
        return sigmoid(z2[:, 0]), (z1, a1)

    def loss(self, X, y):
        p = np.clip(self.forward(X)[0], 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))  # binary cross-entropy

    def gradients(self, X, y):
        """Backpropagation: apply the chain rule from the loss back to every weight."""
        P, n = self.params, len(y)
        p, (z1, a1) = self.forward(X)
        dz2 = ((p - y) / n)[:, None]          # dLoss/dlogit: sigmoid + cross-entropy simplify to (p - y)
        da1 = dz2 @ P["W2"].T                 # pass the error back through W2
        dz1 = da1 * (z1 > 0)                  # ReLU passes gradient only where its input was positive
        return {"W2": a1.T @ dz2, "b2": dz2.sum(0), "W1": X.T @ dz1, "b1": dz1.sum(0)}

    def fit(self, X, y, X_val=None, y_val=None):
        self.history = {"train": [], "val": []}
        for _ in range(self.epochs):
            order = self.rng.permutation(len(X))
            for start in range(0, len(X), self.batch_size):  # mini-batch gradient descent
                idx = order[start:start + self.batch_size]
                for name, g in self.gradients(X[idx], y[idx]).items():
                    self.params[name] -= self.lr * g      # w <- w - learning_rate * dLoss/dw
            self.history["train"].append(self.loss(X, y))
            if X_val is not None:
                self.history["val"].append(self.loss(X_val, y_val))
        return self

    def predict_proba(self, X):
        return self.forward(X)[0]

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


def gradient_check(model, X, y, eps=1e-5, n_checks=20, seed=0):
    """Compare backprop gradients with numerical ones, (L(w + eps) - L(w - eps)) / (2 eps), for random weights.
    Returns the largest relative error; below ~1e-6 means backprop is implemented correctly."""
    rng, analytic, worst = np.random.default_rng(seed), model.gradients(X, y), 0.0
    for name, W in model.params.items():
        for _ in range(n_checks):
            i = tuple(rng.integers(0, s) for s in W.shape)
            old = W[i]
            W[i] = old + eps; plus = model.loss(X, y)
            W[i] = old - eps; minus = model.loss(X, y)
            W[i] = old
            numeric = (plus - minus) / (2 * eps)
            worst = max(worst, abs(numeric - analytic[name][i]) / max(1e-12, abs(numeric) + abs(analytic[name][i])))
    return worst


if __name__ == "__main__":  # self-check on random data: backprop must agree with numerical gradients
    X, y = np.random.default_rng(1).normal(size=(50, 5)), (np.random.default_rng(2).random(50) < 0.3).astype(float)
    err = gradient_check(NumpyMLP(5, 8), X, y)
    assert err < 1e-6, err
    print(f"gradient check passed, max relative error {err:.2e}")
