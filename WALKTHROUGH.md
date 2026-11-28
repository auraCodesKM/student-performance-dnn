# Project walkthrough: what we built, line by line

This is the study guide for the viva. It has six parts:

- **A.** The project in one page
- **B.** The notebook, section by section
- **C.** The Kaggle version: how it runs, the data source and the results
- **D.** The code explained line by line (every file in `src/` and `app.py`)
- **E.** Engineering decisions mapped to the course handout (CHO): what we used, and why not the alternatives
- **F.** The comparison the project brief asks for: DNN vs traditional ML

The full decision log, with maths and evidence, is in `DECISIONS.md`. The demo plan for the app is in `DEMO_SCRIPT.md`.

---

## A. The project in one page

**Brief** (`DNN-Project_UPDATED SCHEDULE.pdf`): build a DNN that predicts a student's final academic performance from demographic, academic, social and behavioural data. It must include:
- preprocessing, EDA and feature selection;
- architecture design, training, hyperparameter tuning and evaluation;
- a comparison with at least one traditional ML algorithm;
- a Streamlit app that shows the predicted performance category.

**What we built:**

| Step | What | Where |
|---|---|---|
| Data | UCI Student Performance, Portuguese course: 649 students, 33 columns | `data/student-por.csv`, `src/data.py` |
| Target | G3 ≥ 10 → **Pass**, G3 < 10 → **At risk** (100 students, 15.4%) | `load_data()` |
| Split | Stratified 80/20: 519 training, 130 test (locked) | `split()` |
| Preprocessing | One-hot encoding (17 categorical), standardisation (15 numeric): 45 columns | `make_preprocessor()` |
| Feature selection | SelectKBest, discrete mutual information, keeps 10 columns | `discrete_mi()` |
| DNN | 10 → 16 → 8 → 1; BatchNorm, ReLU, Dropout 0.48; He init; BCE loss; AdamW; early stopping; average of 5 networks | `src/model.py` |
| Tuning | Optuna, 40 trials, 5-fold × 2 CV macro-F1, training data only | `src/tune.py` |
| Comparison | Majority class, the "G2 < 10" rule, logistic regression, random forest, XGBoost, all with the same pipeline and CV | `src/evaluate.py` |
| Results | Test accuracy **91.5%**, 5 seeds **92.3 ± 0.8%**, CV **92.5%**, ROC-AUC 0.953 | `results/RESULTS.md` |
| App | Streamlit: grouped form, live Pass / At-risk prediction with explanation, model tab | `app.py` |
| Extra (CHO practicals) | NumPy network with backprop and gradient check; Keras Sequential and Functional; DataLoader; comparisons of optimisers, schedules, GD variants, initialisation and activations | `src/numpy_mlp.py`, `src/experiments.py`, notebook 13–15 |

---

## B. The notebook, section by section

File: `student_performance_dnn.ipynb`. It runs top to bottom in about 2 minutes on a laptop and about 9 minutes on Kaggle.
**Every section calls functions from `src/`**, so the notebook shows and runs the same code the scripts and the app use. Nothing is copy-pasted.

### 0. Title and data sources
- **What it shows.** The task, the dataset, the model and a table of contents.
- **Data sources:**
  - UCI repository (dataset 320, DOI 10.24432/C5TG7T, CC BY 4.0) and the Cortez & Silva 2008 paper;
  - on Kaggle, the attached dataset `uciml/student-alcohol-consumption`;
  - locally, the repo copy `data/student-por.csv`.

### 1. Setup
- **What it does:**
  1. Sets `OMP_NUM_THREADS=1`, which avoids a crash when PyTorch and XGBoost share a process on macOS and is fastest for tiny data.
  2. Sets `KERAS_BACKEND=torch`, so Keras runs on PyTorch.
  3. On Kaggle, it finds the attached `student-por.csv` and points `STUDENT_DATA_PATH` at it.
  4. If the `src/` folder is missing (on Kaggle), it `git clone`s the GitHub repo to get the code.
  5. Imports everything and prints the library versions.
- **Why:** one notebook that runs unchanged locally and on Kaggle.

### 2. The dataset: all 33 columns
- **What it does:**
  - Prints where the data was read from, and checks it is **identical** to the UCI file in the repo (`True`).
  - Builds a table of all 33 columns: description, brief group, type, number of distinct values and role (input or target).
- **Result:** 32 inputs (15 numeric including G1 and G2, plus 17 categorical) and 1 target, G3.

### 3. Exploratory data analysis
- **What it does.** `audit(df)` prints the data checks, and `figures(df)` saves and shows 5 plots: the G3 distribution, G2 vs G3, correlations with G3, at-risk rate by category, and absences.
- **Findings, each tied to a decision:**

| Finding | Decision |
|---|---|
| 0 missing values, 0 duplicate rows | No imputation |
| 15.4% at risk; "everyone passes" already scores 84.6% | Macro-F1 for tuning; report recall |
| 15 students with G3 = 0, all with 0 absences (likely dropouts) | Keep them as real fails |
| G2 correlates 0.92 with G3 | The "G2 < 10" rule becomes a baseline |
| Absences skewed (max 32), but within the codebook range | No outlier removal; standardise |

### 4. Target and locked test split
- **What it does.** `split(df)` gives 519 training students (80 at risk) and 130 test students (20 at risk), stratified so both keep 15.4%.
- **Why.** The test students are never used for any choice. They are scored once, in section 10.

### 5. Feature audit and selection
- **What it does:**
  - `feature_audit(train)` builds one row per input: group, type, encoding, rank correlation with G3, the at-risk gap between categories, **how many of the 15 CV folds keep it**, and whether the final model uses it.
  - `vif(train)` gives the variance inflation factor of each numeric column.
  - A table of the tuning results grouped by k.
- **Results:**
  - 8 features are kept in all 15 folds: G1, G2, failures, higher, school, studytime, Fedu and absences.
  - G1 and G2 have VIF ≈ 4, below 5, so their multicollinearity is acceptable.
  - k = 10 scored best (0.833 mean tuning macro-F1, against 0.791 with all 45 columns).
- **Why.** It proves that no feature was removed by hand, and that selection is stable and leakage-free.
- **Add-back / remove test.** Each unused input is added back and each used input removed, one at a time, on the same 15 folds.
  - Only removing G2 (−0.111 macro-F1) or G1 (−0.040) matters. Every other change is inside the seed-to-seed noise (0.017).
  - The **address check** shows rural students fail more (21% vs 13%), but students with the same G2 fail at the same rate.

### 6. Preprocessing pipeline
- **What it does:**
  - Shows the fitted scikit-learn `Pipeline` diagram (encode → select → model).
  - Prints 32 inputs → 45 encoded columns → 10 kept, and names the 10.
- **Why.** The whole pipeline is fitted on training rows only, so there is no leakage.

### 7. The DNN: architecture and training loop
- **What it does:**
  - Builds the `MLP` with the tuned settings and prints its layers.
  - Prints a parameter table (369 trainable per network, 1,845 for 5).
  - Prints the source of `_train_one`, the training loop.
- **Why.** The examiner sees the exact architecture and the loop: `zero_grad → forward → loss → backward → step`, `train()` and `eval()`, and early stopping.

### 8. Hyperparameter tuning (Optuna)
- **What it does:**
  - Prints the search space (9 hyperparameters).
  - Plots all 40 trials with the best-so-far line, and shows the top 5 trials and the chosen parameters.
  - `RUN_TUNING = False` by default; set it to True to rerun (about 10 minutes).
- **Result.** Best tuning macro-F1 0.870 with 2 layers (16 → 8), dropout 0.48, BatchNorm, learning rate 0.002, weight decay 0.0084, batch 32, no class weights, k = 10. The top 5 trials agree on this region.

### 9. Cross-validated comparison with traditional ML
- **What it does:**
  - `evaluate.main()` grid-searches the 3 traditional models, adds the 2 simple baselines, and runs 5-fold × 3 CV for all 6 models.
  - It refits each model, scores the test set once, runs the 5-seed DNN retrains, computes confidence intervals and McNemar tests, and saves the model, figures and `RESULTS.md`.
  - Then it shows the CV table and the **comparison chart**.
- **Result** (CV accuracy): DNN 0.925, logistic regression 0.934, XGBoost 0.929, random forest 0.921, rule 0.896, majority 0.846.

### 10. Final evaluation on the locked test set
- **What it shows:**
  - the test table for all models and the DNN's classification report;
  - Wilson and bootstrap 95% intervals, and the 5 seed runs;
  - the confusion matrix, ROC curves and learning curve;
  - the McNemar table;
  - the **threshold analysis** (out-of-fold predictions on training students only).
- **Results:**
  - DNN test accuracy **0.915**, precision 0.714, recall 0.750, macro-F1 0.841, ROC-AUC 0.953.
  - Confusion matrix 104 / 6 / 5 / 15. McNemar p ≈ 0.73–0.75 against the strong baselines.
  - A threshold of 0.3 would raise recall to 0.84 at precision 0.63.

### 11. What the model relies on
- **What it does.** Permutation importance on the test set: shuffle one column and measure the drop in ROC-AUC. This is analysis only.
- **Result.** G2 (ROC-AUC drop 0.168) and G1 (0.070) dominate. School, failures and famrel add small amounts (≤ 0.004).

### 12. Predicting individual students
- **What it does.** Runs 3 at-risk and 3 passing test students through the saved pipeline and shows P(at risk), the prediction and the truth. This is exactly what the app does.

### 13. Under the hood: NumPy
- **13a.** Recomputes the saved network's output with plain NumPy (matmul, BatchNorm with running statistics, ReLU, sigmoid). The largest difference from PyTorch is about 1e-7.
- **13b.** A network **trained from scratch in NumPy** (`src/numpy_mlp.py`):
  - the gradient check passes (error 2e-9);
  - 5-fold CV accuracy 0.929; test accuracy 0.923.

### 14. The same network in Keras
- **What it does:**
  - Builds the architecture with the **Sequential** and the **Functional** API (same 369 trainable parameters).
  - Trains with `EarlyStopping` and `ModelCheckpoint`, **reloads** the checkpoint, and plots the training history.
  - Compares one Keras network (test 0.923) with one PyTorch network (0.885). The gap is the random start, not the framework.

### 15. Training experiments from the course practicals
All on one 85/15 train/validation split of the **training students**, with a PyTorch `DataLoader`:
- **15a. Optimisers:** SGD 0.344 > RMSprop 0.177 ≈ Adam ≈ AdamW 0.185 > momentum 0.167 (best validation loss; lower is better).
- **15b. Schedules:** constant 0.185 is better than step 0.232 and exponential 0.335.
- **15c. Gradient-descent variants:**
  - batch GD is slow (0.307);
  - stochastic GD (1 row) is best early (0.181 at epoch 9), then overfits and is 20× slower;
  - mini-batch GD: 0.208.
- **15d. Initialisation:** He 0.185, Xavier 0.205, zeros stuck at 0.595 (the symmetry problem).
- **15e. Activations:** plots of the functions and their derivatives; ReLU 0.185 beats tanh 0.255 and sigmoid 0.270.

### 16. Hardware: CPU vs GPU
- **What it does.** Times 50 epochs on every available device.
- **Result.** M3 CPU 0.25 s vs MPS 0.74 s; Kaggle CPU 1.66 s vs Tesla T4 1.81 s. The CPU wins.

### 17. Supplementary
- **No-grades ablation:** without G1 and G2 nothing beats the majority class.
- **Results history:** v1.0 → v1.1 → v2, explained in `DECISIONS.md` section 7.

### 18. Conclusions and limitations
- The DNN performs as well as the best traditional models, at about 91–92%.
- **Limitations:** small data, the test set seen once per version, recall of 75%, and sensitive attributes.

---

## C. The Kaggle version

- **Link:** https://www.kaggle.com/code/kavinthakurkm/student-performance-dnn (public)
- **Settings:** GPU accelerator (Tesla T4) and internet on, because the first cell clones the GitHub repo for `src/`.
- **Data source (the "Input" panel on Kaggle):** `uciml/student-alcohol-consumption`, Kaggle's mirror of the UCI files. The notebook reads its `student-por.csv`.
  - That file uses commas while the UCI original uses semicolons. `load_data()` detects the separator.
  - The notebook prints `identical to the UCI file in the repository: True`, which proves it is the same 649 × 33 data.
- **How it runs, step by step:**
  1. `glob("/kaggle/input/**/student-por.csv")` finds the attached file, and `STUDENT_DATA_PATH` is set to it.
  2. `git clone` fetches `src/`, the saved results and the figures from GitHub, and the notebook changes into that folder.
  3. Every section then runs as it does locally.
- **Environment differences.** Kaggle has Python 3.13, torch 2.11, scikit-learn 1.6.1 and pandas 2.3.
  - The DNN's results came out **identical** to the local run: test 0.915, 5 seeds 0.923 ± 0.008, CV macro-F1 0.846.
  - XGBoost's test accuracy was 0.877 there (0.900 locally) because of a different xgboost version, a difference of 3 students.
- **Why Kaggle at all.** It shows the project is reproducible on someone else's machine, and it gave us a real GPU to benchmark.
- **How sir can run it.** Open the link → *Copy & Edit* → *Run All*. It takes about 9 minutes.

---

## D. The code explained line by line

### D1. `src/data.py`: data, target, split, preprocessing

```python
import os
import warnings
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
```
- `os` reads the optional data-path environment variable; `warnings` silences one expected scikit-learn warning.
- `pandas` reads the CSV.
- `ColumnTransformer` applies different preprocessing to different columns.
- `SelectKBest` and `mutual_info_classif` do the feature selection.
- `train_test_split` makes the locked split.
- `OneHotEncoder` and `StandardScaler` do the encoding and scaling.

```python
DATA_PATH = os.environ.get("STUDENT_DATA_PATH", "data/student-por.csv")
SEED = 42
```
- The data file is the repo copy by default; on Kaggle the notebook sets `STUDENT_DATA_PATH` to the attached dataset.
- `SEED = 42` is used for every random choice, so results are reproducible.

```python
CATEGORICAL = ["school", "sex", ..., "romantic"]        # 17 columns
NUMERIC = ["age", "Medu", ..., "absences"]              # 13 columns
GRADES = ["G1", "G2"]
FEATURES = NUMERIC + GRADES + CATEGORICAL               # 32 inputs
```
- **Categorical** columns have no natural order (for example Mjob), so they get one-hot encoded.
- **Numeric** columns are counts and ordered scales (1–5, 0–4), so they keep their order and get standardised.
- `FEATURES` is the list of all 32 inputs. G3 is not in it: it is the target.

```python
GROUPS = {"Demographic": [...], "Academic": [...], "Social": [...], "Behavioural": [...]}
DESCRIPTIONS = {"school": "School (GP or MS)", ...}
assert sorted(sum(GROUPS.values(), [])) == sorted(FEATURES) == sorted(DESCRIPTIONS)
```
- These are the brief's four groups and human-readable labels, used by the app form and the audit table.
- The `assert` is a self-check: every input must be in exactly one group and have a description. If someone adds a column and forgets one, the import fails immediately.

```python
def load_data(path=DATA_PATH):
    df = pd.read_csv(path, sep=None, engine="python")
    df["at_risk"] = (df["G3"] < 10).astype(int)
    return df
```
- `sep=None, engine="python"` makes pandas *detect* the separator: `;` in the UCI file, `,` in the Kaggle mirror.
- `(df["G3"] < 10)` gives True/False, and `.astype(int)` turns it into 1/0. **1 = at risk** is the positive class, so recall means "share of at-risk students found".

```python
def split(df):
    return train_test_split(df, test_size=0.2, stratify=df["at_risk"], random_state=SEED)
```
- 20% goes to the test set (130 students).
- `stratify` keeps the 15.4% at-risk share the same in both parts.
- `random_state=SEED` gives the same split every run.

```python
def discrete_mi(X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return mutual_info_classif(X, y, discrete_features=True)
```
- This scores each column by **mutual information** with the target: how much knowing the column reduces uncertainty about pass or fail.
- `discrete_features=True` computes MI from counts, because every input is an integer code, a 0/1 flag or a whole-number grade.
- The warning filter hides scikit-learn's note that scaled values "look continuous". They are still the same discrete values, just relabelled.

```python
def make_preprocessor(k="all"):
    encode = ColumnTransformer([
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC + GRADES),
    ], verbose_feature_names_out=False).set_output(transform="pandas")
    select = SelectKBest(discrete_mi, k=k)
    return [("encode", encode), ("select", select)]
```
- **Categorical branch.**
  - `OneHotEncoder` gives one 0/1 column per category.
  - `drop="if_binary"` keeps a single column for yes/no features.
  - `handle_unknown="ignore"` means an unseen category becomes all zeros instead of crashing.
  - `sparse_output=False` returns a normal dense array.
- **Numeric branch.** `StandardScaler` computes z = (x − mean) / std, using the mean and std of the rows it is fitted on.
- **Other columns** (G3 and at_risk) are not listed, so `ColumnTransformer` **drops** them. The target cannot leak in.
- `verbose_feature_names_out=False` and `.set_output("pandas")` keep readable column names such as `school_MS`.
- `SelectKBest(discrete_mi, k)` keeps the k highest-MI columns.
- The function returns a **list of pipeline steps**, so the caller can put a model on the end. Every fit then refits all of it on training rows only.

### D2. `src/model.py`: the neural network and its training

```python
class MLP(nn.Module):
    def __init__(self, n_inputs, hidden, dropout, batchnorm):
        super().__init__()
        layers, width = [], n_inputs
        for units in hidden:
            linear = nn.Linear(width, units)
            nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
            nn.init.zeros_(linear.bias)
            layers += [linear, nn.BatchNorm1d(units) if batchnorm else nn.Identity(), nn.ReLU(), nn.Dropout(dropout)]
            width = units
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)
```
- `nn.Module` is PyTorch's base class for networks. `super().__init__()` registers the layers so their weights are tracked.
- `hidden` is the list of layer sizes, (16, 8) for the final model. `width` tracks the size of the previous layer.
- `nn.Linear(width, units)` computes **y = xWᵀ + b**.
- `kaiming_normal_(..., "relu")` is **He initialisation**: weights ~ N(0, 2 / fan_in), which keeps the signal's variance stable through ReLU layers. Biases start at 0.
- **Each hidden block** is Linear → **BatchNorm** (normalise each unit over the batch, then learned scale and shift) → **ReLU** (max(0, z)) → **Dropout** (randomly zero units during training).
- `nn.Identity()` means "do nothing", so the BatchNorm slot can be switched off.
- The final `nn.Linear(width, 1)` outputs one **logit** z, and P(at risk) = sigmoid(z).
- `nn.Sequential(*layers)` chains the layers in order.
- `forward` runs the chain. `.squeeze(-1)` turns shape (batch, 1) into (batch,), to match the labels.

```python
class DNNClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, hidden=(32, 16), dropout=0.2, batchnorm=True, lr=1e-3, weight_decay=1e-3,
                 batch_size=32, class_weight=True, max_epochs=300, patience=30, n_models=5, seed=42):
```
- This wraps the network in the **scikit-learn estimator API** (`fit`, `predict_proba`, `predict`), so the DNN fits in the same Pipeline, `cross_validate` and `GridSearchCV` code as logistic regression or XGBoost. That makes the comparison fair.
- The constructor only stores hyperparameters, a scikit-learn rule that lets `set_params` and cloning work. The tuned values come from `best_params.json` through `build_dnn`.

```python
    def fit(self, X, y):
        X, y = np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)
        self.classes_ = np.array([0, 1])
        runs = [self._train_one(X, y, self.seed + i) for i in range(self.n_models)]
        self.models_ = [model for model, _ in runs]
        self.history_ = runs[0][1]
        self.best_epoch_ = int(np.argmin(self.history_["val"])) + 1
        return self
```
- Converts the inputs to float32 arrays, the type PyTorch uses.
- `classes_` is required by scikit-learn for classifiers.
- It trains `n_models` (5) networks with seeds 42, 43, …, 46, which is the **seed ensemble**.
- It keeps the first network's loss history for the learning-curve plot, and its best epoch.

```python
    def _train_one(self, X, y, seed):
        torch.manual_seed(seed)
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, stratify=y, random_state=seed)
        X_tr, y_tr, X_val, y_val = map(torch.from_numpy, (X_tr, y_tr, X_val, y_val))
```
- `torch.manual_seed` fixes the random initialisation and batch order for this network.
- 15% of the *rows passed to `fit`* become the early-stopping validation set, stratified. In CV those rows are already a training fold, so **the test set is never touched**.
- `torch.from_numpy` turns the arrays into tensors.

```python
        model = MLP(X.shape[1], self.hidden, self.dropout, self.batchnorm)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        pos_weight = (len(y_tr) - y_tr.sum()) / y_tr.sum() if self.class_weight else torch.tensor(1.0)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```
- `X.shape[1]` is the number of inputs, 10 after feature selection.
- **AdamW** adapts the step size per weight (Adam) and applies **weight decay** separately (the W).
- `pos_weight = #pass / #fail ≈ 5.5` would make each at-risk student count 5.5 times. Optuna turned it off (`class_weight=False`), so it is 1.
- `BCEWithLogitsLoss` is binary cross-entropy computed straight from the logit, which is numerically stable.

```python
        best_loss, bad_epochs, history = float("inf"), 0, {"train": [], "val": []}
        best_state = copy.deepcopy(model.state_dict())
        for epoch in range(self.max_epochs):
            model.train()
            order, total = torch.randperm(len(X_tr)), 0.0
            for start in range(0, len(X_tr), self.batch_size):
                idx = order[start:start + self.batch_size]
                if len(idx) < 2:
                    continue
                optimizer.zero_grad()
                loss = loss_fn(model(X_tr[idx]), y_tr[idx])
                loss.backward()
                optimizer.step()
                total += loss.item() * len(idx)
```
- `best_state` stores a copy of the best weights so far.
- `model.train()` turns dropout **on** and makes BatchNorm use **batch** statistics.
- `torch.randperm` reshuffles the rows every epoch, which is mini-batch SGD's randomness.
- The inner loop walks through mini-batches of 32. A batch of 1 row is skipped, because BatchNorm cannot compute a variance from one row.
- **The 4 lines of training:**
  1. `zero_grad()` clears the old gradients;
  2. `model(...)` is the **forward pass** and `loss_fn` computes the loss;
  3. `loss.backward()` is **backpropagation**: autograd applies the chain rule to fill every weight's `.grad`;
  4. `optimizer.step()` updates the weights.
- `total` accumulates the epoch's training loss for the learning curve.

```python
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(X_val), y_val).item()
            history["train"].append(total / len(X_tr))
            history["val"].append(val_loss)
            if val_loss < best_loss:
                best_loss, bad_epochs, best_state = val_loss, 0, copy.deepcopy(model.state_dict())
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break
        model.load_state_dict(best_state)
        model.eval()
        return model, history
```
- `model.eval()` turns dropout **off** and makes BatchNorm use its **running** averages. `torch.no_grad()` skips gradient tracking, which is faster.
- **Early stopping:**
  - if the validation loss improved, save the weights and reset the counter;
  - otherwise count up, and after `patience` (30) bad epochs, stop.
- `load_state_dict(best_state)` **restores the best epoch's weights**, not the last epoch's.

```python
    def predict_proba(self, X):
        X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            p = np.mean([torch.sigmoid(m(X)).numpy() for m in self.models_], axis=0)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
```
- Each of the 5 networks gives logits. `sigmoid` turns them into probabilities, and `np.mean` **averages the ensemble**.
- `column_stack([1 - p, p])` follows scikit-learn's format: column 0 is P(pass), column 1 is P(at risk).
- `predict` applies the 0.5 cut-off.

### D3. `src/tune.py`: Optuna hyperparameter search

```python
os.environ.setdefault("OMP_NUM_THREADS", "1")
N_TRIALS = 40
```
- One thread per process: fastest for tiny data, and it avoids an OpenMP crash between torch and xgboost on macOS. It must be set before those libraries load.
- 40 trials.

```python
def build_dnn(params, n_models=5):
    hidden = tuple(max(8, params["width"] // 2**i) for i in range(params["n_layers"]))
    dnn = DNNClassifier(hidden=hidden, dropout=..., ..., n_models=n_models)
    return Pipeline(make_preprocessor(params["k"]) + [("model", dnn)])
```
- Turns the flat parameter dictionary into layer sizes with a **funnel** shape: each layer has half the width of the previous one, with a minimum of 8. Width 16 with 2 layers gives (16, 8).
- Returns **preprocessing + DNN as one Pipeline**. This function is used by tuning, evaluation and the notebook, so all of them build the identical model.

```python
def objective(trial, train):
    params = {"n_layers": trial.suggest_int("n_layers", 1, 3), "width": trial.suggest_categorical(...), ...}
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
    scores = cross_val_score(build_dnn(params, n_models=1), train, train["at_risk"], cv=cv, scoring="f1_macro", n_jobs=-1)
    ... append the trial to results/experiments.csv ...
    return scores.mean()
```
- Optuna calls `objective` 40 times. Each `trial.suggest_*` call draws one hyperparameter value.
  - Learning rate and weight decay are drawn on a **log scale**, because 0.0001 and 0.001 are as different as 0.001 and 0.01.
- Each trial is scored by **macro-F1** averaged over 10 folds (5-fold × 2 repeats), using **training data only**.
- Tuning trains 1 network per fold for speed; the final model uses 5.
- `n_jobs=-1` runs the folds in parallel on all CPU cores.
- Every trial is appended to the CSV log, which is the experiment log.

```python
def main():
    train, _ = split(load_data())
    if os.path.exists("results/experiments.csv"):
        os.remove("results/experiments.csv")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(lambda t: objective(t, train), n_trials=N_TRIALS)
    json.dump(study.best_params, open("results/best_params.json", "w"), indent=2)
```
- `train, _ = ...` throws away the test set: **tuning never sees it**.
- The old log is deleted so the file describes this study only.
- The TPE sampler is seeded, so the same 40 trials happen every run (verified from a fresh clone).
- The best parameters are saved for `evaluate.py`.

### D4. `src/evaluate.py`: comparison, test set and statistics

```python
SCORING = {"accuracy": ..., "balanced_acc": ..., "precision_at_risk": make_scorer(precision_score, zero_division=0),
           "recall_at_risk": "recall", "f1_macro": "f1_macro", "roc_auc": "roc_auc"}
```
- The six metrics computed in every CV fold.
- `zero_division=0` defines precision as 0 for the majority-class model, which never predicts "at risk" (0/0).

```python
class G2Rule(ClassifierMixin, BaseEstimator):
    def fit(self, X, y): self.classes_ = np.array([0, 1]); return self
    def predict_proba(self, X): p = (20 - X["G2"].to_numpy()) / 20; return np.column_stack([1 - p, p])
    def predict(self, X): return (X["G2"].to_numpy() < 10).astype(int)
```
- The domain baseline, "at risk if G2 < 10", written as a scikit-learn model so it runs through the same CV and metrics.
- `fit` learns nothing.
- `predict_proba` gives a risk score (lower G2 means a higher score), needed for ROC-AUC.

```python
def tuned_baselines(train):
    grids = {"Logistic regression": (pipe(LogisticRegression(max_iter=2000)), {"model__C": [...], ...}),
             "Random forest": (...), "XGBoost": (...)}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    models = {"Majority class": DummyClassifier(strategy="most_frequent"), "Naive rule (G2 < 10)": G2Rule()}
    for name, (estimator, grid) in grids.items():
        search = GridSearchCV(estimator, grid, cv=cv, scoring="f1_macro", n_jobs=-1).fit(train, train["at_risk"])
        models[name] = search.best_estimator_
```
- Each traditional model sits in **the same preprocessing pipeline** (`pipe(...)`) and gets its own grid search with **the same CV and metric** the DNN was tuned with. Without that, the comparison would not be fair.
- `DummyClassifier("most_frequent")` always predicts "pass". It is the floor every model must beat.
- `model__C` means "parameter C of the pipeline step called model". `select__k` lets logistic regression also choose how many features to keep.

```python
def evaluate(params, train, test):
    models = tuned_baselines(train); models[DNN] = build_dnn(params)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=7)
    for name, model in models.items():
        s = cross_validate(model, train, train["at_risk"], cv=cv, scoring=SCORING, n_jobs=-1)
        rows[name] = {mean of each metric} | {std of accuracy and macro-F1}
        models[name] = model.fit(train, train["at_risk"])
        rows[name] |= test_metrics(models[name], test)
```
- **CV:** 5 folds × 3 repeats = 15 scores per model, on the training students only. Seed 7 differs from tuning's seed 42, so these are fresh splits.
- Then each model is **refitted on all 519 training students** and scored **once** on the 130 test students.

```python
    seeds = [test_metrics(build_dnn(params).set_params(model__seed=100 * s).fit(train, ...), test) for s in range(5)]
    extra = {"seed_mean": ..., "seed_std": ..., "bootstrap_95ci": ..., "wilson_95ci_accuracy": ...,
             "confusion_matrix": ..., "classification_report": ..., "mcnemar_vs": ..., "selected_features": ...}
    joblib.dump(dnn, "models/dnn.joblib")
```
- **5-seed check:** retrain the whole DNN with 5 different seeds to see how much the test score depends on luck (± 0.008).
- `bootstrap_ci` resamples the 130 test students 1,000 times and takes the 2.5th and 97.5th percentiles.
- `wilson_ci` is the formula interval for a proportion, better than ±1.96·SE on small samples.
- `mcnemar` counts students where only one of two models is right; if the models were equal each such student would be a coin flip, giving an exact binomial p-value.
- `joblib.dump` saves the **whole fitted pipeline**, which is the file the app loads.
- `plots()` draws the confusion matrix, ROC curves and learning curve; `comparison_plot()` draws the DNN-vs-traditional chart; `write_report()` writes `RESULTS.md`.

### D5. `src/numpy_mlp.py`: a network from scratch (CHO lab 6)

```python
self.params = {"W1": rng.normal(0, np.sqrt(2 / n_inputs), (n_inputs, n_hidden)), "b1": np.zeros(n_hidden),
               "W2": rng.normal(0, np.sqrt(2 / n_hidden), (n_hidden, 1)), "b2": np.zeros(1)}
```
He initialisation by hand: std = √(2 / fan_in).

```python
def forward(self, X):
    z1 = X @ P["W1"] + P["b1"]      # weighted sums
    a1 = np.maximum(z1, 0)          # ReLU
    z2 = a1 @ P["W2"] + P["b2"]     # logit
    return sigmoid(z2[:, 0]), (z1, a1)
```
The forward pass as matrix maths. It returns the intermediate values `z1` and `a1`, which backprop needs.

```python
def loss(self, X, y):
    p = np.clip(self.forward(X)[0], 1e-7, 1 - 1e-7)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
```
Binary cross-entropy. `clip` avoids log(0).

```python
def gradients(self, X, y):
    dz2 = ((p - y) / n)[:, None]    # dL/dz2: sigmoid + cross-entropy simplify to (p - y)
    da1 = dz2 @ P["W2"].T           # error sent back through W2
    dz1 = da1 * (z1 > 0)            # ReLU derivative: 1 where z1 > 0, else 0
    return {"W2": a1.T @ dz2, "b2": dz2.sum(0), "W1": X.T @ dz1, "b1": dz1.sum(0)}
```
**Backpropagation by the chain rule**, line by line:
1. The output error is p − y.
2. It flows back through W2.
3. It is gated by ReLU's derivative.
4. Each weight's gradient = (input to that layer)ᵀ × (error at that layer).

```python
def fit(...):
    for each epoch: shuffle; for each mini-batch: for name, g in gradients(...): params[name] -= lr * g
```
Mini-batch gradient descent: **w ← w − learning_rate × gradient**.

```python
def gradient_check(model, X, y, eps=1e-5):
    W[i] = old + eps; plus = loss;  W[i] = old - eps; minus = loss;  numeric = (plus - minus) / (2 * eps)
    relative error = |numeric - analytic| / (|numeric| + |analytic|)
```
- **Numerical gradient checking** (CHO lab 2): nudge one weight up and down, measure the loss change, and compare with backprop's answer.
- An error below 1e-6 means backprop is correct; we get 2e-9.
- `python -m src.numpy_mlp` runs this as a self-test.

### D6. `src/experiments.py`: the course-practical experiments

- **`prepare(train)`:** splits the **training students** 85/15, fits the preprocessing on the 85% only, and returns tensors.
- **`make_optimizer(name, ...)`:** a dictionary from names to `torch.optim.SGD` (optionally with `momentum=0.9`), `RMSprop`, `Adam` or `AdamW`.
- **`run(data, optimizer, lr, batch_size, init, activation, batchnorm, schedule, epochs)`:**
  1. Builds the final architecture.
  2. **Swaps the activation** (ReLU → Tanh or Sigmoid) and **re-initialises weights** (Xavier or zeros) when asked.
  3. Wraps the data in **`TensorDataset` + `DataLoader(shuffle=True)`**, PyTorch's standard batching (CHO lab 10).
  4. Optionally attaches a **learning-rate schedule**: `StepLR` (halve every 30 epochs) or `ExponentialLR` (×0.97 per epoch).
  5. Runs a plain train loop plus a **validation loop** for 100 epochs and records the losses, validation accuracy and time.
- **`summary(results)`:** best validation loss, its epoch, validation accuracy at that epoch, final loss and seconds.
- **`feature_ablation(train, params)`:** the add-back / remove test behind the app's "Feature choices" tab.
  1. It builds a pipeline that keeps a **fixed column list**: `ColumnTransformer([("keep", "passthrough", cols)])` after encoding.
  2. It scores the final 10 inputs, then each variant (one input removed or added), all with `cross_validate` on the same 15 folds. The per-fold differences give a paired mean ± standard error.
  3. It also retrains the 10-input model with 4 more seeds to measure the noise floor.
  4. Results are saved to `results/feature_ablation.csv` and `feature_ablation_baseline.json`.
- **No early stopping here,** on purpose: fixed 100 epochs keeps the curves comparable.

### D7. `src/eda.py`: audit, plots, feature audit

- **`audit(df)`:** prints the shape, missing values, duplicates, at-risk share, the G3 = 0 analysis, correlations of G1/G2 with G3, the G2-rule accuracy and the absences skew.
- **`figures(df)`:** five matplotlib plots saved to `results/figures/`.
- **`feature_audit(train)`:**
  1. For each of the 15 evaluation CV folds, it refits the preprocessor with k = 10 on that fold's training rows and records which raw columns survive. A one-hot column such as `Mjob_teacher` counts for `Mjob`.
  2. Then it builds the per-feature table: rank correlation with G3, the at-risk rate gap between categories, kept-in-folds and in-final-model.
- **Also in `feature_audit`:** each input's mutual-information value and rank (1–45, for its best encoded column). `python -m src.eda` saves the table to `results/feature_audit.csv` for the app.
- **`vif(train)`:** the variance inflation factor for all numeric columns at once, as the diagonal of the inverse correlation matrix. VIF = 1 / (1 − R²).

### D8. `app.py`: the Streamlit app

```python
import src.model
from src.data import CATEGORICAL, DESCRIPTIONS, FEATURES, GROUPS, load_data, split
```
- The saved pipeline contains a `DNNClassifier`. Python can only unpickle it if `src.model` is importable, which is why that import is there.
- It reuses the same feature lists, groups and labels as training (single source of truth).

```python
THRESHOLD = 0.5
CHOICES = {"school": {"GP": "Gabriel Pereira", ...}, ...}
SLIDERS = {"age": (15, 22), "G1": (0, 20), "G2": (0, 20), "failures": (0, 3)}
```
- `THRESHOLD` is the decision cut-off.
- `CHOICES` gives readable labels for coded values.
- `SLIDERS` restricts inputs to the codebook ranges. This is the input validation: impossible values cannot be entered.

```python
st.set_page_config(...); st.markdown("""<style> ... </style>""", unsafe_allow_html=True)
```
- Page title and wide layout.
- A little CSS: the PASS / AT RISK badges (green and red), muted text, a **sticky** result panel on desktop, and on phones the result shown above the form.

```python
@st.cache_resource
def load_model(): return joblib.load("models/dnn.joblib")

@st.cache_data
def load_reference():
    train, _ = split(load_data())
    metrics = json.load(open("results/metrics.json"))
```
- `cache_resource` loads the model **once**. It never retrains.
- `load_reference` recreates the same 519 training students (same seed) for the default values, plus the saved metrics for the Model tab.

```python
used = [c for c in FEATURES if any(n == c or n.startswith(c + "_") for n in dnn["selected_features"])]
typical = {c: train[c].mode()[0] if c in CATEGORICAL else int(train[c].median()) for c in FEATURES}
```
- `used` maps the 10 selected encoded columns back to raw inputs (`school_MS` → `school`), so the form can mark "used by model".
- `typical` is the default for each input: the most common category, or the median number, among training students.

```python
def field(container, name):
    label = DESCRIPTIONS[name] + (" · used by model" if name in used else "")
    if name in SLIDERS: return container.slider(label, *SLIDERS[name], typical[name], key=name)
    if name == "absences": return container.number_input(label, 0, 93, typical[name], key=name)
    options = sorted(train[name].unique())
    return container.selectbox(label, options, index=options.index(typical[name]), key=name, format_func=...)
```
- One function draws any input: a slider for ranges, a number box for absences, and a dropdown for coded values.
- Dropdown options come from the training data, so only valid categories can be chosen.

```python
def risk(rows):
    return model.predict_proba(pd.DataFrame(rows)[FEATURES])[:, 1]
```
- **The inference line.** It builds a DataFrame, orders the columns **exactly as in training** (`[FEATURES]`), and runs the saved pipeline: encode → scale → select → 5 networks → averaged probability.

```python
for box, (group, cols) in zip([*top, *bottom], GROUPS.items()):
    with box.container(border=True):
        for c in cols: student[c] = field(st, c)
```
Draws the four bordered boxes (Demographic, Academic, Social, Behavioural) in a 2 × 2 grid and collects all 32 answers into the `student` dictionary.

```python
p = float(risk([student])[0]); at_risk = p >= THRESHOLD
... badge PASS / AT RISK, "Estimated probability of failing", "Close to the cut-off" if |p - 0.5| < 0.15, progress bar ...
if student["absences"] > train["absences"].max(): st.warning(...)
```
- Shows the prediction. Within 15 points of the cut-off it says the result is uncertain.
- It warns when absences go beyond anything seen in training (32).

```python
changed = [c for c in used if student[c] != typical[c]]
effects = dict(zip(changed, p - risk([student | {c: typical[c]} for c in changed])))
shown = [(c, d) for c, d in sorted(effects.items(), key=lambda kv: -abs(kv[1])) if abs(d) >= 0.01][:4]
```
- **Occlusion explanation:**
  1. for each used input that differs from typical, make a copy of the student with only that input set back to typical;
  2. predict all the copies in one call;
  3. the difference from `p` is that input's effect.
- It shows the 4 biggest effects of at least 1 point.

**Model performance tab:** the test metrics from `metrics.json`, the confusion matrix, the ROC curves, the comparison table and chart, and a one-line architecture summary.
**Feature choices tab:**
- explains the selection rule and the add-back / remove test;
- has a **"Look up any input"** box (default: address) showing the input's reason, mutual-information rank, folds kept and CV change;
- lists all 32 inputs with reasons in two tables, used and not used.

Every input in the Predict form also has a **(?) tooltip** with its reason (`help=why` in `field()`, text from `REASONS` in `src/data.py`).

**About tab:** project, data source, method and a responsible-use note.

---

## E. Engineering decisions mapped to the CHO: what we used, and why not the alternatives

| CHO topic (lecture / lab) | We used | Why | Alternatives, and why not |
|---|---|---|---|
| Python for DL: preprocessing, missing values, scaling, encoding (L8–9, Lab 3) | Checked for missing values (none); one-hot encoding; StandardScaler; all inside a Pipeline | Nominal categories have no order; standardised inputs make gradient descent converge faster; the Pipeline prevents leakage | Label encoding invents an order; min-max scaling is sensitive to the absences outlier; imputation is unnecessary (0 missing) |
| Train / validation / test, overfitting (L8–9) | Stratified 80/20 locked test; 5×3 CV; 15% early-stopping validation from training rows | Honest estimates and no leakage; stratification keeps 15.4% at risk everywhere | A single validation split wastes data; nested CV is about 40× the compute (its optimism is disclosed instead) |
| Perceptron, activations: ReLU, sigmoid, tanh, softmax (L10–12, Lab 4–5) | ReLU in hidden layers, sigmoid output | ReLU's gradient doesn't vanish; it won the experiment (0.185 vs tanh 0.255, sigmoid 0.270) | Sigmoid and tanh hidden layers saturate; softmax with 2 outputs is equivalent to one sigmoid with twice the weights |
| Loss functions: MSE, BCE, CCE (L13–14, Lab 5) | BCEWithLogitsLoss | Maximum likelihood for a binary label; gradient p − y; numerically stable | MSE + sigmoid has a vanishing gradient when the model is confidently wrong; CCE is for more than 2 classes |
| Backpropagation, gradient checking (L13–16, Lab 2, 6) | PyTorch autograd; hand-written NumPy backprop with gradient check | Autograd for the real model; NumPy proves we understand the chain rule (error 2e-9) | – |
| Batch, SGD and mini-batch GD (L15–16, Lab 7) | Mini-batch of 32 | Batch GD: 1 step per epoch, slow (0.307); SGD: noisy, overfits, 20× slower; mini-batch: balanced | Shown in notebook 15c |
| Initialisation: zero, Xavier, He (L17–19, Lab 8) | He | Designed for ReLU; slightly better than Xavier here (0.185 vs 0.205) | Zeros never learns (symmetry, 0.595); Xavier is designed for tanh and sigmoid |
| BatchNorm, LayerNorm (L17–19, Lab 8) | BatchNorm1d | Stabilises training; Optuna kept it in 35 of 40 trials | LayerNorm suits Transformers and RNNs, with no benefit for a small MLP |
| L1, L2, dropout, early stopping (L17–19, Lab 8) | Dropout 0.48, weight decay (L2-style via AdamW), early stopping with patience 30 | 519 rows need strong regularisation; the learning curve shows no overfitting | L1 is redundant with feature selection; a fixed epoch count would have to be guessed |
| NumPy ANN from scratch (L20–21, Lab 6) | `src/numpy_mlp.py` | Shows forward pass, loss, backprop and update by hand | – |
| Keras Sequential, Functional, callbacks, checkpoints (L22–23, Lab 9) | Keras twin with both APIs; EarlyStopping; ModelCheckpoint; reload | Covers the Keras workflow and gives the same network | Keras for the main model: PyTorch's explicit loop is easier to explain line by line |
| PyTorch tensors, autograd, Dataset/DataLoader, nn.Module, training loop (L24–26, Lab 10) | Main model in PyTorch; DataLoader in the experiments | `nn.Module` + explicit loop; wrapped as a scikit-learn estimator for a fair comparison | – |
| Optimisers: SGD, momentum, RMSprop, Adam, AdamW (L24–26, Lab 11) | AdamW | Adaptive and robust to the learning rate; weight decay applied correctly | SGD: slowest (0.344); momentum: good but needs learning-rate tuning; Adam: L2 interacts badly with adaptive steps |
| Learning-rate schedules, grid / random / Optuna (L27–29, Lab 11) | Constant learning rate + early stopping; Optuna TPE (40 trials); grid search for the baselines | Schedules didn't help at lr 0.001 (step 0.232, exponential 0.335); TPE handles 9 dimensions efficiently | A grid over 9 hyperparameters is about 20,000 runs; random search doesn't learn (TPE starts with 10 random trials anyway) |
| Metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrix (L27–29) | All of them, plus macro-F1, balanced accuracy, Wilson and bootstrap CIs, McNemar | Imbalance makes accuracy alone misleading; small test set needs intervals | Accuracy-only reporting would hide that "everyone passes" already scores 84.6% |
| Learning curves, debugging, experiment log, leakage, reproducibility (L27–29) | Learning-curve plot, `experiments.csv`, seeds, pinned versions, a fresh-clone check | All required for a trustworthy result | – |
| CNN, transfer learning (CLO04, Lab 12–14) | Not used | Tabular data has no spatial structure | – |
| RNN, LSTM, GRU (CLO05, Lab 15) | Not used | No sequences or time order | – |
| Hardware | CPU | Faster than the GPU for this network (M3 0.25 s vs 0.74 s; Kaggle 1.66 s vs T4 1.81 s) | GPU launch overhead exceeds the maths for tiny matrices |

Status per lab (implemented, partial, not implemented or not required) is in `DECISIONS.md`, section 6.

---

## F. The comparison the brief asks for: DNN vs traditional ML

**What the brief says:** *"The developed model must be compared with at least one traditional Machine Learning algorithm."* We compare against **three tuned traditional models and two baselines**.

**How the comparison is made fair:**
1. **Same data and split:** 519 training students, the same 130 locked test students.
2. **Same preprocessing:** every model sits behind the same one-hot + scaling pipeline. Logistic regression also tunes k; the tree models use all columns, because trees do their own selection.
3. **Same tuning effort and metric:** each traditional model gets a grid search with 5-fold CV on macro-F1. The DNN gets Optuna on macro-F1.
4. **Same evaluation:** 5-fold × 3-repeat CV (15 folds, seed 7) for all, then one test score each.
5. **Significance test:** McNemar's exact test, DNN vs each model, on the same test students.

**Models compared:**

| Model | Type | Tuned over |
|---|---|---|
| Majority class | Baseline: always predicts "pass" | – |
| Rule "G2 < 10" | Domain baseline (Cortez & Silva 2008) | – |
| Logistic regression | Linear model | C, class weight, k features |
| Random forest | Bagged decision trees (300) | max depth, min leaf size, class weight |
| XGBoost | Gradient-boosted trees | max depth, number of trees, positive-class weight |
| **DNN** | Our MLP | Optuna, 9 hyperparameters |

**Results** (`results/figures/comparison.png`, `results/RESULTS.md`):

| Model | CV accuracy | CV macro-F1 | CV recall | Test accuracy | Test precision | Test recall | Test macro-F1 | Test ROC-AUC | McNemar p vs DNN |
|---|---|---|---|---|---|---|---|---|---|
| Majority class | 0.846 | 0.458 | 0.000 | 0.846 | 0.000 | 0.00 | 0.458 | 0.500 | 0.078 |
| Rule G2 < 10 | 0.896 | 0.831 | 0.887 | 0.900 | 0.621 | 0.90 | 0.837 | 0.953 | 0.754 |
| Logistic regression | **0.934** | **0.870** | 0.771 | 0.900 | 0.652 | 0.75 | 0.819 | 0.944 | 0.754 |
| Random forest | 0.921 | 0.867 | **0.908** | 0.862 | 0.531 | 0.85 | 0.784 | 0.946 | 0.065 |
| XGBoost | 0.929 | 0.866 | 0.804 | 0.900 | 0.640 | 0.80 | 0.825 | 0.925 | 0.727 |
| **DNN** | 0.925 | 0.846 | 0.671 | **0.915** | **0.714** | 0.75 | **0.841** | **0.953** | – |

**What it means, and how to say it in the viva:**
1. **On the locked test set** the DNN is the best: highest accuracy, precision, macro-F1 and (tied) ROC-AUC.
2. **In cross-validation** logistic regression and the tree models are slightly ahead on macro-F1, and random forest catches the most at-risk students.
3. **McNemar:** no difference is statistically significant (all p > 0.05). The DNN and the best traditional models are **equally good**.
4. **Why that is the expected result:**
   - with G1 and G2 known, the problem is almost linearly separable, so even logistic regression reaches about 93%;
   - on small tabular datasets, tree ensembles and linear models usually match neural networks (Grinsztajn et al., NeurIPS 2022).
5. **What the DNN adds over the one-line rule:** it combines the grades with past failures, higher-education plans, school and absences. That is how it separates borderline students (`DEMO_SCRIPT.md`, step 5) and gets 91.5% vs 90.0% on test (92.5% vs 89.6% in CV). With 130 test students that gain is not significant.

**One-sentence answer:** "We compared the DNN with three tuned traditional models and two baselines under identical preprocessing and cross-validation. The DNN reached the best test accuracy, 91.5%, but McNemar's test shows it is statistically tied with logistic regression and XGBoost, so the honest conclusion is that it matches the best traditional models."
