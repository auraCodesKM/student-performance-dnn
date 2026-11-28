# The code, explained line by line

Every line of the project's code, in the order it runs, with a plain explanation of each block.
Files: `src/data.py` (step 1, data) → `src/model.py` (step 2, network) → `src/train.py` (step 3, choices, training, comparison) → `src/eda.py` (data checks) → `app.py` (the app).

## `src/data.py` (75 lines)

*Step 1: load the data, create the target, split it, and prepare the inputs.*

**Line 1.** The file's description.

```python
  1  """Step 1: load the data, create the target (Pass / At risk), split it, and prepare the inputs for the models."""
```

**Lines 2–10.** Libraries used here. `pandas` reads tables. From scikit-learn: `ColumnTransformer` treats text and number columns differently; `SelectKBest` and `mutual_info_classif` choose the best columns; `train_test_split` makes the train/test split; `Pipeline` chains steps together; `OneHotEncoder` and `StandardScaler` encode and rescale.

```python
  2  import os
  3  import warnings
  4  
  5  import pandas as pd
  6  from sklearn.compose import ColumnTransformer
  7  from sklearn.feature_selection import SelectKBest, mutual_info_classif
  8  from sklearn.model_selection import train_test_split
  9  from sklearn.pipeline import Pipeline
 10  from sklearn.preprocessing import OneHotEncoder, StandardScaler
```

**Lines 12–13.** Where the data file is. Normally `data/student-por.csv`; on Kaggle the notebook switches it to Kaggle's copy of the same file.

```python
 12  # The UCI file in this repo. (On Kaggle the notebook points this at Kaggle's copy of the same file.)
 13  DATA_PATH = os.environ.get("STUDENT_DATA_PATH", "data/student-por.csv")
```

**Line 14.** A fixed random number, so every run gives exactly the same split and results.

```python
 14  SEED = 42  # fixed random seed, so every run gives the same result
```

**Lines 16–18.** The 17 text columns, which will be one-hot encoded.

```python
 16  # Text columns (categories). They become 0/1 columns ("one-hot encoding").
 17  CATEGORICAL = ["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian",
 18                 "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"]
```

**Lines 19–21.** The 15 number columns, which will be standardised. G1 and G2 are included.

```python
 19  # Number columns. They are rescaled to a similar range ("standardisation").
 20  NUMERIC = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel", "freetime", "goout",
 21             "Dalc", "Walc", "health", "absences", "G1", "G2"]
```

**Line 22.** All 32 inputs. G3 is not in the list, because it is the answer we predict.

```python
 22  FEATURES = NUMERIC + CATEGORICAL  # all 32 inputs. G3 is NOT an input: it is what we predict.
```

**Lines 24–30.** The four groups from the project brief, used to lay out the app's form.

```python
 24  GROUPS = {  # the four groups named in the project brief
 25      "Demographic": ["sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob", "guardian"],
 26      "Academic": ["school", "reason", "G1", "G2", "failures", "studytime", "traveltime", "schoolsup", "famsup", "paid",
 27                   "nursery", "higher"],
 28      "Social": ["famrel", "freetime", "goout", "romantic", "activities", "internet"],
 29      "Behavioural": ["absences", "Dalc", "Walc", "health"],
 30  }
```

**Lines 31–43.** A readable name for every column (from the dataset's codebook), shown in the app.

```python
 31  DESCRIPTIONS = {  # readable names (from the dataset's codebook, data/student.txt)
 32      "school": "School (GP or MS)", "sex": "Sex", "age": "Age (15-22)", "address": "Home address (urban / rural)",
 33      "famsize": "Family size (<=3 / >3)", "Pstatus": "Parents together / apart", "Medu": "Mother's education (0-4)",
 34      "Fedu": "Father's education (0-4)", "Mjob": "Mother's job", "Fjob": "Father's job",
 35      "reason": "Reason for choosing school", "guardian": "Guardian", "traveltime": "Travel time to school (1-4)",
 36      "studytime": "Weekly study time (1-4)", "failures": "Past class failures (0-3)", "schoolsup": "Extra school support",
 37      "famsup": "Family educational support", "paid": "Extra paid classes", "activities": "Extra-curricular activities",
 38      "nursery": "Attended nursery school", "higher": "Wants higher education", "internet": "Internet at home",
 39      "romantic": "In a relationship", "famrel": "Family relationship quality (1-5)",
 40      "freetime": "Free time after school (1-5)", "goout": "Going out with friends (1-5)",
 41      "Dalc": "Workday alcohol use (1-5)", "Walc": "Weekend alcohol use (1-5)", "health": "Current health (1-5)",
 42      "absences": "School absences", "G1": "1st period grade (0-20)", "G2": "2nd period grade (0-20)",
 43  }
```

**Lines 46–48.** Reads the CSV file. `sep=None` lets pandas detect whether the file uses ';' or ','.

```python
 46  def load_data(path=DATA_PATH):
 47      """Read the CSV and add the target column: at_risk = 1 if the final grade G3 is below 10, else 0."""
 48      df = pd.read_csv(path, sep=None, engine="python")  # works with ";" (UCI file) or "," (Kaggle copy)
```

**Lines 49–50.** Creates the target: 1 (AT RISK) if the final grade is below 10, otherwise 0 (PASS).

```python
 49      df["at_risk"] = (df["G3"] < 10).astype(int)
 50      return df
```

**Lines 53–55.** Puts 20% of students (130) aside as the test set and keeps 80% (519) for training. `stratify` keeps about 15% failing students in both parts; `random_state=SEED` makes the split the same every time.

```python
 53  def split(df):
 54      """80% of students for training, 20% kept aside as the test set. 'stratify' keeps 15% at-risk in both parts."""
 55      return train_test_split(df, test_size=0.2, stratify=df["at_risk"], random_state=SEED)
```

**Lines 58–62.** Scores each column by mutual information: how much knowing it tells us about pass / at risk (0 = nothing). `discrete_features=True` because all our values are whole numbers or categories. The warning filter just hides a harmless message.

```python
 58  def information_score(X, y):
 59      """How much each column tells us about pass / at risk (mutual information). 0 = nothing."""
 60      with warnings.catch_warnings():
 61          warnings.simplefilter("ignore")
 62          return mutual_info_classif(X, y, discrete_features=True)  # all our columns are whole numbers or categories
```

**Lines 65–70.** Builds the preparation steps. They are learned from training students only.

```python
 65  def make_preprocessor(k=10):
 66      """Turns the raw student table into the numbers the models use. Fitted on training students only.
 67  
 68      1. One-hot encode the text columns and standardise the number columns  -> 45 columns
 69      2. Keep the k columns with the highest information score                 -> k columns
 70      """
```

**Lines 71–74.** Step 1: `OneHotEncoder` turns each text column into 0/1 columns (`drop="if_binary"` keeps one column for yes/no; `handle_unknown="ignore"` means a new value doesn't crash). `StandardScaler` rescales each number column: (value − average) ÷ spread. Result: 45 columns with readable names.

```python
 71      encode = ColumnTransformer([
 72          ("text", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False), CATEGORICAL),
 73          ("numbers", StandardScaler(), NUMERIC),
 74      ], verbose_feature_names_out=False).set_output(transform="pandas")
```

**Line 75.** Step 2: `SelectKBest` keeps the k columns with the highest information score (k = 5 in the final model). Both steps are returned together as one `Pipeline`.

```python
 75      return Pipeline([("encode", encode), ("select", SelectKBest(information_score, k=k))])
```

## `src/model.py` (34 lines)

*Step 2: the neural network, built with Keras.*

**Line 1.** The file's description.

```python
  1  """Step 2: the deep neural network (DNN), built with Keras."""
```

**Lines 2–4.** Tells Keras to run on top of PyTorch, so TensorFlow is not needed. This must happen before Keras is imported.

```python
  2  import os
  3  
  4  os.environ.setdefault("KERAS_BACKEND", "torch")  # Keras runs on top of PyTorch here (no TensorFlow needed)
```

**Lines 6–7.** Imports Keras and its layers.

```python
  6  import keras
  7  from keras import layers
```

**Lines 10–11.** Builds the network. Settings: number of inputs, hidden layer sizes, dropout rate, learning rate, activation function and optimizer.

```python
 10  def build_model(n_inputs, hidden=(16, 8), dropout=0.3, learning_rate=0.001, activation="relu", optimizer="adam"):
 11      """inputs -> hidden layers (each: Dense + Dropout) -> 1 output with sigmoid = probability of being at risk."""
```

**Line 12.** Fixes the random starting weights, so the same settings always give the same network.

```python
 12      keras.utils.set_random_seed(42)  # same starting weights every time, so results are repeatable
```

**Line 13.** A Sequential model is a stack of layers, one after another. The first item says how many inputs there are.

```python
 13      model = keras.Sequential([keras.Input(shape=(n_inputs,))])
```

**Lines 14–15.** For each hidden layer (16, then 8): a `Dense` layer, where every neuron is connected to every input, with the chosen activation (ReLU) and He initialisation (good starting weights for ReLU).

```python
 14      for units in hidden:
 15          model.add(layers.Dense(units, activation=activation, kernel_initializer="he_normal"))
```

**Line 16.** After each hidden layer, `Dropout` switches off 40% of the neurons at random during training, so the network can't memorise the training students.

```python
 16          model.add(layers.Dropout(dropout))  # randomly switches off neurons while training, to avoid overfitting
```

**Line 17.** The output layer: 1 neuron with sigmoid, giving a number between 0 and 1, the chance of failing.

```python
 17      model.add(layers.Dense(1, activation="sigmoid"))  # output between 0 and 1
```

**Lines 18–21.** Chooses the optimizer, the method that updates the weights: Adam (adjusts its own step size) or SGD (fixed step size). `learning_rate` is the step size.

```python
 18      if optimizer == "adam":
 19          opt = keras.optimizers.Adam(learning_rate=learning_rate)
 20      else:
 21          opt = keras.optimizers.SGD(learning_rate=learning_rate)
```

**Lines 22–23.** Prepares the model for training: the optimizer, the loss (binary cross-entropy, the standard loss for yes/no answers) and accuracy as an extra measure.

```python
 22      model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
 23      return model
```

**Lines 26–27.** Trains the network.

```python
 26  def train_model(model, X, y):
 27      """Train with early stopping: stop when the validation loss hasn't improved for 20 epochs, keep the best weights."""
```

**Line 28.** Early stopping: watches the validation loss and stops when it hasn't improved for 20 epochs, then restores the best weights.

```python
 28      stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
```

**Line 29.** Training: 15% of the training students are held back as the validation set; at most 300 epochs; batches of 32 students; `verbose=0` hides the progress bar. It returns the history of the losses (used for the learning curve).

```python
 29      return model.fit(X, y, validation_split=0.15, epochs=300, batch_size=32, callbacks=[stop], verbose=0)
```

**Lines 32–34.** Returns each student's chance of failing (the sigmoid output).

```python
 32  def predict_at_risk(model, X):
 33      """Probability of being at risk for each student."""
 34      return model.predict(X, verbose=0)[:, 0]
```

## `src/train.py` (170 lines)

*Step 3: choose every setting with cross-validation, train the final network, compare with traditional ML, save everything.*

**Lines 1–7.** The file's description: how choices are made, and that the test students are used only once.

```python
  1  """Step 3: make every choice with cross-validation, train the final DNN, compare it with traditional ML, save everything.
  2  
  3  Run:  python -m src.train        (about 8 minutes on a laptop)
  4  
  5  How choices are made: 5-fold cross-validation on the 519 training students (train on 4/5, check on 1/5, five times).
  6  The 130 test students are used only once, at the very end, to report the final scores.
  7  """
```

**Lines 8–10.** Keras runs on PyTorch.

```python
  8  import os
  9  
 10  os.environ.setdefault("KERAS_BACKEND", "torch")
```

**Lines 12–25.** Libraries: `json` and `joblib` save results and the preprocessor; `matplotlib` draws graphs (Agg = save to file); `numpy`; the three traditional models; the metrics; `GridSearchCV` and `StratifiedKFold` for cross-validation; `Pipeline`.

```python
 12  import json
 13  
 14  import joblib
 15  import matplotlib
 16  
 17  matplotlib.use("Agg")
 18  import matplotlib.pyplot as plt
 19  import numpy as np
 20  from sklearn.ensemble import RandomForestClassifier
 21  from sklearn.linear_model import LogisticRegression
 22  from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
 23  from sklearn.model_selection import GridSearchCV, StratifiedKFold
 24  from sklearn.pipeline import Pipeline
 25  from sklearn.tree import DecisionTreeClassifier
```

**Lines 27–28.** Our own functions from steps 1 and 2.

```python
 27  from src.data import SEED, load_data, make_preprocessor, split
 28  from src.model import build_model, predict_at_risk, train_model
```

**Line 30.** The 5 cross-validation folds (the same for every experiment, so comparisons are fair).

```python
 30  FOLDS = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)  # the same 5 folds for every experiment
```

**Line 31.** The starting settings, before any tuning.

```python
 31  START = {"hidden": (16, 8), "dropout": 0.3, "learning_rate": 0.001, "activation": "relu", "optimizer": "adam"}
```

**Lines 32–33.** Where graphs are saved, and the two colours used.

```python
 32  FIG = "results/figures"
 33  GREEN, GREY = "#16a34a", "#a1a1aa"
```

**Lines 36–38.** Calculates accuracy, precision, recall and F1 for a set of predictions.

```python
 36  def scores(y_true, y_pred):
 37      return {"accuracy": accuracy_score(y_true, y_pred), "precision": precision_score(y_true, y_pred, zero_division=0),
 38              "recall": recall_score(y_true, y_pred), "f1": f1_score(y_true, y_pred)}
```

**Lines 41–43.** 5-fold cross-validation of the neural network with given settings.

```python
 41  def cv_dnn(train, k, settings):
 42      """5-fold cross-validation of the DNN. Scaling and feature selection are re-learned inside each fold (no leakage)."""
 43      folds = []
```

**Lines 44–49.** For each of the 5 folds: learn the preprocessing from the 4 training parts only (no leakage), build and train the network, predict the 5th part (AT RISK if chance ≥ 0.5) and score it.

```python
 44      for fit_rows, check_rows in FOLDS.split(train, train.at_risk):
 45          part, check = train.iloc[fit_rows], train.iloc[check_rows]
 46          prep = make_preprocessor(k).fit(part, part.at_risk)
 47          model = build_model(prep.transform(part).shape[1], **settings)
 48          train_model(model, prep.transform(part), part.at_risk.to_numpy())
 49          folds.append(scores(check.at_risk, predict_at_risk(model, prep.transform(check)) >= 0.5))
```

**Lines 50–52.** Averages the 5 fold scores, prints them and returns them.

```python
 50      result = {m: float(np.mean([f[m] for f in folds])) for m in folds[0]}
 51      print(f"  k={k!s:>3} {settings} -> CV accuracy {result['accuracy']:.3f}, F1 {result['f1']:.3f}", flush=True)
 52      return result
```

**Lines 55–56.** Picks the option with the highest F1.

```python
 55  def best_by_f1(options):
 56      return max(options, key=lambda name: options[name]["f1"])
```

**Line 59.** Runs everything.

```python
 59  def main():
```

**Lines 60–61.** Loads the data, makes the split and gets the answers (0/1) for both parts.

```python
 60      train, test = split(load_data())
 61      y_train, y_test = train.at_risk.to_numpy(), test.at_risk.to_numpy()
```

**Lines 63–67.** Experiment A: cross-validate with 5, 10, 20 and all columns, and keep the best number (5 won).

```python
 63      # A. How many input columns to keep?
 64      print("A. number of features")
 65      features = {str(k): cv_dnn(train, k, START) for k in [5, 10, 20, "all"]}
 66      k = best_by_f1(features)
 67      k = k if k == "all" else int(k)
```

**Lines 69–74.** Experiment B (grid search): 2 layer sizes × 2 dropout rates × 2 learning rates = 8 combinations, each cross-validated; keep the best (16-8, dropout 0.4, learning rate 0.01).

```python
 69      # B. Grid search: 2 layer sizes x 2 dropout rates x 2 learning rates = 8 combinations
 70      print("B. grid search")
 71      grid = {f"{h[0]}-{h[1]} | dropout {d} | lr {lr}": START | {"hidden": h, "dropout": d, "learning_rate": lr}
 72              for h in [(16, 8), (32, 16)] for d in [0.2, 0.4] for lr in [0.001, 0.01]}
 73      grid_scores = {name: cv_dnn(train, k, s) for name, s in grid.items()}
 74      settings = grid[best_by_f1(grid_scores)]
```

**Lines 76–81.** Experiments C and D: test ReLU vs tanh vs sigmoid (ReLU won), then Adam vs SGD (Adam won). Each time the winner is kept.

```python
 76      # C. Which activation function?  D. Which optimizer?
 77      print("C. activation"); activation = {a: cv_dnn(train, k, settings | {"activation": a}) for a in ["relu", "tanh", "sigmoid"]}
 78      settings = settings | {"activation": best_by_f1(activation)}
 79      print("D. optimizer"); optimizer = {o: cv_dnn(train, k, settings | {"optimizer": o}) for o in ["adam", "sgd"]}
 80      settings = settings | {"optimizer": best_by_f1(optimizer)}
 81      print("final settings:", k, settings)
```

**Lines 83–88.** Train the final network on all 519 training students: learn the preprocessing, build the network with the chosen settings, train it, then predict the 130 test students once.

```python
 83      # E. Train the final DNN on all 519 training students, then test once on the 130 test students
 84      prep = make_preprocessor(k).fit(train, y_train)
 85      X_train, X_test = prep.transform(train), prep.transform(test)
 86      dnn = build_model(X_train.shape[1], **settings)
 87      history = train_model(dnn, X_train, y_train)
 88      dnn_pred = (predict_at_risk(dnn, X_test) >= 0.5).astype(int)
```

**Lines 89–90.** The network's cross-validation score is the score from experiment D for the chosen optimizer (exactly the final settings), plus its test scores.

```python
 89      # The last experiment (D) already cross-validated exactly these final settings, so that is the DNN's CV score.
 90      results = {"DNN (neural network)": {"cv": optimizer[settings["optimizer"]], "test": scores(y_test, dnn_pred)}}
```

**Lines 92–97.** The three traditional models, each with a small grid of its main setting: logistic regression (C: regularisation strength), decision tree (max depth), random forest (300 trees, max depth).

```python
 92      # F. Traditional machine learning: same inputs, same folds, each with a small grid search of its main setting
 93      traditional = {
 94          "Logistic regression": (LogisticRegression(max_iter=2000), {"model__C": [0.01, 0.1, 1, 10]}),
 95          "Decision tree": (DecisionTreeClassifier(random_state=SEED), {"model__max_depth": [2, 3, 5, 8]}),
 96          "Random forest": (RandomForestClassifier(n_estimators=300, random_state=SEED), {"model__max_depth": [3, 6, None]}),
 97      }
```

**Lines 98–103.** For each: put the same preprocessing in front, run a grid search with the same 5 folds (choosing by F1), record its cross-validation scores, and score it on the test students.

```python
 98      for name, (model, grid_ml) in traditional.items():
 99          search = GridSearchCV(Pipeline([("prep", make_preprocessor(k)), ("model", model)]), grid_ml, cv=FOLDS,
100                                scoring=["accuracy", "precision", "recall", "f1"], refit="f1").fit(train, y_train)
101          i = search.best_index_
102          results[name] = {"cv": {m: float(search.cv_results_[f"mean_test_{m}"][i]) for m in ["accuracy", "precision", "recall", "f1"]},
103                           "test": scores(y_test, search.predict(test)), "setting": str(search.best_params_)}
```

**Line 104.** The simplest baseline: always say PASS (84.6% accuracy, but it finds no failing students).

```python
104      results["Always predict 'pass'"] = {"cv": None, "test": scores(y_test, np.zeros_like(y_test))}
```

**Lines 106–119.** Saves the preprocessor and the network (used by the app), the chosen columns and their information scores, every experiment, all model scores, the confusion matrix and the epochs into `results/results.json`, then draws the graphs.

```python
106      # G. Save the model, the results and the graphs
107      os.makedirs("models", exist_ok=True)
108      joblib.dump(prep, "models/preprocessor.joblib")
109      dnn.save("models/dnn.keras")
110      kept = list(prep.named_steps["select"].get_feature_names_out())
111      info = dict(zip(prep.named_steps["select"].feature_names_in_, prep.named_steps["select"].scores_))
112      report = {"k": k, "settings": settings, "kept_columns": kept, "info_scores": info,
113                "experiments": {"features": features, "grid": grid_scores, "activation": activation, "optimizer": optimizer},
114                "models": results, "confusion_matrix": confusion_matrix(y_test, dnn_pred).tolist(),
115                "epochs_trained": len(history.history["loss"]), "best_epoch": int(np.argmin(history.history["val_loss"])) + 1}
116      with open("results/results.json", "w") as f:
117          json.dump(report, f, indent=2, default=float)
118      draw_graphs(report, history, y_test, dnn_pred)
119      print("test scores:", {n: round(r["test"]["accuracy"], 3) for n, r in results.items()})
```

**Line 122.** Draws the five graphs used in the notebook, app and guide.

```python
122  def draw_graphs(report, history, y_test, dnn_pred):
```

**Lines 123–126.** Graph 1: the confusion matrix of the network on the test students.

```python
123      # 1. Confusion matrix of the DNN on the test students
124      fig, ax = plt.subplots(figsize=(4.2, 4))
125      ConfusionMatrixDisplay.from_predictions(y_test, dnn_pred, display_labels=["Pass", "At risk"], cmap="Greens", colorbar=False, ax=ax)
126      ax.set_title("DNN on the 130 test students"); fig.tight_layout(); fig.savefig(f"{FIG}/confusion.png", dpi=150); plt.close(fig)
```

**Lines 128–134.** Graph 2: training and validation loss per epoch, with the best epoch marked.

```python
128      # 2. Learning curve: loss per epoch
129      fig, ax = plt.subplots(figsize=(6, 3.5))
130      ax.plot(history.history["loss"], color=GREY, label="training loss")
131      ax.plot(history.history["val_loss"], color=GREEN, label="validation loss")
132      ax.axvline(report["best_epoch"] - 1, color="black", ls="--", lw=1, label=f"best epoch ({report['best_epoch']}), weights kept")
133      ax.set(xlabel="epoch", ylabel="binary cross-entropy loss", title="How the DNN learned"); ax.legend()
134      fig.tight_layout(); fig.savefig(f"{FIG}/learning_curve.png", dpi=150); plt.close(fig)
```

**Lines 136–145.** Graph 3: test accuracy and test F1 of every model (the network in green).

```python
136      # 3. DNN vs traditional ML on the test students
137      names = list(report["models"])
138      fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
139      for ax, metric, title in [(axes[0], "accuracy", "Test accuracy"), (axes[1], "f1", "Test F1 (finding at-risk students)")]:
140          values = [report["models"][n]["test"][metric] for n in names]
141          ax.barh(names, values, color=[GREEN if n.startswith("DNN") else GREY for n in names])
142          for y, v in enumerate(values):
143              ax.text(v + 0.01, y, f"{v:.3f}", va="center", fontsize=9)
144          ax.set(xlim=(0, 1.1), title=title)
145      axes[0].invert_yaxis(); fig.tight_layout(); fig.savefig(f"{FIG}/comparison.png", dpi=150); plt.close(fig)
```

**Lines 147–159.** Graph 4: cross-validation F1 for the number of columns, the activation function and the optimizer (the chosen one in green).

```python
147      # 4. Why these choices: number of features, activation, optimizer (cross-validation F1)
148      exp = report["experiments"]
149      fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
150      for ax, key, title, chosen in [(axes[0], "features", "Number of features kept", str(report["k"])),
151                                     (axes[1], "activation", "Activation function", report["settings"]["activation"]),
152                                     (axes[2], "optimizer", "Optimizer", report["settings"]["optimizer"])]:
153          names = list(exp[key]); values = [exp[key][n]["f1"] for n in names]
154          ax.bar(names, values, color=[GREEN if n == chosen else GREY for n in names])
155          for x, v in enumerate(values):
156              ax.text(x, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
157          ax.set(ylim=(0, 1), title=title, ylabel="cross-validation F1")
158      fig.suptitle("Why these choices (green = chosen, best cross-validation F1)"); fig.tight_layout()
159      fig.savefig(f"{FIG}/choices.png", dpi=150); plt.close(fig)
```

**Lines 161–166.** Graph 5: the information score of the top 20 columns (the kept ones in green).

```python
161      # 5. Information score of every column (top 20); green = kept
162      info = sorted(report["info_scores"].items(), key=lambda kv: kv[1])[-20:]
163      fig, ax = plt.subplots(figsize=(6, 5.5))
164      ax.barh([c for c, _ in info], [v for _, v in info], color=[GREEN if c in report["kept_columns"] else GREY for c, _ in info])
165      ax.set(xlabel="information score (mutual information with pass / at risk)", title="Which columns tell us most (green = kept)")
166      fig.tight_layout(); fig.savefig(f"{FIG}/feature_scores.png", dpi=150); plt.close(fig)
```

**Lines 169–170.** Runs `main()` when you type `python -m src.train`.

```python
169  if __name__ == "__main__":
170      main()
```

## `src/eda.py` (73 lines)

*Exploratory data analysis: checks the data and saves 5 graphs.*

**Lines 1–4.** The file's description.

```python
  1  """Exploratory data analysis (EDA): checks the data and saves 5 graphs to results/figures/.
  2  
  3  Run:  python -m src.eda
  4  """
```

**Lines 5–12.** Graph library (Agg = save graphs to files) and our data functions.

```python
  5  import matplotlib
  6  
  7  matplotlib.use("Agg")
  8  import matplotlib.pyplot as plt
  9  
 10  from src.data import CATEGORICAL, NUMERIC, load_data
 11  
 12  FIG = "results/figures"
```

**Lines 15–25.** Prints the checks: size, missing values, duplicates, the share of failing students, the students with G3 = 0, how closely G1 and G2 follow G3, how accurate the simple rule 'pass if G2 ≥ 10' is, and how skewed absences are.

```python
 15  def audit(df):
 16      print("Shape:", df.shape)
 17      print("Missing values:", int(df.isna().sum().sum()))
 18      print("Duplicate rows:", int(df.duplicated().sum()))
 19      print("At-risk (G3 < 10) share:", f"{df.at_risk.mean():.1%}", "-> majority-class accuracy would be", f"{1 - df.at_risk.mean():.1%}")
 20      zero = df[df.G3 == 0]
 21      print(f"G3 == 0: {len(zero)} students; with zero absences: {(zero.absences == 0).sum()} "
 22            f"(vs {(df.absences == 0).mean():.1%} of all students) -> likely dropped out / missed the final exam")
 23      print("Correlation with G3:", df[["G1", "G2"]].corrwith(df.G3).round(3).to_dict())
 24      print("'Pass if G2 >= 10' rule accuracy:", f"{((df.G2 < 10) == df.at_risk).mean():.1%}")
 25      print("Absences skew:", round(df.absences.skew(), 2), "max:", df.absences.max())
```

**Lines 28–66.** Draws 5 graphs: (1) how many students got each final grade, (2) G2 against G3, (3) correlation of each number column with G3, (4) fail rate by category, (5) absences.

```python
 28  def figures(df):
 29      # 1. Target: G3 distribution with pass threshold
 30      fig, ax = plt.subplots(figsize=(7, 4))
 31      ax.bar(*zip(*df.G3.value_counts().sort_index().items()), color=["#eb6834" if g < 10 else "#2a78d6" for g in sorted(df.G3.unique())])
 32      ax.axvline(9.5, color="grey", ls="--")
 33      ax.set(xlabel="Final grade G3 (0-20)", ylabel="Students", title="G3 distribution (orange = at risk, G3 < 10)")
 34      fig.tight_layout(); fig.savefig(f"{FIG}/g3_distribution.png", dpi=150); plt.close(fig)
 35  
 36      # 2. G2 vs G3: the 2nd period grade almost decides the final grade
 37      fig, ax = plt.subplots(figsize=(5, 5))
 38      ax.scatter(df.G2, df.G3, alpha=0.3, s=15, color="#2a78d6")
 39      zero = df[df.G3 == 0]
 40      ax.scatter(zero.G2, zero.G3, s=25, color="#eb6834", label="G3 = 0 (likely dropout)")
 41      ax.set(xlabel="G2 (second period)", ylabel="G3 (final)", title=f"G2 vs G3, r = {df.G2.corr(df.G3):.2f}")
 42      ax.legend(); fig.tight_layout(); fig.savefig(f"{FIG}/g2_vs_g3.png", dpi=150); plt.close(fig)
 43  
 44      # 3. Spearman correlation of every numeric feature with G3 (rank-based, fine for ordinal scales)
 45      corr = df[NUMERIC].corrwith(df.G3, method="spearman").sort_values()
 46      fig, ax = plt.subplots(figsize=(6, 5))
 47      ax.barh(corr.index, corr.values, color=["#eb6834" if v < 0 else "#2a78d6" for v in corr.values])
 48      ax.set(xlabel="Spearman correlation with G3", title="Numeric features vs final grade")
 49      fig.tight_layout(); fig.savefig(f"{FIG}/correlations.png", dpi=150); plt.close(fig)
 50  
 51      # 4. At-risk rate for each categorical feature (largest gaps first)
 52      gaps = {c: df.groupby(c).at_risk.mean() for c in CATEGORICAL}
 53      order = sorted(gaps, key=lambda c: gaps[c].max() - gaps[c].min(), reverse=True)[:8]
 54      fig, axes = plt.subplots(2, 4, figsize=(12, 5), sharey=True)
 55      for ax, c in zip(axes.flat, order):
 56          ax.bar(gaps[c].index.astype(str), gaps[c].values * 100, color="#2a78d6")
 57          ax.set_title(c); ax.tick_params(axis="x", rotation=45)
 58      axes[0, 0].set_ylabel("% at risk"); axes[1, 0].set_ylabel("% at risk")
 59      fig.suptitle("At-risk rate by category (8 largest gaps)"); fig.tight_layout()
 60      fig.savefig(f"{FIG}/risk_by_category.png", dpi=150); plt.close(fig)
 61  
 62      # 5. Absences: heavy right tail, but real students, so kept
 63      fig, ax = plt.subplots(figsize=(6, 3.5))
 64      ax.hist(df.absences, bins=32, color="#2a78d6")
 65      ax.set(xlabel="Absences", ylabel="Students", title=f"Absences (skew {df.absences.skew():.1f})")
 66      fig.tight_layout(); fig.savefig(f"{FIG}/absences.png", dpi=150); plt.close(fig)
```

**Lines 69–73.** Runs the checks and graphs when you type `python -m src.eda`.

```python
 69  if __name__ == "__main__":
 70      data = load_data()
 71      audit(data)
 72      figures(data)
 73      print(f"Figures saved to {FIG}/")
```

## `app.py` (236 lines)

*The Streamlit web app.*

**Lines 1–6.** The file's description: the app only loads the saved model; it never trains.

```python
  1  """Streamlit app: enter a student's details and the neural network predicts PASS or AT RISK.
  2  
  3  Run:  streamlit run app.py
  4  The app does not train anything. It loads the saved preprocessor (models/preprocessor.joblib) and the saved
  5  network (models/dnn.keras) created by `python -m src.train`.
  6  """
```

**Lines 7–19.** Keras runs on PyTorch; then the libraries and our own data, names and reasons.

```python
  7  import os
  8  
  9  os.environ.setdefault("KERAS_BACKEND", "torch")
 10  
 11  import json
 12  
 13  import joblib
 14  import keras
 15  import pandas as pd
 16  import streamlit as st
 17  
 18  from src.data import CATEGORICAL, DESCRIPTIONS, FEATURES, GROUPS, load_data, split
 19  from src.reasons import REASONS
```

**Line 21.** AT RISK if the chance of failing is 50% or more.

```python
 21  THRESHOLD = 0.5  # probability of failing at or above 50% -> AT RISK
```

**Lines 22–33.** Readable labels for coded values (for example GP → Gabriel Pereira, 1 → '< 2 h / week').

```python
 22  LEVEL_1_5 = {1: "1 · very low", 2: "2 · low", 3: "3 · medium", 4: "4 · high", 5: "5 · very high"}
 23  CHOICES = {  # readable labels for the coded values in the dataset
 24      "school": {"GP": "Gabriel Pereira", "MS": "Mousinho da Silveira"}, "sex": {"F": "Female", "M": "Male"},
 25      "address": {"U": "Urban", "R": "Rural"}, "famsize": {"LE3": "3 or fewer", "GT3": "More than 3"},
 26      "Pstatus": {"T": "Living together", "A": "Apart"},
 27      "Medu": {0: "None", 1: "Primary (4th grade)", 2: "5th–9th grade", 3: "Secondary", 4: "Higher education"},
 28      "traveltime": {1: "< 15 min", 2: "15–30 min", 3: "30 min – 1 h", 4: "> 1 h"},
 29      "studytime": {1: "< 2 h / week", 2: "2–5 h / week", 3: "5–10 h / week", 4: "> 10 h / week"},
 30      "famrel": LEVEL_1_5, "freetime": LEVEL_1_5, "goout": LEVEL_1_5, "Dalc": LEVEL_1_5, "Walc": LEVEL_1_5,
 31      "health": {1: "1 · very bad", 2: "2 · bad", 3: "3 · fair", 4: "4 · good", 5: "5 · very good"},
 32  }
 33  CHOICES["Fedu"] = CHOICES["Medu"]
```

**Line 34.** Inputs shown as sliders, with the allowed ranges from the codebook. This is input validation: impossible values can't be entered.

```python
 34  SLIDERS = {"age": (15, 22), "G1": (0, 20), "G2": (0, 20), "failures": (0, 3)}  # allowed ranges from the codebook
```

**Lines 36–46.** Page title, wide layout and a little styling: the green PASS and red AT RISK badges, and the result panel staying in view.

```python
 36  st.set_page_config(page_title="Student Performance Predictor", layout="wide")
 37  st.markdown("""<style>
 38    .block-container {padding-top: 2rem; max-width: 1200px;}
 39    h1 {font-weight: 650; letter-spacing: -0.02em;}
 40    .badge {display: inline-block; padding: 0.35rem 0.8rem; border-radius: 6px; font-weight: 650; font-size: 1.6rem;}
 41    .pass {background: #dcfce7; color: #166534;}
 42    .risk {background: #fee2e2; color: #991b1b;}
 43    .muted {color: #71717a; font-size: 0.9rem;}
 44    @media (min-width: 900px) {div[data-testid="stColumn"]:has(.result-anchor) {position: sticky; top: 3.5rem; align-self: flex-start;}}
 45    @media (max-width: 899px) {div[data-testid="stHorizontalBlock"]:has(.result-anchor) {flex-direction: column-reverse;}}
 46  </style>""", unsafe_allow_html=True)
```

**Lines 49–51.** Loads the saved preprocessor and network once (cached), so they are not reloaded on every click.

```python
 49  @st.cache_resource
 50  def load_models():
 51      return joblib.load("models/preprocessor.joblib"), keras.models.load_model("models/dnn.keras")
```

**Lines 54–58.** Loads the training students (for typical values and fail rates) and `results/results.json` (all scores).

```python
 54  @st.cache_data
 55  def load_reference():
 56      train, _ = split(load_data())  # the same 519 training students the model learned from
 57      with open("results/results.json") as f:
 58          return train, json.load(f)
```

**Lines 61–66.** Runs the loaders. `used` = the 5 inputs the model uses. `typical` = the most common value of each input, used as the defaults.

```python
 61  prep, net = load_models()
 62  train, report = load_reference()
 63  kept = report["kept_columns"]
 64  # Which of the 32 inputs the model uses (a kept column like "school_MS" belongs to the input "school")
 65  used = [c for c in FEATURES if any(k == c or k.startswith(c + "_") for k in kept)]
 66  typical = {c: train[c].mode()[0] if c in CATEGORICAL else int(train[c].median()) for c in FEATURES}
```

**Lines 69–72.** THE PREDICTION: put the inputs in the training order, apply the same preprocessing as in training, and ask the network for the chance of failing.

```python
 69  def risk(students):
 70      """Probability of being at risk, using exactly the same preprocessing as in training."""
 71      X = prep.transform(pd.DataFrame(students)[FEATURES])
 72      return net.predict(X, verbose=0)[:, 0]
```

**Lines 75–85.** Draws one input: its label ('used by model' if used), its (?) tooltip with the reason, and a slider, number box or dropdown.

```python
 75  def field(name):
 76      label = DESCRIPTIONS[name] + (" · used by model" if name in used else "")
 77      why = ("Used by the model. " if name in used else "Not used by the model. ") + REASONS[name]
 78      if name in SLIDERS:
 79          return st.slider(label, *SLIDERS[name], typical[name], key=name, help=why)
 80      if name == "absences":
 81          return st.number_input(label, 0, 93, typical[name], key=name, help=why)
 82      options = sorted(train[name].unique())
 83      fmt = CHOICES.get(name, {}).get
 84      return st.selectbox(label, options, index=options.index(typical[name]), key=name, help=why,
 85                          format_func=lambda v: fmt(v) or str(v).replace("_", " ").capitalize())
```

**Lines 88–93.** The title, the subtitle and the four tabs.

```python
 88  st.title("Student Performance Predictor")
 89  st.markdown('<p class="muted">A deep neural network predicts whether a Portuguese secondary-school student will '
 90              '<b>pass</b> (final grade G3 ≥ 10 out of 20) or is <b>at risk</b> of failing.</p>', unsafe_allow_html=True)
 91  
 92  tab_predict, tab_results, tab_features, tab_about = st.tabs(
 93      ["1 · Predict", "2 · Results & graphs", "3 · Feature choices", "4 · About"])
```

**Lines 95–105.** Tab 1: the form in four boxes (one per group), every answer stored in `student`.

```python
 95  # ---------------------------------------------------------------- 1. Predict
 96  with tab_predict:
 97      form_col, result_col = st.columns([3, 2], gap="large")
 98      student = {}
 99      with form_col:
100          boxes = [*st.columns(2), *st.columns(2)]
101          for box, (group, cols) in zip(boxes, GROUPS.items()):
102              with box.container(border=True):
103                  st.markdown(f"**{group}**")
104                  for c in cols:
105                      student[c] = field(c)
```

**Lines 107–121.** The result panel: calculate the chance of failing and show PASS or AT RISK, the percentage, an 'uncertain' note near 50%, a progress bar, and a warning for unusually high absences.

```python
107      with result_col:
108          st.markdown('<div class="result-anchor"></div>', unsafe_allow_html=True)
109          p = float(risk([student])[0])
110          with st.container(border=True):
111              st.markdown("Prediction")
112              st.markdown(f'<span class="badge {"risk" if p >= THRESHOLD else "pass"}">'
113                          f'{"AT RISK" if p >= THRESHOLD else "PASS"}</span>', unsafe_allow_html=True)
114              st.markdown(f"Chance of failing: **{p:.0%}**  \n<span class='muted'>AT RISK if 50% or more."
115                          f"{' Close to 50%, so treat it as uncertain.' if abs(p - THRESHOLD) < 0.15 else ''}</span>",
116                          unsafe_allow_html=True)
117              st.progress(p)
118              if student["absences"] > train["absences"].max():
119                  st.warning(f"No training student had more than {train['absences'].max()} absences, so this is less reliable.")
120  
121          with st.container(border=True):
```

**Lines 122–134.** The explanation: for each used input that differs from typical, put it back to typical, predict again, and show how much the chance changed (largest effects first).

```python
122              st.markdown("**What pushed this result**")
123              # For each input the model uses: set it back to a typical value and see how much the chance changes.
124              changed = [c for c in used if student[c] != typical[c]]
125              effects = dict(zip(changed, p - risk([student | {c: typical[c]} for c in changed]))) if changed else {}
126              shown = [(c, d) for c, d in sorted(effects.items(), key=lambda kv: -abs(kv[1])) if abs(d) >= 0.01][:4]
127              if not shown:
128                  st.markdown("This student looks like a typical student on everything the model uses.")
129              for c, d in shown:
130                  st.markdown(f"- **{DESCRIPTIONS[c]}** = {student[c]} (typical: {typical[c]}) "
131                              f"{'raises' if d > 0 else 'lowers'} the chance of failing by {abs(d):.0%}")
132              st.markdown("<span class='muted'>Each line sets one input back to a typical value and measures the change.</span>",
133                          unsafe_allow_html=True)
134          st.caption(f"The model uses {len(used)} of the 32 inputs: {', '.join(used)}.")
```

**Lines 136–179.** Tab 2: the test scores (accuracy, precision, recall, F1), a note on why accuracy alone is not enough, the confusion matrix in words, the comparison graph and table, the choices graph, the learning curve, and a table of the final settings.

```python
136  # ---------------------------------------------------------------- 2. Results & graphs
137  with tab_results:
138      dnn = report["models"]["DNN (neural network)"]["test"]
139      (tn, fp), (fn, tp) = report["confusion_matrix"]
140      st.markdown("**Final scores on the 130 test students** (kept aside; the model never saw them while learning or choosing settings)")
141      c = st.columns(4)
142      c[0].metric("Accuracy", f"{dnn['accuracy']:.1%}", help="Share of all students predicted correctly")
143      c[1].metric("Precision", f"{dnn['precision']:.1%}", help="Of the students we flagged AT RISK, how many really failed")
144      c[2].metric("Recall", f"{dnn['recall']:.1%}", help="Of the students who really failed, how many we flagged")
145      c[3].metric("F1 score", f"{dnn['f1']:.3f}", help="Balance of precision and recall")
146      st.info("Only 15% of students fail, so a model that always says 'pass' already gets 84.6% accuracy. "
147              "That is why we also look at recall and F1: they check that we actually find the failing students.")
148  
149      left, right = st.columns([2, 3])
150      left.image("results/figures/confusion.png", width=340)
151      right.markdown(f"""**Confusion matrix, in words**
152  - **{tn}** passing students correctly predicted PASS
153  - **{tp}** failing students correctly flagged AT RISK
154  - **{fp}** false alarms (flagged AT RISK but passed)
155  - **{fn}** missed (predicted PASS but failed)""")
156  
157      st.markdown("**DNN vs traditional machine learning** (same students, same inputs, same 5-fold cross-validation)")
158      st.image("results/figures/comparison.png")
159      rows = {name: {"CV accuracy": r["cv"]["accuracy"] if r["cv"] else None, "CV F1": r["cv"]["f1"] if r["cv"] else None,
160                     "Test accuracy": r["test"]["accuracy"], "Test precision": r["test"]["precision"],
161                     "Test recall": r["test"]["recall"], "Test F1": r["test"]["f1"]} for name, r in report["models"].items()}
162      st.dataframe(pd.DataFrame(rows).T.style.format("{:.3f}", na_rep="–"), width="stretch")
163  
164      st.markdown("**Why these choices** (each tested with 5-fold cross-validation on the training students; green = chosen)")
165      st.image("results/figures/choices.png")
166      st.markdown("**How the network learned**")
167      st.image("results/figures/learning_curve.png", width=620)
168      s = report["settings"]
169      st.markdown(f"""**The final network**
170  | Part | Choice | Why |
171  |---|---|---|
172  | Inputs | {len(kept)} columns | Feature selection (tab 3) |
173  | Hidden layers | {s['hidden'][0]} then {s['hidden'][1]} neurons | Best of the grid search |
174  | Activation | {s['activation']} | Best in the activation test (graph above) |
175  | Dropout | {s['dropout']} | Best of the grid search; stops the network memorising |
176  | Output | 1 neuron, sigmoid | Gives a probability between 0 and 1 |
177  | Loss | binary cross-entropy | Standard loss for yes/no predictions |
178  | Optimizer | {s['optimizer']}, learning rate {s['learning_rate']} | Best in the optimizer test and grid search |
179  | Early stopping | stop after 20 epochs without improvement | Trained {report['epochs_trained']} epochs, kept epoch {report['best_epoch']} |""")
```

**Lines 181–194.** Tab 3: how the inputs were chosen, the 5 / 10 / 20 / all results, and the information-score graph.

```python
181  # ---------------------------------------------------------------- 3. Feature choices
182  with tab_features:
183      exp = report["experiments"]["features"]
184      st.markdown(f"""**How we chose the inputs**
185  1. All **32** inputs were considered. Text columns become 0/1 columns, so there are 45 columns in total.
186  2. Each column gets an **information score**: how much it tells us about pass / at risk (mutual information), calculated on the training students only.
187  3. We kept the **top {len(kept)}**. We tested keeping 5, 10, 20 or all 45 columns with cross-validation, and {report['k']} gave the best F1:
188  """)
189      st.table(pd.DataFrame({k: {"CV accuracy": f"{v['accuracy']:.3f}", "CV F1": f"{v['f1']:.3f}"} for k, v in exp.items()}).T
190               .rename_axis("Columns kept"))
191      st.image("results/figures/feature_scores.png", width=560)
192  
193      info = report["info_scores"]
194      ranked = sorted(info, key=info.get, reverse=True)
```

**Lines 196–199.** Finds an input's best information score and its rank out of 45.

```python
196      def evidence(f):
197          cols = [c for c in info if c == f or c.startswith(f + "_")]
198          best = max(cols, key=info.get)
199          return info[best], ranked.index(best) + 1
```

**Lines 201–212.** The lookup box: shows whether the input is used, its reason, score and rank, and the fail rate for each of its values.

```python
201      pick = st.selectbox("Ask about any input", FEATURES, index=FEATURES.index("address"),
202                          format_func=lambda f: f"{DESCRIPTIONS[f]} ({f})")
203      score, rank = evidence(pick)
204      with st.container(border=True):
205          st.markdown(f"**{DESCRIPTIONS[pick]}** is **{'USED' if pick in used else 'NOT USED'}** by the model.  \n{REASONS[pick]}")
206          a, b = st.columns(2)
207          a.metric("Information score", f"{score:.3f}")
208          b.metric("Rank", f"{rank} of 45", help=f"The top {len(kept)} are kept")
209          if train[pick].nunique() <= 6:
210              rates = train.groupby(pick).at_risk.agg(["mean", "size"])
211              st.table(pd.DataFrame({"At-risk rate": rates["mean"].map("{:.0%}".format), "Students": rates["size"]})
212                       .rename(index=lambda v: CHOICES.get(pick, {}).get(v, v)))
```

**Lines 214–217.** A table of all 32 inputs: used or not, rank, and reason.

```python
214      st.markdown("**All 32 inputs**")
215      table = pd.DataFrame([{"Input": DESCRIPTIONS[f], "Used": "yes" if f in used else "no", "Rank": evidence(f)[1],
216                             "Why": REASONS[f]} for f in FEATURES]).sort_values("Rank").set_index("Input")
217      st.table(table)
```

**Lines 219–236.** Tab 4: project, data source, method and a responsible-use note.

```python
219  # ---------------------------------------------------------------- 4. About
220  with tab_about:
221      st.markdown("""
222  **Project.** Student Academic Performance Prediction using Deep Neural Networks · 24CAI0306 Deep Neural Networks ·
223  B.E. CSE-AIML, Semester 5 · Kavin Thakur.
224  
225  **Data.** UCI Student Performance dataset (Portuguese course): 649 students from two schools in Portugal, 2005–06
226  (P. Cortez and A. Silva, 2008, CC BY 4.0). AT RISK means a final grade G3 below 10 out of 20.
227  
228  **Method.** 80% of students for training, 20% kept aside for testing. Text columns one-hot encoded, numbers standardised,
229  top columns chosen by information score. A Keras neural network (Dense + Dropout layers, sigmoid output) chosen by
230  grid search with 5-fold cross-validation, and compared with logistic regression, a decision tree and a random forest.
231  
232  **Use responsibly.** Trained on two schools in one year, and some inputs are sensitive (sex, family). Use it to start a
233  conversation with a student, never to decide something about them.
234  
235  Code: github.com/auraCodesKM/student-performance-dnn
236  """)
```
