# Student Academic Performance Prediction using Deep Neural Networks

A deep neural network that predicts whether a secondary-school student will **pass** the year or is **at risk** of failing. It uses demographic, academic, social and behavioural data, and a Streamlit app serves the predictions.

Course project for 24CAI0306 Deep Neural Networks (B.E. CSE-AIML, Semester 5).

- **Notebook (runs everything):** [`student_performance_dnn.ipynb`](student_performance_dnn.ipynb), also on Kaggle: [kaggle.com/code/kavinthakurkm/student-performance-dnn](https://www.kaggle.com/code/kavinthakurkm/student-performance-dnn)
- **Every design decision, its alternatives and viva Q&A:** [`DECISIONS.md`](DECISIONS.md)
- **Line-by-line code walkthrough, notebook and Kaggle explained, CHO mapping, DNN vs ML comparison:** [`WALKTHROUGH.md`](WALKTHROUGH.md)
- **What to show in the Streamlit demo, step by step:** [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md)
- **Full result tables:** [`results/RESULTS.md`](results/RESULTS.md)

## Problem statement

Schools want to find students who are likely to fail early enough to help them. Given a student's background, habits and first two period grades, the system predicts the final grade category:

| Final grade G3 (0–20) | Category |
|---|---|
| G3 ≥ 10 | **Pass** |
| G3 < 10 | **At risk** |

## Dataset

[UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance) (DOI [10.24432/C5TG7T](https://doi.org/10.24432/C5TG7T)), Portuguese language course (P. Cortez and A. Silva, 2008, CC BY 4.0). Kaggle mirror: [`uciml/student-alcohol-consumption`](https://www.kaggle.com/datasets/uciml/student-alcohol-consumption).

- **Size:** 649 students from two secondary schools in Portugal, 2005–06; 33 columns, no missing values.
- **Target:** G3, the final grade. 100 students (15.4%) are at risk.
- **Inputs:** the 32 other columns. The full data dictionary is in notebook section 2.

| Group | Columns |
|---|---|
| Demographic | sex, age, address, famsize, Pstatus, Medu, Fedu, Mjob, Fjob, guardian |
| Academic | school, reason, G1, G2, failures, studytime, traveltime, schoolsup, famsup, paid, nursery, higher |
| Social | famrel, freetime, goout, romantic, activities, internet |
| Behavioural | absences, Dalc, Walc, health |

G1 and G2 are the 1st and 2nd period grades. They are known before G3, so using them is not leakage. Without them no model beats the majority class (see the ablation in `DECISIONS.md`, D3).

## Method

```
649 students → stratified 80/20 split (130 test students locked)
  → one scikit-learn Pipeline, fitted on training rows only:
      one-hot encoding (17 categorical) + standardisation (15 numeric)    45 columns
      → SelectKBest, discrete mutual information                        10 columns
      → DNN: 10 → 16 → 8 → 1, BatchNorm + ReLU + Dropout(0.48),
             He init, BCE loss, AdamW, early stopping, average of 5 networks
  → Optuna (40 trials, 5-fold × 2 CV macro-F1) for the DNN's hyperparameters
  → same CV (5-fold × 3) for the DNN and 5 baselines, then the locked test set once
```

**Final 10 inputs:** G1, G2, failures, absences, studytime, higher, school, Fedu, Medu, famrel. The network has 369 trainable weights; the ensemble of 5 has 1,845.

## Results

**Locked test set** (130 students, scored once):

| Model | Accuracy | Precision (at risk) | Recall (at risk) | Macro-F1 | ROC-AUC |
|---|---|---|---|---|---|
| **DNN (final)** | **0.915** | 0.714 | 0.750 | **0.841** | **0.953** |
| XGBoost | 0.900 | 0.640 | 0.800 | 0.825 | 0.925 |
| Logistic regression | 0.900 | 0.652 | 0.750 | 0.819 | 0.944 |
| Rule "G2 < 10" | 0.900 | 0.621 | 0.900 | 0.837 | 0.953 |
| Random forest | 0.862 | 0.531 | 0.850 | 0.784 | 0.946 |
| Majority class | 0.846 | – | 0.000 | 0.458 | 0.500 |

- **DNN on the test set:**
  - Confusion matrix: 104 passes correct, 6 false alarms, 5 at-risk students missed, 15 caught.
  - 95% confidence interval for accuracy: 85.5–95.2% (Wilson).
  - 5 retrained copies: **92.3 ± 0.8%** accuracy.
- **Cross-validation** (5-fold × 3 repeats on the 519 training students), accuracy: DNN 92.5 ± 2.7%, logistic regression 93.4%, XGBoost 92.9%, random forest 92.1%, rule 89.6%.

**Conclusion.**
- **Ceiling.** With the period grades available, every reasonable model reaches about 90–93%, and the DNN is among the best.
- **Significance.** McNemar's tests show no significant difference between the DNN and the strong baselines (p > 0.7), so the honest claim is "as good as", not "better than".
- **Weak point.** The DNN's at-risk recall (75% on test, 67% in CV); notebook section 10 shows the threshold trade-off.

## Tech stack

| Purpose | Tools |
|---|---|
| Deep learning | PyTorch 2.15 (main model); Keras 3.15 (Sequential and Functional twin, on the PyTorch backend); NumPy (network written from scratch) |
| Classical ML, pipeline, metrics | scikit-learn 1.9, XGBoost 3.4 |
| Hyperparameter tuning | Optuna 5.1 (TPE sampler) |
| Data and plots | pandas 3.1, matplotlib 3.11 |
| App | Streamlit 1.67 |
| Notebook | Jupyter, Kaggle |

Python 3.12. All versions are pinned in `requirements.txt`. Everything runs on a laptop CPU. A GPU is slower for a network this small: 50 epochs take 0.25 s on the M3 CPU vs 0.74 s on its GPU, and 1.66 s on Kaggle's CPU vs 1.81 s on its Tesla T4 (notebook section 16).

## Repository structure

```
student_performance_dnn.ipynb   the whole project, step by step, with outputs
app.py                          Streamlit app
.streamlit/config.toml          app theme
src/data.py                     load data, target, locked split, preprocessing + feature selection
src/model.py                    MLP (nn.Module) and DNNClassifier: training loop, early stopping, seed ensemble
src/tune.py                     Optuna search (training data only)
src/evaluate.py                 baselines, repeated CV, locked test scoring, CIs, McNemar, plots
src/eda.py                      data audit, EDA figures, feature audit, VIF
src/numpy_mlp.py                network with hand-written backprop + gradient check (NumPy only)
src/experiments.py              course-practical experiments: optimisers, schedules, GD variants, init, activations
data/                           UCI CSVs + codebook (student.txt)
models/dnn.joblib               the fitted pipeline used by the app
results/RESULTS.md, metrics.json    final results
results/experiments.csv         every Optuna trial
results/figures/                EDA, confusion matrix, ROC, learning curve
results/supplementary/          earlier versions' logs and metrics, early-warning ablation
DECISIONS.md                    decisions, alternatives, CHO compliance matrix, viva questions
```

## Installation and execution

Requires Python 3.12. PyTorch has no wheels for 3.14 yet.

```bash
git clone https://github.com/auraCodesKM/student-performance-dnn.git
cd student-performance-dnn
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Streamlit app** (uses the saved model; nothing is retrained):
```bash
streamlit run app.py
```

**Notebook** (about 2 minutes on a laptop):
```bash
jupyter lab student_performance_dnn.ipynb
```

**Scripts:**
```bash
python -m src.tune        # Optuna search, about 10 min; rewrites results/best_params.json and experiments.csv
python -m src.evaluate    # baselines + DNN, CV and locked test set, about 1 min; rewrites results/ and models/dnn.joblib
python -m src.eda         # data audit + EDA figures
python -m src.numpy_mlp   # gradient-check self-test of the NumPy network
```

**Reproducibility.** All randomness is seeded. A fresh clone with a new virtual environment reproduced `best_params.json` and every number in `metrics.json` exactly. On Kaggle (Python 3.13, different library versions) the DNN's results were identical; XGBoost's test accuracy moved by 3 students (0.877).

**On Kaggle:** open the notebook link above and choose *Copy & Edit*. The data comes from the attached Kaggle dataset [`uciml/student-alcohol-consumption`](https://www.kaggle.com/datasets/uciml/student-alcohol-consumption), Kaggle's mirror of the UCI files; the notebook checks it is identical to `data/student-por.csv`. The first cell clones this repository for the code, so internet access must be on. The notebook runs on CPU or GPU.

## Limitations

- **Small data.** 649 students from two schools in one year. Results will not transfer to other schools or years without new data.
- **Small test set.** 130 students, only 20 of them at risk: one student moves accuracy by 0.8 points.
- **Test set seen more than once.** It was scored once per model version (3 times in total); every change was decided on cross-validation. The history is in `DECISIONS.md`, section 7.
- **Responsible use.** The inputs include sensitive attributes (sex, family situation, address). The model should prompt a conversation with a student, never decide anything about them.
