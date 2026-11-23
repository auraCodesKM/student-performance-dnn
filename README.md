# Student Performance Prediction with a Deep Neural Network

Predicts whether a Portuguese secondary-school student will finish the year **at risk** (final grade G3 < 10/20) from demographic, family, school and lifestyle data, using a PyTorch multilayer perceptron. The DNN is compared with a majority-class baseline, a naive rule, logistic regression, random forest and XGBoost. A Streamlit app serves the predictions.

Course project for 24CAI0306 Deep Neural Networks (B.E. CSE-AIML, Semester 5).

## Results (130 held-out test students, scored once)

| Scenario | Model | Test accuracy | Test macro-F1 | At-risk recall | ROC-AUC |
|---|---|---|---|---|---|
| **Late term** (uses G1, G2) | **DNN** | **0.900** (5-seed mean 0.912 ± 0.004) | 0.819 | 0.75 | 0.953 |
| | Naive rule "G2 < 10" | 0.900 | 0.837 | 0.90 | 0.953 |
| | XGBoost | 0.900 | 0.825 | 0.80 | 0.925 |
| | Majority class | 0.846 | 0.458 | 0.00 | 0.500 |
| **Early warning** (no grades) | **DNN** | **0.762** (5-seed mean 0.803 ± 0.020) | 0.622 | 0.50 | 0.797 |
| | Random forest | 0.838 | 0.653 | 0.35 | 0.790 |
| | Majority class | 0.846 | 0.458 | 0.00 | 0.500 |

Full tables, cross-validation scores and plots: [results/RESULTS.md](results/RESULTS.md).

**Summary.** With the first two period grades, every reasonable model reaches about 90% accuracy, and the DNN matches the best of them. Without grades the problem is much harder: no model beats "everyone passes" on accuracy. The DNN still has the highest balanced accuracy (0.744) and at-risk recall (0.671) in cross-validation, which is what matters for an early-warning tool. Why each choice was made is in [DECISIONS.md](DECISIONS.md).

## Run it

Python 3.12 (PyTorch does not support 3.14 yet).

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

streamlit run app.py          # app, uses the trained models in models/

python -m src.eda             # data audit + figures      (~5 s)
python -m src.tune            # Optuna search, 40 trials per scenario (~10 min, CPU)
python -m src.evaluate        # baselines vs DNN, test set, saves models (~1 min)
```

All randomness is seeded (`SEED = 42`), so re-running gives the same numbers.

## Project layout

```
data/              UCI Student Performance CSVs + codebook (student.txt)
src/data.py        load data, define target, locked 80/20 split, preprocessing pipeline
src/model.py       MLP (nn.Module) + DNNClassifier: training loop, early stopping, seed ensemble
src/tune.py        Optuna hyperparameter search with 5-fold CV on the training set
src/evaluate.py    baselines, repeated CV, test-set scoring, plots, RESULTS.md
src/eda.py         audit printout + EDA figures
app.py             Streamlit app
models/            trained pipelines (preprocessing + DNN) used by the app
results/           RESULTS.md, metrics.json, experiments.csv (every tuning trial), figures/
```

## Syllabus topics used

| Topic (course plan) | Where |
|---|---|
| Train/validation/test split, data leakage | `src/data.py` `split`, Pipeline refit per fold; `DNNClassifier._train_one` validation split |
| Scaling, encoding (Pandas preprocessing) | `make_preprocessor`: `OneHotEncoder`, `StandardScaler` |
| Feedforward network, ReLU, sigmoid, logits | `MLP` in `src/model.py` |
| Binary cross-entropy | `nn.BCEWithLogitsLoss` with `pos_weight` for class imbalance |
| Backpropagation, mini-batch gradient descent | `loss.backward()`, `optimizer.step()` over shuffled batches |
| He initialisation | `nn.init.kaiming_normal_` |
| Batch normalisation, dropout, L2 / weight decay, early stopping | `MLP`, `AdamW(weight_decay=...)`, patience loop |
| Optimisers (Adam / AdamW) | `torch.optim.AdamW` |
| Hyperparameter tuning (Optuna) | `src/tune.py` |
| Accuracy, precision, recall, F1, ROC-AUC, confusion matrix | `src/evaluate.py` |
| Learning curves, experiment log, reproducibility | `results/figures/learning_curve_*.png`, `results/experiments.csv`, seeds |
| PyTorch `nn.Module`, training loop, saving/loading | `src/model.py`, `joblib.dump` / `joblib.load` |

## Data and limitations

P. Cortez and A. Silva, "Using Data Mining to Predict Secondary School Student Performance", FUBUTEC 2008. [UCI dataset 320](https://archive.ics.uci.edu/dataset/320/student+performance), CC BY 4.0.

- 649 students from two schools in Portugal, 2005–06. Results will not transfer to other schools, countries or years without new data.
- The inputs include sensitive attributes (sex, family situation, address). A model like this should support a conversation with a student, never decide anything about them on its own.
- The test set is small (130 students, 20 at risk). One student changes accuracy by 0.8 points, so differences of a few points between models are not meaningful. The bootstrap 95% interval for the late-term DNN's test accuracy is 0.846–0.946.
