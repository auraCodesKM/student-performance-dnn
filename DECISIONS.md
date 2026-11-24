# Design decisions and viva guide

Every important choice in this project, the evidence behind it, and the questions an examiner is likely to ask. Numbers come from `results/RESULTS.md` and `results/experiments.csv` unless marked as research-phase checks (run on the full dataset before the project code existed).

## 1. The pipeline end to end

```
student-por.csv (649 rows)
  └─ load_data(): add target at_risk = (G3 < 10)
      └─ split(): stratified 80/20  →  train 519  |  test 130 (locked, used once)
          │
          ├─ tune.py: Optuna, 40 trials × 5-fold CV on train → best_params.json
          │
          └─ evaluate.py, per scenario (late / early):
               Pipeline = OneHotEncoder + StandardScaler → SelectKBest(mutual info, k) → model
               • baselines get a grid search with the same CV + metric
               • every model: 5-fold × 3-repeat CV on train  → CV table
               • refit on all 519 → score the 130 test rows once → test table, plots
               • DNN pipeline saved to models/dnn_<scenario>.joblib → app.py
```

The scaler, encoder and feature selector sit **inside** the Pipeline. During cross-validation scikit-learn refits them on each training fold only, so no statistic from a validation or test student leaks into training.

## 2. Data decisions

| Decision | Evidence | Why |
|---|---|---|
| **Portuguese, not Maths** | 649 vs 395 rows. On Maths, no model beat the rule "pass if G2 ≥ 10" (91.9%); the original paper found the same. On Portuguese, models add about 3 points over that rule. | A neural network needs as much data as possible, and a project whose best model loses to a one-line `if` is hard to defend. |
| **Binary target: at risk = G3 < 10** | Brief asks for a performance *category*. 10/20 is the Portuguese pass mark. Five-level A–F tops out around 73–76% accuracy even with grades (research check). | The one formulation that is both useful and accurate enough to act on. |
| **"At risk" is the positive class (1)** | 15.4% of students. | Precision, recall and F1 then describe how well the model finds the students who need help. |
| **Two scenarios** | Late: G1, G2 known (after the 2nd grading period). Early: no grades (start of year). | G1/G2 correlate 0.83/0.92 with G3. Using them is allowed, because they are known before G3, but it changes the question. The two are reported separately so neither flatters the other. |
| **Keep the 15 students with G3 = 0** | All 15 have 0 absences, while only 37.6% of all students do. The chance of that by luck is about 4 × 10⁻⁷, so they most likely missed the final exam or dropped out. | They are still real fails. Removing them from the test set would hide the hardest cases and inflate accuracy. |
| **No outlier removal** | No values outside the codebook ranges. Absences are skewed (max 32) but plausible. | Deleting real students on a guess biases the model against exactly the unusual students it should flag. |
| **No SMOTE; class weights instead** | 85/15 imbalance. `pos_weight = #pass / #fail ≈ 5.5` in the loss is an Optuna option. | Weights change the loss, not the data, so no synthetic students and no new dependency. SMOTE before splitting would leak test students into training. |
| **One-hot for nominal, numbers for ordinal** | Mjob, Fjob, reason, guardian have no order. Medu, studytime, Dalc etc. are ordered 0–4 / 1–5 scales. | One-hot avoids inventing an order. Keeping ordinals numeric keeps the input small (45 columns after encoding with grades, 43 without). |
| **StandardScaler on numeric inputs** | Features range from 0–1 to 0–32. | Gradient descent converges faster when inputs share a scale, and BatchNorm assumes roughly centred inputs. |
| **Feature selection: SelectKBest(mutual information), k tuned** | Research check: cutting to 10 features lifted the MLP from 89.8% to 92.9% accuracy (macro-F1 0.79 → 0.86). Optuna then chose k = 10 (late) and k = 20 (early). | With 519 rows, 43–45 inputs is many noise dimensions for a network. Mutual information also catches non-linear links. It runs inside the Pipeline, so it is refit per fold. |

## 3. Evaluation decisions

| Decision | Why |
|---|---|
| **Stratified 80/20 split, test locked** | Stratify keeps 15.4% at-risk in both parts. The test set is touched only by `evaluate.py`, after tuning is finished. |
| **5-fold × 3-repeat CV on the training set** | The test set is small: one student = 0.8 accuracy points. 15 folds give a mean and a standard deviation, which is a much steadier comparison. |
| **Macro-F1 as the main metric** | "Everyone passes" scores 84.6% accuracy and 0.458 macro-F1. Macro-F1 averages F1 of both classes, so ignoring the at-risk class is punished. |
| **Two floors to beat** | Majority class (84.6%) for both scenarios; the naive rule "at risk if G2 < 10" for late-term. |
| **Baselines tuned too** | Logistic regression, random forest and XGBoost each get a grid search with the same CV and metric. Comparing a tuned DNN against untuned baselines would be unfair. |
| **5-seed test score + bootstrap CI for the DNN** | A network's result depends on its random start. The bootstrap (resampling the 130 test students 1000 times) shows how much the test score itself could move. |

## 4. Model decisions

The final late-term network: **10 inputs → 64 → 32 → 16 → 1**, each hidden layer `Linear → BatchNorm → ReLU → Dropout(0.16)`.
The early-warning network: **20 inputs → 16 → 8 → 8 → 1**, no BatchNorm, dropout 0.13, class weights on.

| Choice | Why |
|---|---|
| **MLP** | The data is a table of independent students, with no images or sequences, so CNNs and RNNs have nothing to exploit. A feedforward network is the right DNN. |
| **ReLU** | Does not saturate for positive inputs, so gradients don't vanish; cheap to compute. |
| **He (Kaiming) initialisation** | Designed for ReLU: scales the initial weights by √(2 / fan_in) so activations keep roughly the same variance from layer to layer. |
| **One output logit + `BCEWithLogitsLoss`** | Binary task, so one sigmoid output is enough. The "with logits" version fuses sigmoid and cross-entropy for numerical stability. |
| **BatchNorm** | Normalises each layer's inputs per mini-batch, which stabilises and speeds up training. Optuna kept it for late-term and dropped it for early-warning. |
| **Dropout** | Randomly zeroes units during training so the network can't depend on any one of them. A regulariser for small data. |
| **AdamW + weight decay** | Adam adapts the step size per weight. AdamW applies weight decay (L2-style shrinkage) correctly, separate from the adaptive step. |
| **Mini-batch training (32 or 64)** | A middle ground between noisy single-sample SGD and slow full-batch gradient descent. |
| **Early stopping, patience 30, best weights restored** | Stops when validation loss hasn't improved for 30 epochs and rolls back to the best epoch. See `results/figures/learning_curve_late.png`: validation loss bottoms out around epoch 73 while training loss keeps falling, which is the start of overfitting. The validation slice is 15% of the *training* data, never the test set. |
| **Seed ensemble of 5 networks** | In v1, one network per model gave a CV macro-F1 std of 0.072, against about 0.045 for the other models. The saved early model scored 0.731 test accuracy against a 5-seed mean of 0.797, purely from its random start. Averaging 5 networks' probabilities cut the seed-to-seed test-accuracy spread from 0.010 to 0.004 (late-term). Tuning uses 1 network per fold for speed; the final model uses 5. |
| **Optuna (TPE) over 9 hyperparameters** | Depth 1–3, width, dropout, BatchNorm, learning rate, weight decay, batch size, class weights, k. TPE learns which regions score well and samples there, which beats grid search for this many dimensions. All 80 trials are logged in `results/experiments.csv`. |
| **PyTorch, not Keras** | Both are in the syllabus. The explicit loop (`zero_grad → forward → loss → backward → step`) shows each step of backpropagation. |
| **Wrapped as a scikit-learn estimator** | `DNNClassifier` has `fit` / `predict_proba`, so it drops into the same Pipeline, CV and metric code as the baselines. That makes the comparison fair and leakage-safe by construction. |
| **CPU, not GPU** | Notebook section 6, 50 epochs of the same network: M3 CPU 0.36 s vs M3 GPU (MPS) 1.03 s; Kaggle CPU 1.86 s vs Kaggle Tesla T4 1.92 s. With batches of 32 rows × at most 45 columns, GPU launch overhead outweighs the maths, so the laptop CPU is the fastest option. |

## 5. What the results mean

- **Late-term:** the DNN reaches 0.900 test accuracy (0.912 ± 0.004 over 5 seeds) and ROC-AUC 0.953, level with XGBoost and the naive rule. Its CV ROC-AUC (0.967) matches the other models (0.963–0.974), so it ranks students just as well. Its weaker point is at-risk recall (0.68 CV): Optuna chose *no* class weights: trials without them averaged 0.826 CV macro-F1, against 0.800 with them (32 vs 8 trials). With G2 available, a 5.5× weight makes the network over-flag students.
- **Early-warning:** no model beats "everyone passes" on accuracy. The DNN has the **best CV balanced accuracy (0.744) and at-risk recall (0.671)**, because class weights make it flag more students at some cost in accuracy. Random forest has the best CV macro-F1 (0.721).
- **Why the DNN does not win outright:** tree ensembles still beat neural networks on small tabular datasets (Grinsztajn, Oyallon, Varoquaux, NeurIPS 2022). Matching them with a careful, leakage-free setup is the realistic result.

## 6. Honest notes

- The test set was scored in v1 (single network) before the seed ensemble was added. The change was motivated by the CV standard deviation, not by the test score, and the code was not re-tuned afterwards. v1 DNN test scores: late 0.908 accuracy / 0.830 macro-F1, early 0.731 / 0.607.
- The research-phase 10-feature check picked features using all 649 rows, so its 92.9% was slightly optimistic. The project selects features inside each fold instead.
- The notebook was also run on Kaggle, which has different library versions (Python 3.13, torch 2.11, scikit-learn 1.6.1). The DNN's 5-seed test means came out identical (late 0.912 ± 0.004, early 0.803 ± 0.020). Single-model test accuracies moved by at most 3 students: late-term DNN 0.908 (local 0.900), XGBoost 0.877 (local 0.900); early-warning random forest 0.823 (local 0.838). Seeds fix the randomness, but library versions change the floating-point arithmetic slightly. `requirements.txt` pins the versions behind the numbers in this repo.
- On macOS, importing torch and xgboost in one process crashed (three bundled copies of OpenMP). Setting `OMP_NUM_THREADS=1` at the top of each script fixes it, and single-threaded is fastest on this data anyway.

## 7. Likely viva questions

**Isn't using G1 and G2 data leakage?**
Leakage means using information that wouldn't exist at prediction time. G1 and G2 are recorded before G3, so a late-term model may use them. Because they nearly determine G3, I also report the naive "G2 < 10" rule and a separate model without grades.

**Why is accuracy not your main metric?**
84.6% of students pass, so predicting "pass" for everyone already scores 84.6% while finding nobody at risk. Macro-F1 and at-risk recall expose that.

**How do you prevent leakage in preprocessing?**
The encoder, scaler and feature selector are steps of a scikit-learn Pipeline. `cross_validate` and `fit` refit them on training rows only. The test set is split off first and used once. Early stopping uses a slice of the training data.

**Why does your DNN not beat XGBoost?**
649 rows of mixed tabular data is the setting where tree ensembles are strongest. The DNN matches them on accuracy and ROC-AUC, and the 5-seed spread (± 0.004) shows the result is stable.

**What does BatchNorm do at test time?**
In `model.eval()` mode it uses running averages of the mean and variance collected during training, not the statistics of the current batch. That is why `predict_proba` works on a single student.

**Why `model.train()` and `model.eval()`?**
They switch dropout and BatchNorm between training behaviour (random dropping, batch statistics) and inference behaviour (no dropping, running statistics).

**What is `pos_weight`?**
A multiplier on the loss of positive (at-risk) examples. 5.5 ≈ #pass / #fail makes the 15% minority count as much as the majority in total.

**How was the architecture chosen?**
Optuna searched depth, width, dropout, BatchNorm, learning rate, weight decay, batch size, class weights and number of features, scoring each trial by 5-fold CV macro-F1 on the training set. The best late-term configuration (3 layers, 64-32-16, BatchNorm, 10 features) appeared in all of the top 8 trials.

**What does the learning curve show?**
Training and validation loss fall together until about epoch 73. After that, training loss keeps falling while validation loss rises slightly, which means overfitting has started. Early stopping keeps the epoch-73 weights.

**How confident are you in the 90% test accuracy?**
The bootstrap 95% interval is 84.6–94.6%, because 130 test students is a small sample. The cross-validation estimate (91.9% over 15 folds) agrees.

**Why PyTorch for the main model and not Keras?**
Both are in the syllabus, and notebook section 9 builds the same network in Keras: same inputs, layers and hyperparameters, with `EarlyStopping` and `ModelCheckpoint` callbacks. One Keras network scores 0.885 test accuracy, against 0.908 for one PyTorch network. The gap is within the noise of a single random start. PyTorch is the main model because its explicit loop shows every step of backpropagation.

**Why no `DataLoader`?**
All 519 rows fit in memory as one tensor. Shuffling with `torch.randperm` and slicing batches is what `DataLoader(shuffle=True)` does internally, without the extra worker and collate machinery.

**Can you show what the network actually computes?**
Notebook section 10 recomputes the saved network's predictions with plain NumPy: `h @ W.T + b`, BatchNorm with the running mean and variance, `max(0, h)`, then the sigmoid. The largest difference from PyTorch is about 1e-7.

**What would improve the early-warning model?**
More students and more signal: attendance over time, homework records, earlier-year grades. The 30 survey features carry little information about G3, and `failures` alone dominates.
