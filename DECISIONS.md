# Design decisions and viva guide

This is the main preparation document for the viva. For every important engineering decision it gives:
- **What** was used and **why**;
- **how** it works, mathematically and in code;
- the **alternatives** that were considered and why they were rejected;
- **where** it is implemented;
- what an examiner might **ask**.

It ends with a compliance matrix against the course handout (CHO), the results history, the limitations and a list of rapid-fire viva questions.

**Evidence labels used throughout.**

| Label | Meaning |
|---|---|
| **[Test]** | The 130 locked test students, scored once per model version |
| **[CV]** | 5-fold × 3-repeat cross-validation on the 519 training students (`random_state=7`), mean over 15 folds |
| **[Tuning-CV]** | 5-fold × 2-repeat CV used inside Optuna (`random_state=42`) |
| **[Val]** | A single 85/15 train/validation split of the training students, used for early stopping and the practical experiments |
| **[Audit]** | Statistics computed on the training students only |

All numbers come from `results/metrics.json`, `results/RESULTS.md`, `results/experiments.csv` or the executed notebook `student_performance_dnn.ipynb`.

---

## 1. The project at a glance

| | |
|---|---|
| **Problem** | Predict whether a secondary-school student will **pass** (final grade G3 ≥ 10/20) or is **at risk** (G3 < 10) |
| **Data** | UCI Student Performance, Portuguese course: 649 students, 33 columns (30 attributes, G1, G2, G3) |
| **Inputs** | All 32 non-target columns are candidates. Automatic feature selection keeps 10 encoded columns. |
| **Model** | PyTorch MLP, 10 → 16 → 8 → 1, with BatchNorm, ReLU, Dropout 0.48 and He initialisation. Trained with BCE loss, AdamW and early stopping. The final model averages 5 networks. |
| **Tuning** | Optuna TPE, 40 trials, objective = macro-F1 [Tuning-CV] |
| **Baselines** | Majority class, the rule "G2 < 10", and grid-searched logistic regression, random forest and XGBoost |
| **App** | Streamlit, one form, live Pass / At-risk prediction with an explanation |

**Headline results, each labelled by where it comes from:**

| Measure | Value |
|---|---|
| **[Test]** DNN accuracy, saved model used by the app | **0.915** (119/130). Wilson 95% CI 0.855–0.952; bootstrap 0.862–0.954 |
| **[Test]** DNN, 5 retrained copies (each a 5-network ensemble) | **0.923 ± 0.008** accuracy; macro-F1 0.850 ± 0.017 |
| **[Test]** DNN precision / recall / F1 for at-risk; macro-F1; ROC-AUC | 0.714 / 0.750 / 0.732; 0.841; 0.953 |
| **[Test]** Confusion matrix | 104 true pass, 6 false alarms, 5 missed at-risk, 15 caught |
| **[CV]** DNN accuracy / macro-F1 / ROC-AUC | 0.925 ± 0.027 / 0.846 ± 0.051 / 0.971 |
| **[Tuning-CV]** best Optuna trial | macro-F1 0.870 (one network per fold) |
| **Significance** | McNemar vs the G2 rule, logistic regression and XGBoost: p = 0.75, 0.75 and 0.73. **No significant difference.** |

**One-sentence summary for the viva.** The DNN reaches about 91–92% accuracy on unseen students, as good as the best traditional models; with the period grades available, that is the realistic ceiling for this data. The project's value is a leakage-free, reproducible pipeline whose every choice is backed by evidence.

---

## 2. The pipeline end to end

```
data/student-por.csv  (649 rows, 33 columns)
  └─ load_data()                 add target at_risk = (G3 < 10)                       src/data.py
      └─ split()                 stratified 80/20, seed 42 → train 519 | test 130 (locked)
          │
          ├─ tune.py             Optuna, 40 trials × (5-fold × 2) CV on train only → best_params.json
          │
          └─ evaluate.py
               Pipeline = ColumnTransformer(OneHotEncoder | StandardScaler)          src/data.py
                          → SelectKBest(discrete mutual information, k)
                          → DNNClassifier (5 × MLP)                                  src/model.py
               • baselines: GridSearchCV with the same preprocessing, CV and metric
               • every model: 5-fold × 3-repeat CV on train            → CV table
               • refit on all 519 → score 130 test students once       → test table, plots
               • DNN: 5 seeds, Wilson + bootstrap CI, McNemar tests
               • save models/dnn.joblib (the whole fitted pipeline)    → app.py
```

**Files.**
- `src/data.py`: data loading, the target, the split and preprocessing.
- `src/model.py`: the network and its training loop.
- `src/tune.py`: Optuna tuning.
- `src/evaluate.py`: comparison, test scoring, statistics and plots.
- `src/eda.py`: data audit, figures and the feature audit.
- `src/numpy_mlp.py`: the from-scratch NumPy network.
- `src/experiments.py`: the course-practical experiments.
- `app.py`: the Streamlit app.
- `student_performance_dnn.ipynb`: runs everything in order.

---

## 3. Data decisions

### D1. Dataset: Portuguese course, not Maths
- **What.** `student-por.csv`, with 649 students.
- **Why.**
  - It is 64% larger than the Maths file (395 students), and a neural network benefits most from more rows.
  - In the research phase, no model beat the rule "pass if G2 ≥ 10" on Maths (91.9%); the original paper found the same. On Portuguese, models add a few points over the rule.
- **Alternatives.**
  - Maths only: too small, and models lose to a one-line rule.
  - Merging both files: 382 students appear in both, so a naive merge could put the same student in both train and test, which is leakage. It would also mix two subjects with different grading.
- **Where.** `DATA_PATH` in `src/data.py`.
- **Examiner may ask:** *"Why not combine both datasets for more data?"*
  - The same students appear in both files.
  - Without a reliable student ID, a merge risks the same person sitting on both sides of the split, which inflates scores.

### D2. Task: binary classification, at risk = G3 < 10
- **What.** `at_risk = 1 if G3 < 10 else 0`. The positive class is "at risk".
- **Why.**
  - The brief asks for a performance *category*, and 10/20 is the Portuguese pass mark.
  - Making "at risk" the positive class means precision, recall and F1 describe how well the model finds struggling students.
- **How.** One sigmoid output gives P(at risk). The student is flagged if P ≥ 0.5.
- **Alternatives.**
  - **Regression on G3 (0–20).** Possible, but a school acts on pass/fail, and a regression error of ±1.5 grades says less to a teacher.
  - **Five-level grades A–F** (Cortez & Silva's scheme). A research-phase check topped out around 73–76% accuracy even with grades. Each class would have very few students, so a 130-student test set could not support reliable claims.
- **Where.** `load_data()` in `src/data.py`.
- **Examiner may ask:** *"Why not predict the grade itself?"*
  - The decision a school makes is binary: intervene or not.
  - The binary task has enough examples per class (80 at-risk students in training) for honest evaluation.

### D3. Including G1 and G2
- **What.** Both period grades are inputs.
- **Why.** The brief lists academic features. G1 and G2 are recorded *before* G3 (after the 1st and 2nd grading periods), so a model used after the 2nd period legitimately has them. They are also by far the strongest signals: Spearman ρ with G3 is 0.885 and 0.943 [Audit].
- **Why it is not leakage.** Leakage means using information that would not exist at prediction time. G3 itself is the only excluded column.
- **Supporting evidence (ablation).** An earlier version also trained a model with **no grades** (git tag `v1-two-scenarios`, frozen results in `results/supplementary/`). Without grades, no model beat predicting "pass" for everyone on accuracy (84.6%):

  | [Test] no grades | Accuracy | Macro-F1 | ROC-AUC |
  |---|---|---|---|
  | DNN | 0.762 | 0.622 | 0.797 |
  | Random forest | 0.838 | 0.653 | 0.790 |

  The grades carry most of the signal, and the project reports that openly.
- **Examiner may ask:** *"Isn't your model just copying G2?"*
  - Largely, yes: G2 is the strongest predictor, and the rule "G2 < 10" alone scores 90.0% [Test].
  - The DNN combines G2 with G1, past failures, absences, the wish for higher education and school, and reaches 91.5% [Test] and 92.5% [CV] against the rule's 89.6% [CV].
  - That gain is **not statistically significant** on 130 students (McNemar p = 0.75). The honest answer is that the problem is close to linearly separable once G2 is known.

### D4. Missing values, duplicates and outliers
- **What.** No imputation, no deduplication and no outlier removal.
- **Why.**
  - [Audit] 0 missing values and 0 duplicate rows, so there is nothing to impute.
  - All values lie inside the codebook ranges.
- **Absences.** The distribution is right-skewed (skew 2.0, max 32, while the codebook allows 0–93). These are real students, so the column is kept and standardised.
- **The 15 students with G3 = 0.** All 15 also have 0 absences, while only 37.6% of all students do. That is unlikely to be chance; they most likely missed the final exam or dropped out. They are real fails and are kept.
- **Alternatives.** IQR or z-score outlier removal; dropping the G3 = 0 rows.
- **Why not.** Removing real but unusual students makes the test set easier than reality, and it hides exactly the cases an at-risk model must catch. The brief forbids manipulating data to raise accuracy.
- **Where.** `audit()` and `figures()` in `src/eda.py`; notebook section 3.
- **Examiner may ask:** *"Why keep the zero-grade students? Aren't they noise?"*
  - They are the clearest at-risk cases: students who disengaged.
  - Deleting them would raise accuracy dishonestly.

### D5. Class imbalance (15.4% at risk)
- **What.** Stratified splitting; macro-F1 as the tuning metric; a class-weight option (`pos_weight`) offered to Optuna; no resampling.
- **Why.**
  - Predicting "pass" for everyone scores 84.6% accuracy but 0.458 macro-F1, so accuracy alone would reward ignoring the at-risk class.
  - Macro-F1 averages the F1 of both classes and punishes that.
- **How `pos_weight` works.**
  - The loss becomes −[w·y·log p + (1 − y)·log(1 − p)] with w = #pass / #fail ≈ 5.5. Each at-risk student then counts 5.5 times.
  - Optuna tried it: trials with class weights averaged 0.797 [Tuning-CV] macro-F1 against 0.833 without (9 vs 31 trials). It was switched off, because with G1 and G2 available the weight makes the network over-flag.
- **Alternatives and why not.**
  - **SMOTE or oversampling.** Creates synthetic students. If done before splitting it leaks into validation folds, and it adds a dependency. Class weights change the loss, not the data.
  - **Undersampling** throws away 80% of the pass students.
- **Consequence to own.** The DNN's at-risk recall is its weakest number: 0.671 [CV], 0.750 [Test]. Notebook section 10 shows, using out-of-fold predictions on training students only, that lowering the threshold to 0.3 raises recall to 0.84 at precision 0.63. That is a policy choice; the deployed model keeps 0.5.
- **Where.** `split()` in `src/data.py`; `pos_weight` in `DNNClassifier._train_one`; `scoring="f1_macro"` in `src/tune.py`.
- **Examiner may ask:** *"Your DNN misses 1 in 4 at-risk students. Is that acceptable?"*
  - It is a threshold trade-off, not a fixed limit (see the curve in section 10).
  - The model was tuned for balanced macro-F1. A school that values recall would choose a lower threshold on validation data.

### D6. Categorical encoding
- **What.** `OneHotEncoder(drop="if_binary", handle_unknown="ignore")` for the 17 categorical columns. Yes/no columns become a single 0/1 column, and columns with several values become one column per value. The result is 45 columns in total.
- **Why.** Nominal categories such as Mjob ∈ {teacher, health, services, at_home, other} have no order. Coding them 0–4 would invent distances; one-hot does not.
- **Ordinal scales.** Medu, studytime, Dalc and the other 1–5 or 0–4 scales stay numeric, because their order is meaningful and keeping them numeric keeps the input small.
- **Alternatives and why not.**
  - **Label encoding everything** invents an order.
  - **Target encoding** uses the target statistics, which leak unless done carefully inside folds, and is overkill for 2–5 categories.
  - **Learned embeddings** need far more data than 649 rows to learn anything useful.
- **Where.** `make_preprocessor()` in `src/data.py`.
- **Examiner may ask:** *"Why `drop='if_binary'`?"* For a yes/no column, two one-hot columns are perfectly redundant (one is 1 minus the other); one 0/1 column carries the same information.

### D7. Numerical scaling
- **What.** `StandardScaler`: z = (x − μ) / σ, with μ and σ computed on the training rows only.
- **Why.** The inputs range from 0–1 (flags) to 0–32 (absences). Gradient descent converges faster when the inputs share a scale; otherwise the loss surface is stretched along the large-scale directions. Scaled inputs also suit He initialisation and BatchNorm.
- **Alternatives and why not.**
  - **Min-max scaling** is sensitive to the single maximum (absences = 32).
  - **Log transform of absences** changes the meaning of the column with no evidence of benefit, and the network can learn non-linear effects itself.
- **Where.** `make_preprocessor()`.

### D8. Feature inclusion and selection
- **What.** All 32 inputs are candidates; nothing is removed by hand. `SelectKBest` keeps the k = 10 encoded columns with the highest **mutual information** (MI) with `at_risk`, computed as discrete MI.
- **Why select at all.** With 519 training rows, 45 inputs give a small network many noise directions to fit. Optuna tested k ∈ {10, 15, 20, all}: k = 10 averaged 0.833 [Tuning-CV] macro-F1, against 0.791 for all 45 columns, and was chosen in 29 of 40 trials.
- **How MI works.**
  - I(X; Y) = Σₓ Σᵧ p(x, y) · log[p(x, y) / (p(x) p(y))], the reduction in uncertainty about Y from knowing X.
  - It is 0 if X and Y are independent.
  - Unlike correlation, it captures non-linear and non-monotonic links.
- **Why discrete MI** (a fix made during the final audit).
  - Every input is an integer code, a 0/1 flag or a whole-number grade.
  - scikit-learn's default MI estimator assumes continuous data (k-nearest-neighbour, Kraskov). On these discrete columns it was unstable: Pstatus and schoolsup were picked for the final model but kept in only 3 and 2 of the 15 CV folds.
  - With `discrete_features=True`, MI is computed from contingency counts, and **8 of the 10 selected features are kept in all 15 folds** [Audit].
  - The DNN's CV macro-F1 rose from 0.835 to 0.846, and its fold-to-fold spread fell from 0.077 to 0.051 [CV].
  - Standardising does not change MI: it only relabels the distinct values.
- **Final 10 columns:** G2, G1, failures, school_MS, higher_yes, absences, studytime, Fedu, Medu, famrel. Kept in 15/15 folds: G1, G2, failures, higher, school, studytime, Fedu and absences. Medu and famrel are near the cut-off.
- **Feature inclusion / exclusion table** (training data; full version in notebook section 5):

| Column | Group | Kept in CV folds | In final model | Why |
|---|---|---|---|---|
| G2, G1 | Academic | 15/15 | yes | Strongest signals (ρ = 0.94, 0.89) |
| failures | Academic | 15/15 | yes | Past failures, ρ = −0.47 |
| higher | Academic | 15/15 | yes | Wanting higher education: at-risk rate differs by 35 points |
| school | Academic | 15/15 | yes | MS students fail more often (23-point gap) |
| studytime, Fedu, absences | Academic / Demographic / Behavioural | 15/15 | yes | Moderate signals |
| Medu, famrel | Demographic / Social | 10/15, 9/15 | yes | Weak, near the cut-off |
| age, goout | Demographic / Social | 6/15, 5/15 | no | Weak, lose the competition for 10 slots |
| the other 20 columns | all groups | ≤ 3/15 | no | Little information about `at_risk` once grades are known |
| **G3** | target | – | **excluded** | It is what we predict; using it would be leakage |

- **Multicollinearity** [Audit].
  - The variance inflation factor is VIF = 1 / (1 − R²), from the diagonal of the inverse correlation matrix.
  - G1 and G2: VIF ≈ 4.05 and 4.03 (r = 0.86), below the usual warning level of 5. Everything else is below 2.
  - For prediction, correlated inputs are not an error; they only make individual weights hard to interpret.
  - Dalc/Walc (r = 0.62) and Medu/Fedu (r = 0.64) are moderate.
- **Alternatives and why not.**
  - **No selection** scored worse in tuning.
  - **Manual selection by correlation** misses non-linear links and is easy to bias.
  - **Recursive feature elimination** retrains the model many times: expensive with a network and noisy with 519 rows.
  - **L1 regularisation** was not offered in the course flow for networks, and SelectKBest is more transparent to explain.
  - **Model-based importance** (random forest) would tie selection to a different model.
- **Where.** `discrete_mi()` and `make_preprocessor()` in `src/data.py`; `feature_audit()` and `vif()` in `src/eda.py`.
- **Examiner may ask:**
  - *"Was feature selection done on the whole dataset?"* No. It is a Pipeline step, so it is refit inside every CV training fold and on the training set only.
  - *"Which features were removed during experiments?"* None by hand. SelectKBest re-chooses in every fit. The only change in the project's history is the switch from continuous to discrete MI, made because all inputs are discrete and the old selection was unstable.
  - *"Is discrete MI biased?"* The plug-in estimator slightly favours columns with many distinct values, such as absences with 23 values. Absences has a real but weak link to G3 (ρ = −0.14), so its selection is plausible, but this bias is a known limitation.

### D9. Splitting and leakage prevention
- **What.**
  1. **Test split.** `train_test_split(test_size=0.2, stratify=at_risk, random_state=42)` gives 519 training and 130 test students. The test set is touched only in `evaluate.py`, after tuning.
  2. **One Pipeline.** All learned preprocessing (encoder categories, scaler μ/σ, MI scores) lives in one scikit-learn `Pipeline` with the model. `cross_validate`, `GridSearchCV` and `fit` therefore refit it on training rows only.
  3. **Early stopping** uses a stratified 15% slice of the *training* rows in each fit, never the test set.
  4. **Tuning** uses only the training set (`tune.py` never loads the test rows).
  5. **Evaluation CV** uses a different seed (7) from tuning CV (42), so evaluation scores are not the tuning scores reused.
- **Why.** Fitting a scaler or feature selector on all 649 rows would let test-set statistics shape the model. The early research-phase check made exactly this mistake (selection on all rows) and was slightly optimistic; the project avoids it.
- **Alternatives and why not.**
  - **A separate fixed validation set:** with 649 rows it wastes data; CV uses every training student for both training and validation.
  - **Nested CV:** the most rigorous option, but about 40× the compute. The gap between the tuning score (0.870) and the independent CV score (0.846) is the optimism it would remove, and it is reported openly here instead.
- **Examiner may ask:** *"Prove there is no leakage."*
  - Show `make_preprocessor()` returning Pipeline steps, `build_dnn()` putting them in front of the model, and `split()` happening before anything else.
  - Show that the early-stopping split inside `_train_one` comes from the `X` passed to `fit`.

---

## 4. Model decisions

### D10. Architecture: a multilayer perceptron (MLP)
- **What.** 10 inputs → Linear(16) → BatchNorm → ReLU → Dropout(0.48) → Linear(8) → BatchNorm → ReLU → Dropout(0.48) → Linear(1) → logit.
- **Parameter count, per network:**

| Layer | Shape | Weights + biases |
|---|---|---|
| Linear 1 | 10 → 16 | 160 + 16 = 176 |
| BatchNorm 1 (γ, β) | 16 | 32 |
| Linear 2 | 16 → 8 | 128 + 8 = 136 |
| BatchNorm 2 (γ, β) | 8 | 16 |
| Output | 8 → 1 | 8 + 1 = 9 |
| **Trainable total** | | **369** |

  BatchNorm also stores 48 non-trainable running statistics (mean and variance per unit). The final model averages 5 networks: 1,845 trainable weights.
- **Why an MLP.**
  - The data is a table of independent students, with no spatial or sequential structure.
  - A CNN needs local spatial patterns (images); an RNN/LSTM needs order in time. Neither exists here, so the feedforward network is the right DNN.
- **Why this size.**
  - Optuna chose 2 hidden layers with a funnel 16 → 8.
  - Two-layer trials averaged 0.831 [Tuning-CV] macro-F1, against 0.818 for one layer and 0.802 for three.
  - Wider layers did not help (width 128: 0.786).
  - With 519 rows, a small network with strong regularisation generalises best.
- **Alternatives and why not.**
  - **A deeper or wider network:** overfits 519 rows and scored worse in tuning.
  - **CNN, RNN or a Transformer for tables:** no structure for them to exploit, and they need far more data.
  - **One hidden layer:** works almost as well. The NumPy one-layer network reaches 0.929 [CV accuracy, 5-fold]. The two-layer design was chosen by tuning on macro-F1, and the brief asks for a *deep* network.
- **Where.** `MLP` in `src/model.py`; `build_dnn()` in `src/tune.py` turns `n_layers` and `width` into `hidden = (16, 8)`.
- **Examiner may ask:** *"Is 2 hidden layers 'deep'?"* It is a deep feedforward network in the textbook sense (more than one hidden layer). Its depth was chosen by evidence: three-layer trials scored lower.

### D11. Weight initialisation: He, not Xavier
- **What.** `nn.init.kaiming_normal_(W, nonlinearity="relu")`, so W ~ N(0, 2 / fan_in); biases are 0.
- **Why.**
  - ReLU zeroes about half its inputs, which halves the variance passing through each layer.
  - He initialisation doubles the weight variance to compensate, so activations keep roughly the same scale from layer to layer: Var(out) ≈ Var(in) when Var(W) = 2 / fan_in.
- **Xavier/Glorot:** Var(W) = 2 / (fan_in + fan_out). It was derived for tanh and sigmoid, which are symmetric around 0.
- **Evidence** [Val, notebook 15d]: best validation loss He 0.185, Xavier 0.205, all-zeros 0.595.
  - For a network this shallow, He and Xavier behave similarly.
  - All-zeros never learns: it stays at the majority answer (84.6%). Every unit in a layer starts identical, receives the identical gradient, and so stays identical (the symmetry problem).
- **Where.** `MLP.__init__`; comparison in `src/experiments.py` (`init=`).
- **Examiner may ask:** *"What goes wrong with zero initialisation?"* Symmetry. All hidden units compute the same function, so the network behaves as if each layer had one unit. Here the BatchNorm output is also 0, so no gradient reaches the weights at all.

### D12. Activation: ReLU in hidden layers, sigmoid at the output
- **ReLU.** f(z) = max(0, z), with f′(z) = 1 for z > 0 and 0 otherwise.
- **Why ReLU in hidden layers.**
  - Its gradient does not shrink for positive inputs.
  - Sigmoid's derivative σ(1 − σ) is at most 0.25 and tanh's 1 − tanh² is at most 1 but near 0 for large |z|. Stacking those layers multiplies small numbers, so gradients vanish.
  - ReLU is also cheap to compute and gives sparse activations.
- **Evidence** [Val, notebook 15e]: best validation loss ReLU 0.185, tanh 0.255, sigmoid 0.270 after 100 epochs.
- **Output layer.** No activation in the network. The sigmoid σ(z) = 1 / (1 + e^(−z)) is applied inside the loss during training and in `predict_proba`, turning the logit into P(at risk).
- **Alternatives and why not.**
  - **Leaky ReLU or GELU:** fix "dead ReLU" units, but no dead-unit problem appeared, and plain ReLU is the course default.
  - **Softmax with 2 outputs:** mathematically equivalent to one sigmoid output for binary classification, but with twice the output weights.
- **Where.** `MLP` (`nn.ReLU()`); `torch.sigmoid` in `DNNClassifier.predict_proba`.
- **Examiner may ask:** *"Why not sigmoid in the hidden layers?"* Vanishing gradients, and the experiment shows it learns more slowly.

### D13. Loss function: binary cross-entropy with logits
- **What.** `nn.BCEWithLogitsLoss(pos_weight=...)`. For one student, L = −[w·y·log σ(z) + (1 − y)·log(1 − σ(z))]; averaged over the batch.
- **Why.**
  - Binary cross-entropy is the negative log-likelihood of a Bernoulli outcome, so minimising it is maximum-likelihood estimation of P(at risk).
  - Its gradient with respect to the logit is simply σ(z) − y: large when the model is confidently wrong, small when it is right.
  - The *with-logits* version combines the sigmoid and the log in one numerically stable formula (the log-sum-exp trick), avoiding log(0) when σ(z) saturates.
- **Alternatives and why not.**
  - **MSE on probabilities:** combined with the sigmoid, its gradient (p − y)·p(1 − p) vanishes exactly when the model is confidently wrong.
  - **Categorical cross-entropy:** for more than 2 classes; it reduces to BCE here.
  - **Focal loss:** for extreme imbalance; 15% is moderate.
- **Where.** `DNNClassifier._train_one`.

### D14. Backpropagation and mini-batch gradient descent
- **What.** PyTorch autograd computes ∂L/∂w for every weight (`loss.backward()`), and the optimiser updates the weights. Each epoch reshuffles the training rows and visits all of them once, in mini-batches of 32.
- **How backprop works.**
  - The chain rule is applied from the loss backwards through each layer.
  - For the output layer: ∂L/∂z = σ(z) − y.
  - For a hidden layer: δ_l = (δ_{l+1} · W_{l+1}ᵀ) ⊙ ReLU′(z_l), and ∂L/∂W_l = a_{l−1}ᵀ · δ_l.
  - Notebook section 13b implements exactly this in NumPy, and a numerical gradient check confirms it (max relative error 2 × 10⁻⁹).
- **Why mini-batches.**
  - *Batch* GD (all 441 rows per step) is stable but takes one step per epoch: validation loss is still 0.307 after 100 epochs [Val].
  - *Stochastic* GD (1 row per step) is noisy. It reached 0.181 at epoch 9, then rose to 0.683, and it is about 20× slower per epoch.
  - *Mini-batch* GD (32 rows) has useful noise, uses vectorised maths and is the practical middle ground (0.208).
- **Where.** `DNNClassifier._train_one` (the loop); `src/numpy_mlp.py` (manual backprop); `src/experiments.py` (variant comparison).

### D15. Optimiser: AdamW
- **Update rule** for gradient g, with β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸ and decay λ:
  - m ← β₁m + (1 − β₁)g, and v ← β₂v + (1 − β₂)g²
  - m̂ = m / (1 − β₁ᵗ), and v̂ = v / (1 − β₂ᵗ) (bias correction)
  - w ← w − lr · (m̂ / (√v̂ + ε) + λw)
- **Why.**
  - Adam adapts the step size per weight: a large v (a steep, noisy direction) means smaller steps.
  - AdamW applies weight decay λw *separately* from the adaptive step. In plain Adam, an L2 penalty added to the gradient gets divided by √v̂, so heavily updated weights are barely regularised. AdamW fixes that.
- **Alternatives** [Val, notebook 15a, best validation loss over 100 epochs]:

| Optimiser | Update | Best val loss |
|---|---|---|
| SGD (lr 0.01) | w ← w − lr·g | 0.344 |
| SGD + momentum 0.9 | u ← 0.9u + g; w ← w − lr·u | 0.167 |
| RMSprop | s ← 0.99s + 0.01g²; w ← w − lr·g / √s | 0.177 |
| Adam | as above, without decoupled decay | 0.185 |
| AdamW | as above | 0.185 |

  Momentum, RMSprop, Adam and AdamW all work; plain SGD is clearly slowest. With 78 validation students these differences are within noise. AdamW was kept for correct weight decay and because it works well without tuning the learning rate.
- **Where.** `torch.optim.AdamW` in `_train_one`; comparison in `src/experiments.py`.
- **Examiner may ask:** *"SGD with momentum had a lower loss. Why not use it?"*
  - On a single 78-student validation slice, 0.167 vs 0.185 is noise.
  - Momentum SGD's learning rate needs tuning, while Adam-family optimisers are robust to it, and AdamW gives proper weight decay.

### D16. Regularisation: BatchNorm, Dropout and weight decay
- **BatchNorm.**
  - For each unit, over the mini-batch: x̂ = (x − μ_B) / √(σ²_B + ε), then y = γx̂ + β, where γ and β are learned.
  - It keeps each layer's input distribution stable during training, which permits higher learning rates and adds mild regularisation from batch noise.
  - At prediction time (`model.eval()`) it uses running averages, updated during training as running = 0.9·running + 0.1·batch. That is why one student can be predicted alone.
  - Optuna kept BatchNorm on in 35 of 40 trials; BN trials averaged 0.827 vs 0.808 [Tuning-CV].
- **Dropout (p = 0.48).**
  - During training each hidden unit is zeroed with probability p, and the survivors are scaled by 1 / (1 − p) (inverted dropout), so expected activations match at prediction time.
  - It stops units relying on each other, which is like averaging many thinned networks.
  - Optuna chose a high rate (0.48); the small 16 → 8 network trained on 441 rows needs strong regularisation.
- **Weight decay (λ = 0.0084)** shrinks weights towards 0 each step (L2-style), preferring simpler functions.
- **Evidence.** The learning curve (`results/figures/learning_curve.png`) shows training and validation loss levelling off together with no widening gap: no overfitting.
- **Alternatives and why not.**
  - **LayerNorm:** normalises per sample instead of per batch; it suits Transformers and RNNs and has no advantage here.
  - **L1:** produces sparsity, but feature selection already does that job.
  - **Data augmentation:** not meaningful for tabular survey answers.
- **Where.** `MLP` (BatchNorm1d, Dropout); `AdamW(weight_decay=...)`.
- **Examiner may ask:** *"What does BatchNorm do at test time?"* It uses the running mean and variance from training, not the statistics of the test batch. Notebook section 13a recomputes this in NumPy and matches PyTorch to 1 × 10⁻⁷.

### D17. Learning rate, schedules and early stopping
- **Learning rate.** 0.002, tuned by Optuna on a log scale from 10⁻⁴ to 10⁻²; it had the strongest link to trial score (Spearman 0.46 across trials).
- **Schedules** [Val, notebook 15b]. Constant 0.001: 0.185. Step decay (halve every 30 epochs): 0.232. Exponential decay (×0.97 per epoch): 0.335. Starting from an already small rate, decay just slows learning. A schedule would pay off with a larger starting rate.
- **Early stopping.**
  - Up to 300 epochs. A stratified 15% of the training rows is held out, and the validation loss is checked every epoch.
  - Training stops after 30 epochs without improvement, and the **best epoch's weights are restored** (best epoch 176 for the first network).
  - This picks the training length automatically and acts as regularisation.
- **Alternatives and why not.**
  - **A fixed number of epochs:** has to be guessed, and either underfits or overfits.
  - **ReduceLROnPlateau:** a reasonable addition, but early stopping already handles the plateau.
- **Where.** `_train_one` (patience loop, `best_state`); schedules in `src/experiments.py`.

### D18. Seed ensemble of 5 networks
- **What.** `n_models=5`: five networks with seeds 42–46, each with its own random initialisation, batch order and early-stopping split. `predict_proba` averages their probabilities.
- **Why.** A single small network on 519 rows is sensitive to its random start: single networks score between 0.885 and 0.931 on the test set depending on the seed. Averaging cancels much of that randomness. In v1, the ensemble cut the seed-to-seed test-accuracy spread from 0.010 to 0.004.
- **Alternatives and why not.**
  - **A single network:** less stable.
  - **Bagging (bootstrap resampling):** throws away about 37% of the rows per network.
  - **Stacking the DNN with XGBoost:** blurs the "DNN" claim and adds complexity for no proven gain.
- **Where.** `DNNClassifier.fit` and `predict_proba`. Tuning uses 1 network per fold for speed; the final model uses 5.
- **Examiner may ask:** *"Is the ensemble taught in the course?"* Not as such. It is the idea of averaging away variance, the same idea behind dropout and random forests, applied to random initialisation.

### D19. Hyperparameter tuning with Optuna
- **What.** `optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))`, 40 trials, over 9 hyperparameters:

| Hyperparameter | Search space | Chosen |
|---|---|---|
| n_layers | 1–3 | 2 |
| width | 16, 32, 64, 128 | 16 |
| dropout | 0–0.5 | 0.48 |
| batchnorm | yes / no | yes |
| lr | 10⁻⁴ to 10⁻², log scale | 0.0020 |
| weight decay | 10⁻⁵ to 10⁻², log scale | 0.0084 |
| batch size | 16, 32, 64 | 32 |
| class weights | yes / no | no |
| k features | 10, 15, 20, all | 10 |

  Each trial is scored by **mean macro-F1 over 5-fold × 2-repeat CV on the training set** (10 folds), with 1 network per fold. Every trial is appended to `results/experiments.csv`.
- **How TPE works.**
  - The first 10 trials are random.
  - After that, TPE (Tree-structured Parzen Estimator) splits past trials into "good" (top scores) and "bad".
  - It models the density of hyperparameters in each group, l(x) and g(x), and proposes the x that maximises l(x) / g(x): values common among good trials and rare among bad ones.
- **Why.** 9 dimensions with continuous ranges are too many for a grid. A grid of only 3 values each is 3⁹ ≈ 20,000 combinations, while TPE found a strong region in 40 trials. The top 5 trials all share 2 layers, width 16, high dropout, BatchNorm and k = 10, which is a stable optimum.
- **Alternatives and why not.**
  - **Grid search:** used for the baselines, whose grids are small, but far too expensive for 9 hyperparameters.
  - **Random search:** reasonable, but it does not learn from earlier trials. TPE begins with 10 random trials anyway.
  - **Manual tuning:** not reproducible and easily biased.
- **Optimism.** The best tuning score (0.870) is higher than the independent CV score of the final model (0.846), because picking the best of 40 noisy scores favours lucky ones. That is why the final numbers come from a separate CV and the locked test set.
- **Where.** `src/tune.py`; trial log in `results/experiments.csv`; notebook section 8.

### D20. PyTorch as the main framework, Keras as a twin
- **What.** The main model is in PyTorch. Notebook section 14 rebuilds the same network in Keras 3:
  - with both the **Sequential** and the **Functional** API, which give the same 369 trainable parameters;
  - trained with `EarlyStopping(restore_best_weights=True)` and `ModelCheckpoint`;
  - the checkpoint is then **reloaded** with `keras.models.load_model`.
- **Why PyTorch for the main model.** The explicit loop (`zero_grad → forward → loss → backward → step`, `train()` / `eval()`) shows every step of training. That is what a line-by-line viva needs, and it makes custom pieces such as `pos_weight` and the seed ensemble easy. Wrapping it as a scikit-learn estimator (`DNNClassifier`) puts it in the same Pipeline, CV and metrics code as the baselines.
- **Why also Keras.** The course covers both frameworks. Keras's `fit` with callbacks is the high-level equivalent: `EarlyStopping` replaces the patience loop, and `ModelCheckpoint` replaces `best_state`.
- **Evidence** [Test, single networks]: Keras 0.923, PyTorch (seed 42) 0.885. Single networks of either framework vary by about ±3 students with their random start, so this is not a framework difference. Keras uses BatchNorm momentum 0.99 against PyTorch's 0.1 convention, and a different validation split.
- **Backend.** Keras 3 runs on the PyTorch backend (`KERAS_BACKEND=torch`), so TensorFlow is not needed.
- **Where.** Notebook section 14; the `DNNClassifier` wrapper in `src/model.py`.

### D21. NumPy implementation
- **What.**
  1. **Forward pass of the trained network in plain NumPy** (notebook 13a): Linear, then BatchNorm with running statistics, then ReLU, then sigmoid. It matches PyTorch to 1.2 × 10⁻⁷.
  2. **A network trained from scratch** (`src/numpy_mlp.py`, notebook 13b): 10 → 16 → 1, hand-derived backprop, mini-batch GD at lr 0.1 for 200 epochs, He initialisation.
  3. **A numerical gradient check:** (L(w + ε) − L(w − ε)) / 2ε against the backprop gradient; max relative error 2 × 10⁻⁹.
- **Results.** NumPy network: 0.929 ± 0.038 [CV, 5-fold accuracy]; 0.923 accuracy and 0.863 macro-F1 [Test].
  It does as well as the PyTorch ensemble. This is further evidence that, given G1 and G2, a small network suffices and the ceiling is the data, not the model.
- **Why.** It proves the maths behind the framework: course labs 2 and 6, and CLO01 and CLO02.
- **Where.** `src/numpy_mlp.py` (self-check: `python -m src.numpy_mlp`).

### D22. Evaluation metrics and statistics
| Metric | Formula / meaning | Why reported |
|---|---|---|
| Accuracy | (TP + TN) / N | The brief's headline; misleading alone (84.6% for "everyone passes") |
| Precision (at risk) | TP / (TP + FP) | How many flagged students really fail (false-alarm cost) |
| Recall (at risk) | TP / (TP + FN) | How many failing students are found (the cost of missing someone) |
| F1 | 2PR / (P + R) | Balances P and R for one class |
| **Macro-F1** (tuning metric) | mean of F1(pass) and F1(at risk) | Punishes ignoring the minority class |
| Balanced accuracy | mean of the two class recalls | Accuracy corrected for imbalance |
| ROC-AUC | P(score of a random at-risk student > score of a random pass student) | Ranking quality, independent of the threshold |
| Confusion matrix | counts of TN, FP, FN, TP | Shows exactly where the errors are |
| Wilson 95% CI | (p̂ + z²/2n ± z√(p̂(1−p̂)/n + z²/4n²)) / (1 + z²/n) | Honest uncertainty for a proportion with small n |
| Bootstrap 95% CI | resample the 130 test students 1000 times | Same question, without a formula |
| McNemar exact test | binomial test on the students where exactly one model is right | Is the difference between two models real? |
| 5-seed test score | retrain with seeds 0, 100, …, 400 | Sensitivity to the random start (not to the data sample) |

- **Why these.** Accuracy is the number examiners expect. Recall, precision and macro-F1 expose the imbalance. ROC-AUC separates ranking quality from the 0.5 threshold. The confidence intervals and McNemar test stop over-claiming on a 130-student test set.
- **Where.** `test_metrics()`, `bootstrap_ci()`, `wilson_ci()`, `mcnemar()` and the `classification_report` call in `src/evaluate.py`.

### D23. Traditional ML comparison
- **What.** Five baselines, all with the same preprocessing Pipeline and the same CV:
  - **Majority class:** the floor.
  - **The "G2 < 10" rule:** the domain baseline from Cortez & Silva 2008.
  - **Logistic regression:** grid over C, class weights and k.
  - **Random forest:** 300 trees; grid over depth, minimum leaf size and class weights.
  - **XGBoost:** grid over depth, number of trees and `scale_pos_weight`.

  Each grid is searched with 5-fold CV on macro-F1, the same metric used for the DNN.
- **Results:**

| Model | [CV] accuracy | [CV] macro-F1 | [CV] recall | [Test] accuracy | [Test] recall | [Test] macro-F1 | [Test] ROC-AUC |
|---|---|---|---|---|---|---|---|
| Majority class | 0.846 | 0.458 | 0.000 | 0.846 | 0.00 | 0.458 | 0.500 |
| Rule G2 < 10 | 0.896 | 0.831 | 0.887 | 0.900 | 0.90 | 0.837 | 0.953 |
| Logistic regression | **0.934** | **0.870** | 0.771 | 0.900 | 0.75 | 0.819 | 0.944 |
| Random forest | 0.921 | 0.867 | **0.908** | 0.862 | 0.85 | 0.784 | 0.946 |
| XGBoost | 0.929 | 0.866 | 0.804 | 0.900 | 0.80 | 0.825 | 0.925 |
| **DNN** | 0.925 | 0.846 | 0.671 | **0.915** | 0.75 | **0.841** | **0.953** |

- **Interpretation.**
  - In CV, logistic regression is the strongest overall, random forest has the best recall and the DNN sits in the same pack.
  - On test, the DNN has the best accuracy and macro-F1.
  - McNemar's tests show no significant difference between the DNN and the rule, logistic regression or XGBoost (p > 0.7).
  - Tree ensembles and linear models are known to match or beat neural networks on small tabular datasets (Grinsztajn et al., NeurIPS 2022). Matching them is the realistic result.
- **Examiner may ask:** *"So why use a DNN?"*
  - The brief requires one.
  - This project shows, with evidence, that a carefully regularised DNN reaches the same ceiling as strong classical models.
  - Being honest that it does not significantly beat them is the scientifically correct conclusion.

### D24. CPU, not GPU
- **What.** Everything runs on the CPU.
- **Evidence** (notebook section 16, 50 epochs of the final network):
  - MacBook Air M3: CPU 0.25 s vs Apple GPU (MPS) 0.74 s.
  - Kaggle: CPU 1.66 s vs NVIDIA Tesla T4 GPU 1.81 s.
- **Why.**
  - Each step multiplies a 32 × 10 batch by a 10 × 16 matrix, a few thousand operations.
  - Launching a GPU kernel and copying data costs more than the maths itself; GPUs pay off with large matrices such as images and big batches.
  - `OMP_NUM_THREADS=1` is set because one thread is fastest on data this small, and it also avoids a crash when PyTorch and XGBoost share a process on macOS (each bundles its own OpenMP).
- **Where.** The top of `src/tune.py`, `src/evaluate.py` and `src/experiments.py`; notebook section 16.

---

## 5. Application and engineering decisions

### D25. Streamlit app
- **Inference uses exactly the training preprocessing.**
  - The app loads `models/dnn.joblib`, the *entire fitted Pipeline*: encoder categories, scaler μ/σ, selected columns and the 5 networks.
  - The input row is ordered with `[FEATURES]`, the training column order.
  - Nothing is retrained when the app starts; `st.cache_resource` loads the model once.
- **Inputs.**
  - All 32 inputs are grouped as in the brief: Demographic, Academic, Social and Behavioural.
  - Codebook labels replace codes, and defaults are the most typical training value.
  - Sliders and select boxes restrict values to the codebook ranges. Absences above the training maximum (32) show a reliability warning.
  - The 10 inputs the model actually uses are marked "used by model".
- **Output.**
  - PASS / AT RISK, with P(fail) and the 50% cut-off shown.
  - A warning when P is within 15 points of the cut-off.
  - An **occlusion explanation**: each used input is replaced by the typical value, and the change in risk is shown.
- **Model tab.** [Test] metrics with the confusion matrix, ROC curves and the baseline table.
- **Why occlusion and not SHAP.** It is a few lines, needs no new dependency, and is easy to explain: "what if this input were typical?"
- **Where.** `app.py`, `.streamlit/config.toml`.

### D26. Reproducibility
- **Seeds.** `SEED = 42` for the split, tuning and networks; CV seed 7 for evaluation; torch seeds per network.
- **Versions.** Pinned in `requirements.txt`, Python 3.12.
- **Experiment log.** Every Optuna trial is logged. The previous versions' logs and metrics are kept in `results/supplementary/`, and the early-warning version at git tag `v1-two-scenarios`.
- **Checks.**
  - A fresh clone with a new virtual environment reproduces `best_params.json` and `metrics.json` exactly (see the README).
  - The notebook runs end to end locally and on Kaggle (Python 3.13, torch 2.11, scikit-learn 1.6.1). On Kaggle the DNN's results were identical: test 0.915, 5 seeds 0.923 ± 0.008. XGBoost moved from 0.900 to 0.877 (3 students), because Kaggle has a different xgboost version.

---

## 6. CHO (course handout) compliance matrix

**Status key:**
- **Implemented:** working code in the project, with output.
- **Partial:** part of the practical is covered.
- **Not implemented:** relevant, but not done.
- **Not required:** does not apply to a tabular binary-classification project.

### Lab plan

| Lab | Practical | Status | Where / note |
|---|---|---|---|
| 1 | NumPy vectors, matrices, tensors, broadcasting | Partial | Used throughout `numpy_mlp.py` and notebook 13a (matmul, broadcasting of biases and BatchNorm); no standalone exercise |
| 2 | Derivatives, chain rule, numerical gradient checking | Implemented | `gradient_check()` in `src/numpy_mlp.py`, max error 2 × 10⁻⁹ |
| 3 | Preprocessing: missing values, scaling, encoding, train/val/test split, visualisation | Implemented | `src/data.py`, `src/eda.py`; no missing values exist (checked) |
| 4 | Perceptron from scratch, decision boundary | Not implemented | Not needed for the project; the NumPy MLP covers the same forward and update ideas |
| 5 | Activation functions and losses (sigmoid, tanh, ReLU, softmax; MSE, BCE, CCE) | Partial | Activations and derivatives plotted and compared (notebook 15e); BCE used and explained; softmax, MSE and CCE explained in D12 and D13 but not plotted |
| 6 | Feedforward ANN from scratch in NumPy with backprop | Implemented | `src/numpy_mlp.py`, notebook 13b |
| 7 | Batch vs stochastic vs mini-batch GD with loss curves | Implemented | `src/experiments.py`, notebook 15c (loss curves; accuracy in the table) |
| 8 | Xavier/He initialisation, BatchNorm, Dropout, L2, early stopping | Implemented | `src/model.py`; He vs Xavier vs zeros in notebook 15d; L2-style decay via AdamW's decoupled weight decay (no separate L2 loss term) |
| 9 | Keras Sequential and Functional API, callbacks, checkpoints, save/load, history plot | Implemented | Notebook 14 |
| 10 | PyTorch tensors, autograd, Dataset/DataLoader, nn.Module, training and validation loop | Implemented | `src/model.py` (nn.Module, autograd, train and validation loop); `TensorDataset` + `DataLoader` in `src/experiments.py`. The main model slices shuffled tensors directly, which is what DataLoader does internally |
| 11 | SGD, Momentum, RMSprop, Adam/AdamW, LR schedule, hyperparameter tuning | Implemented | Notebook 15a and 15b; Optuna in `src/tune.py`; grid search for the baselines |
| 12–14 | CNN, augmentation, transfer learning, CNN error analysis | Not required | No images; a CNN has no spatial structure to exploit (D10) |
| 15 | LSTM/GRU forecasting | Not required | No sequences or time series |

### Theory topics (CLO01–CLO03)

| Topic | Status | Where |
|---|---|---|
| Probability, likelihood, cross-entropy link | Implemented (explained) | D13: BCE as the Bernoulli negative log-likelihood |
| Bias–variance, underfitting / overfitting | Implemented | Learning curve, early stopping, dropout, seed ensemble (D16–D18) |
| Feedforward networks, logits, probabilities, matrix form | Implemented | `MLP`, notebook 7 and 13a |
| Backpropagation, computational graph, gradient flow | Implemented | Autograd plus manual backprop (D14, D21) |
| Initialisation (zero pitfall, Xavier, He) | Implemented | D11 |
| Input normalisation, BatchNorm | Implemented | D7, D16 |
| LayerNorm | Not implemented | D16 explains why it is not needed |
| L1 / L2, dropout, early stopping | Partial | L2-style weight decay, dropout and early stopping implemented; L1 not used (D8, D16) |
| Optimisers, learning-rate schedules | Implemented | D15, D17 |
| Grid search, random search, Optuna | Implemented | Grid (baselines), Optuna TPE (its first 10 trials are random search) |
| Accuracy, precision, recall, F1, ROC-AUC, confusion matrix | Implemented | D22 |
| Validation strategy, learning curves, experiment log, data leakage, reproducibility | Implemented | D9, D26, `results/experiments.csv` |
| Multiclass softmax | Not required | Binary task; a sigmoid output is the binary special case of softmax |
| CNN, RNN, LSTM, GRU (CLO04, CLO05) | Not required | Not applicable to tabular data |

### Project brief

| Requirement | Status |
|---|---|
| Data preprocessing | Implemented (D4–D7, D9) |
| Exploratory data analysis | Implemented (`src/eda.py`, notebook 3) |
| Feature selection | Implemented (D8) |
| DNN architecture design | Implemented (D10–D16) |
| Model training | Implemented (D14, D17, D18) |
| Hyperparameter tuning | Implemented (D19) |
| Performance evaluation | Implemented (D22) |
| Comparison with at least one traditional ML algorithm | Implemented: three tuned models plus two baselines (D23) |
| Streamlit app that accepts student details and displays the predicted category | Implemented (D25) |
| Dataset: UCI Student Performance, target G3 | Implemented (Portuguese, G3 < 10) |

---

## 7. Results history and honest notes

The locked test set was scored once per version. Every change was driven by cross-validation evidence or a methodological audit, never by the test score.

| Version | What changed and why | [CV] accuracy | [CV] macro-F1 | [Test] accuracy | [Test] 5 seeds |
|---|---|---|---|---|---|
| v1.0 | Single network, 3 × 64-32-16, continuous (k-NN) MI | – | – | 0.908 | – |
| v1.1 | 5-network ensemble, because the CV macro-F1 spread between folds (0.072) was far above the other models (about 0.045) | 0.919 | 0.835 | 0.900 | 0.912 ± 0.004 |
| **v2 (final)** | Discrete MI (all inputs are discrete; v1 selection was unstable across folds), and Optuna re-run with 5×2 CV to reduce tuning noise | **0.925** | **0.846** | **0.915** | **0.923 ± 0.008** |

**Honest notes.**
- **The test set has been looked at three times.** Each change had an independent justification, and the v2 CV estimate (92.5%) agrees with the test result (91.5%). Some optimism still cannot be ruled out. The most defensible single number is the **CV accuracy, 92.5% ± 2.7%**, with the locked test set (91.5%, 95% CI 85.5–95.2%) confirming it.
- **The research-phase claim of 92.9% is not used.** That check selected features on all 649 rows, which is optimistic.
- **The earlier 91.2% headline** (v1.1, 5 seeds) is superseded by v2. Both are listed here.
- **The no-grades ablation** is frozen at tag `v1-two-scenarios`; it was not re-run with discrete MI.

---

## 8. Limitations

- **Small data.**
  - 649 students from two schools in one year (2005–06), so the results will not transfer to other schools or years without new data.
  - The test set has 130 students, only 20 of them at risk: one student moves accuracy by 0.8 points and recall by 5.
- **Recall.** At the default threshold, 5 of 20 at-risk test students are missed. Lowering the threshold trades false alarms for recall (notebook 10).
- **Dependence on G2.** Without the period grades, the task is much harder (ablation, D3).
- **Feature selection.** Discrete MI slightly favours columns with many distinct values. The last two selected columns (Medu, famrel) vary between folds.
- **Not significantly better than the baselines.** McNemar p > 0.7 against the strong baselines.
- **Sensitive attributes.** Sex, family situation and address are inputs. The model should prompt a conversation with a student, not decide anything about them.

---

## 9. Rapid-fire viva questions

1. **What does your model predict?** Whether G3 will be below 10/20 (at risk) or not, from 32 attributes including G1 and G2.
2. **Final accuracy?** 91.5% on 130 locked test students (95% CI 85.5–95.2%); 92.3 ± 0.8% over 5 retrained seeds; 92.5% in cross-validation.
3. **Why is accuracy not your tuning metric?** "Everyone passes" already scores 84.6%. Macro-F1 punishes ignoring the at-risk class.
4. **How many parameters?** 369 trainable per network (see the D10 table), 5 networks in the ensemble, plus 48 BatchNorm running statistics each.
5. **Why He initialisation?** ReLU halves the variance; Var(W) = 2 / fan_in restores it. Zero initialisation fails because of symmetry.
6. **Why ReLU?** No saturation for positive inputs, so no vanishing gradient. It beat tanh and sigmoid in our experiment.
7. **Why BCEWithLogitsLoss?** Maximum-likelihood loss for a binary outcome. Its gradient is p − y, and the fused sigmoid is numerically stable.
8. **Explain one training step.** `optimizer.zero_grad()` → forward pass gives logits → loss → `loss.backward()` fills `.grad` via the chain rule → `optimizer.step()` applies AdamW.
9. **What is BatchNorm doing at prediction?** It uses running mean and variance from training (`model.eval()`).
10. **What does dropout do at prediction?** Nothing. Inverted dropout already scaled the activations during training.
11. **Why AdamW over Adam?** Weight decay is applied separately from the adaptive step, so every weight is regularised equally.
12. **How does early stopping work here?** 15% of the training rows are held out; training stops after 30 epochs without improvement and the best weights are restored.
13. **How did you tune?** Optuna TPE: 40 trials, 9 hyperparameters, 5-fold × 2-repeat CV macro-F1 on training data only.
14. **How do you prevent leakage?** Split first. All preprocessing and selection sit inside the Pipeline and are refit on training folds. Early stopping and tuning never see the test set.
15. **Why mutual information with discrete features?** All inputs are discrete codes. The continuous estimator was unstable, and the discrete one keeps 8 of 10 features in every fold.
16. **Which features matter most?** G2, G1, past failures, wanting higher education, school and absences (selection stability plus permutation importance, notebook 11).
17. **Is G1/G2 leakage?** No. They are known before G3. Without them, nothing beats the majority class (ablation).
18. **Is your DNN better than XGBoost?** Equal within noise: McNemar p = 0.73.
19. **Why not a CNN or LSTM?** No spatial or sequential structure in a table of independent students.
20. **Why CPU?** The network is tiny; GPU launch overhead is larger than the computation (benchmarks: M3 CPU 0.25 s vs MPS 0.74 s; Kaggle CPU 1.66 s vs T4 1.81 s).
21. **How does the app guarantee the same preprocessing?** It loads the whole fitted Pipeline and orders the columns as in training; it never refits.
22. **What would you improve with more time?** Nested CV for an unbiased tuning estimate; choosing the threshold on validation data to match a school's recall target; more students and more years of data.
23. **What does the confusion matrix say?** 104 passes correct, 6 false alarms, 5 at-risk students missed, 15 caught.
24. **What is ROC-AUC 0.953?** If you pick a random at-risk and a random passing student, the model ranks the at-risk one higher 95.3% of the time.
25. **Why did the 91.2% from before change?** A methodological fix (discrete MI) and less noisy tuning, both decided on CV. The history is in section 7.
