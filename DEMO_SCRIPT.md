# Streamlit demo script (about 7 minutes)

What to show in the viva, in order, with the exact inputs to set and the result you will see.
Every number below was checked against the saved model (`models/dnn.joblib`).

## Before the viva

```bash
cd ~/Desktop/DNN && source .venv/bin/activate
streamlit run app.py        # opens http://localhost:8501
```
- Open it **before** you are called in. The first load takes a few seconds while the model loads; after that, predictions are instant.
- Keep the notebook open in a second tab (`jupyter lab student_performance_dnn.ipynb`) as a backup and for questions about the code.
- To reset every input to its default, reload the browser page (**Cmd + R**). Pressing **R** inside the app only reruns it and keeps your inputs.

---

## Step 1: what the app is (30 s)

Point at the title and the line under it.

> "This predicts whether a student will pass the year, meaning a final grade G3 of at least 10 out of 20, or is at risk of failing. It is the DNN from my notebook, trained on 519 Portuguese students. The app doesn't train anything; it loads the saved pipeline, so every prediction goes through exactly the same preprocessing as in training."

## Step 2: the input form (45 s)

Scroll slowly through the four boxes: **Demographic, Academic, Social, Behavioural**. These are the four groups the project brief names.

> "All 32 attributes from the UCI dataset are here, grouped as the brief describes. They start at the most typical value among the training students.
> Ten of them are marked 'used by model'. Feature selection inside the pipeline kept the 10 most informative columns: G1, G2, past failures, absences, study time, wanting higher education, school, and parents' education and family relations. The others are shown, but the model ignores them."

## Step 3: a typical student, PASS (30 s)

Leave everything at the defaults (G1 = 11, G2 = 11, no failures).

| You see | |
|---|---|
| Prediction | **PASS** |
| Probability of failing | **1%** |
| Why this result | "No input the model uses moves the risk by 1 point or more compared with a typical student." |

> "With average grades of 11 and no past failures, the risk is about 1%."

## Step 4: a weak student, AT RISK (45 s)

Set **1st period grade G1 = 6** and **2nd period grade G2 = 6**.

| You see | |
|---|---|
| Prediction | **AT RISK** |
| Probability of failing | **72%** |
| Why this result | 2nd period grade = 6 raises the risk by **42%**; 1st period grade = 6 raises the risk by **22%** |

> "Low period grades push the risk over the 50% cut-off. The 'Why' panel uses occlusion: it puts each input back to the typical value and measures how much the risk changes. G2 matters most, which matches the feature selection and the permutation importance in my notebook."

## Step 5: a borderline student, where the other features matter (1.5 min)

This is the most important part of the demo. It shows that the model is **not just the rule "G2 < 10"**.

Set **G1 = 9, G2 = 9**, then change one input at a time:

| Change | Probability of failing | Prediction |
|---|---|---|
| G1 = 9, G2 = 9 | 29% | PASS |
| + Past class failures = **1** | 42% | PASS, with the note "close to the cut-off: treat as uncertain" |
| + Wants higher education = **No** | 46% | PASS, uncertain |
| + School = **Mousinho da Silveira** | **52%** | **AT RISK**, uncertain |

> "The naive rule would call this student at risk just because G2 is below 10. My model says a 9/9 student is borderline, and it is the other factors (a past failure, no plans for higher education, the school) that tip the balance. Those are the patterns the network learned beyond the grades. When the probability is near 50%, the app says so instead of pretending to be certain."

## Step 6: the pass mark matters (20 s)

Reload the page (**Cmd + R**) to reset, then set **G1 = 9, G2 = 10**.

| You see | |
|---|---|
| Prediction | **PASS**, probability **7%** |

> "Moving G2 from 9 to 10 drops the risk sharply. The network has learned that G2 crossing the pass mark is the strongest single signal."

## Step 7: input validation (15 s)

Set **School absences = 40**.

Reload the page first.

> "The codebook allows up to 93 absences, but the training data only goes up to 32. The app still predicts, but it warns that this is outside the range the model has seen."

## Step 8: the Model performance tab (1.5 min)

Click **Model performance**.

1. **The five metric cards**: accuracy 91.5%, precision 71.4%, recall 75.0%, macro-F1 0.841, ROC-AUC 0.953.
   > "These are on 130 test students that were locked away before any training or tuning. Predicting 'pass' for everyone already gives 84.6% accuracy, which is why I focus on recall and macro-F1."
2. **The confusion matrix**: 104 passes correct, 6 false alarms, 15 of 20 at-risk students caught, 5 missed.
3. **The comparison table and chart** (the comparison the brief asks for):
   > "Same preprocessing and the same cross-validation for every model. The DNN has the best test accuracy, 91.5% against 90.0% for logistic regression, XGBoost and the G2 rule. But in cross-validation logistic regression is slightly ahead, and McNemar's test shows the differences are not statistically significant. My honest conclusion: the DNN matches the best traditional models; it does not clearly beat them. With the period grades known, about 92% is the ceiling for this data."
4. **The architecture line**: 10 → 16 → 8 → 1, BatchNorm, ReLU, Dropout, He initialisation, AdamW, early stopping, an average of 5 networks.

## Step 9: the Feature choices tab (1 min, or whenever sir asks "why not feature X?")

Click **Feature choices**. The lookup box opens on **address**.

> "All 32 inputs were candidates. On the training students only, I ranked every column by mutual information and kept the top 10; Optuna chose 10 over 15, 20 or all 45. Then I checked the choice: I added each unused input back, and removed each used one, on the same 15 cross-validation folds.
> Retraining the same model with a different random seed already moves the score by 0.017, and only removing G1 or G2 changes it by more than that.
> Address, for example: rural students do fail more, 21% vs 13%. But they also have lower grades and most attend MS. Among students with the same G2, rural and urban fail at the same rate, so the grades and school already carry that information. Adding address back changed the score by +0.003."

Pick any other input sir names (sex, alcohol, internet…) and read its card. Every input also has a **(?) tooltip** on the Predict form with the same reason.

## Step 10: the About tab (15 s)

> "Dataset source and licence, method in one paragraph, and a responsible-use note. The inputs include sensitive attributes such as sex and family situation, so this should start a conversation with a student, never make a decision about them."

---

## Questions sir may ask during the demo

| Question | Answer |
|---|---|
| *Is it retraining when I change inputs?* | No. `load_model()` is cached with `st.cache_resource`; changing an input only runs `predict_proba` on one row (milliseconds). |
| *How do you make sure the app's preprocessing matches training?* | `models/dnn.joblib` is the **whole fitted Pipeline**: the one-hot encoder's categories, the scaler's mean and std, the selected columns and the 5 networks. The input row is ordered with `[FEATURES]`, the training column order. Nothing is refitted. |
| *Why is the probability 52% "at risk" but 46% "pass"?* | The decision cut-off is 0.5. Near the cut-off the app shows "treat as uncertain". Notebook section 10 shows how a lower cut-off would catch more at-risk students at the cost of more false alarms. |
| *What does the 'Why' panel compute?* | Occlusion: for each input the model uses that differs from the typical value, it sets that one input back to typical and records the change in P(fail). |
| *Why does the model ignore address, alcohol, health and so on?* | Open the **Feature choices** tab and look the input up. Each one ranked below the top 10 by mutual information, and adding it back changed CV macro-F1 by less than the 0.017 seed-to-seed noise. Only G1 and G2 change the result when removed. |
| *What if G1 and G2 are not known yet?* | That is the supplementary "no grades" experiment: without them no model beats predicting "pass" for everyone (84.6%). The grades carry most of the signal. |
| *Can it run online?* | Yes. `streamlit run app.py` locally, or deploy to Streamlit Community Cloud from the GitHub repo (not done, because it is not required). |

## If the app fails during the viva

- Open the notebook, **section 12**: the same prediction for six real test students, produced by the same saved pipeline.
- Then **section 10**: the confusion matrix, ROC curve and comparison chart.
- The Kaggle notebook (link in `README.md`) shows the same outputs online.
