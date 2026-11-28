# Student Performance Prediction with a Neural Network

Predicts whether a secondary-school student will **PASS** (final grade G3 ≥ 10 out of 20) or is **AT RISK** of failing, using a Keras neural network. A Streamlit app shows the prediction with an explanation.

Course project for 24CAI0306 Deep Neural Networks (B.E. CSE-AIML, Semester 5).

| Read this | What it is |
|---|---|
| [`GUIDE.md`](GUIDE.md) | **Start here.** Everything in plain words: data, choices and why, results, app, how to present, questions and answers |
| [`CODE_EXPLAINED.md`](CODE_EXPLAINED.md) | Every line of code with a simple explanation |
| [`student_performance_dnn.ipynb`](student_performance_dnn.ipynb) | The notebook, step by step (also on [Kaggle](https://www.kaggle.com/code/kavinthakurkm/student-performance-dnn)) |

## In short

| | |
|---|---|
| Data | [UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance), Portuguese course: 649 students, 32 inputs (Cortez & Silva 2008, CC BY 4.0) |
| Split | 519 training students, 130 test students (kept aside until the end) |
| Inputs used | Top 5 by information score: G2, G1, past failures, school, wants higher education |
| Model | 5 inputs → Dense 16 (ReLU) → Dropout 0.4 → Dense 8 (ReLU) → Dropout 0.4 → Dense 1 (sigmoid) |
| Training | Adam (learning rate 0.01), binary cross-entropy, batch 32, early stopping |
| Choices | Number of inputs, layer sizes, dropout, learning rate, activation and optimizer chosen by 5-fold cross-validation |

## Results (130 test students)

| Model | Test accuracy | Test F1 | CV F1 | CV recall |
|---|---|---|---|---|
| **Neural network** | **0.931** | 0.769 | **0.767** | **0.812** |
| Logistic regression | 0.923 | 0.750 | 0.753 | 0.725 |
| Decision tree | 0.938 | 0.778 | 0.755 | 0.662 |
| Random forest | 0.900 | 0.711 | 0.751 | 0.725 |
| Always say "pass" | 0.846 | 0 | – | – |

Confusion matrix of the neural network: 106 passes correct, 15 of 20 failing students caught, 4 false alarms, 5 missed.

On Kaggle (different library versions, so slightly different random starting weights) the same notebook gets 92.3% test accuracy with recall 85%.

## Run it

Needs Python 3.12.

```bash
git clone https://github.com/auraCodesKM/student-performance-dnn.git
cd student-performance-dnn
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

streamlit run app.py                         # the app (uses the saved model)
jupyter lab student_performance_dnn.ipynb    # the notebook (about 15 seconds)
python -m src.train                          # redo all experiments and training (about 25 minutes)
python -m src.eda                            # data checks and graphs
```

## Files

```
app.py                      Streamlit app (tabs: Predict, Results & graphs, Feature choices, About)
src/data.py                 load data, target, split, encoding + scaling + column selection
src/model.py                the Keras neural network and its training
src/train.py                experiments (cross-validation), final training, comparison, graphs
src/eda.py                  data checks and graphs
src/reasons.py              why each input is or isn't used (shown in the app)
models/                     saved preprocessor and network used by the app
results/results.json        all scores and experiment results
results/figures/            all graphs
data/                       the UCI data files and codebook
```

Earlier, more advanced versions are kept in git history (tags `v1-two-scenarios` and `v2-advanced`).
