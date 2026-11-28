"""Streamlit app: enter a student's details and the neural network predicts PASS or AT RISK.

Run:  streamlit run app.py
The app does not train anything. It loads the saved preprocessor (models/preprocessor.joblib) and the saved
network (models/dnn.keras) created by `python -m src.train`.
"""
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import json

import joblib
import keras
import pandas as pd
import streamlit as st

from src.data import CATEGORICAL, DESCRIPTIONS, FEATURES, GROUPS, load_data, split
from src.reasons import REASONS

THRESHOLD = 0.5  # probability of failing at or above 50% -> AT RISK
LEVEL_1_5 = {1: "1 · very low", 2: "2 · low", 3: "3 · medium", 4: "4 · high", 5: "5 · very high"}
CHOICES = {  # readable labels for the coded values in the dataset
    "school": {"GP": "Gabriel Pereira", "MS": "Mousinho da Silveira"}, "sex": {"F": "Female", "M": "Male"},
    "address": {"U": "Urban", "R": "Rural"}, "famsize": {"LE3": "3 or fewer", "GT3": "More than 3"},
    "Pstatus": {"T": "Living together", "A": "Apart"},
    "Medu": {0: "None", 1: "Primary (4th grade)", 2: "5th–9th grade", 3: "Secondary", 4: "Higher education"},
    "traveltime": {1: "< 15 min", 2: "15–30 min", 3: "30 min – 1 h", 4: "> 1 h"},
    "studytime": {1: "< 2 h / week", 2: "2–5 h / week", 3: "5–10 h / week", 4: "> 10 h / week"},
    "famrel": LEVEL_1_5, "freetime": LEVEL_1_5, "goout": LEVEL_1_5, "Dalc": LEVEL_1_5, "Walc": LEVEL_1_5,
    "health": {1: "1 · very bad", 2: "2 · bad", 3: "3 · fair", 4: "4 · good", 5: "5 · very good"},
}
CHOICES["Fedu"] = CHOICES["Medu"]
SLIDERS = {"age": (15, 22), "G1": (0, 20), "G2": (0, 20), "failures": (0, 3)}  # allowed ranges from the codebook

st.set_page_config(page_title="Student Performance Predictor", layout="wide")
st.markdown("""<style>
  .block-container {padding-top: 2rem; max-width: 1200px;}
  h1 {font-weight: 650; letter-spacing: -0.02em;}
  .badge {display: inline-block; padding: 0.35rem 0.8rem; border-radius: 6px; font-weight: 650; font-size: 1.6rem;}
  .pass {background: #dcfce7; color: #166534;}
  .risk {background: #fee2e2; color: #991b1b;}
  .muted {color: #71717a; font-size: 0.9rem;}
  @media (min-width: 900px) {div[data-testid="stColumn"]:has(.result-anchor) {position: sticky; top: 3.5rem; align-self: flex-start;}}
  @media (max-width: 899px) {div[data-testid="stHorizontalBlock"]:has(.result-anchor) {flex-direction: column-reverse;}}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    return joblib.load("models/preprocessor.joblib"), keras.models.load_model("models/dnn.keras")


@st.cache_data
def load_reference():
    train, _ = split(load_data())  # the same 519 training students the model learned from
    with open("results/results.json") as f:
        return train, json.load(f)


prep, net = load_models()
train, report = load_reference()
kept = report["kept_columns"]
# Which of the 32 inputs the model uses (a kept column like "school_MS" belongs to the input "school")
used = [c for c in FEATURES if any(k == c or k.startswith(c + "_") for k in kept)]
typical = {c: train[c].mode()[0] if c in CATEGORICAL else int(train[c].median()) for c in FEATURES}


def risk(students):
    """Probability of being at risk, using exactly the same preprocessing as in training."""
    X = prep.transform(pd.DataFrame(students)[FEATURES])
    return net.predict(X, verbose=0)[:, 0]


def field(name):
    label = DESCRIPTIONS[name] + (" · used by model" if name in used else "")
    why = ("Used by the model. " if name in used else "Not used by the model. ") + REASONS[name]
    if name in SLIDERS:
        return st.slider(label, *SLIDERS[name], typical[name], key=name, help=why)
    if name == "absences":
        return st.number_input(label, 0, 93, typical[name], key=name, help=why)
    options = sorted(train[name].unique())
    fmt = CHOICES.get(name, {}).get
    return st.selectbox(label, options, index=options.index(typical[name]), key=name, help=why,
                        format_func=lambda v: fmt(v) or str(v).replace("_", " ").capitalize())


st.title("Student Performance Predictor")
st.markdown('<p class="muted">A deep neural network predicts whether a Portuguese secondary-school student will '
            '<b>pass</b> (final grade G3 ≥ 10 out of 20) or is <b>at risk</b> of failing.</p>', unsafe_allow_html=True)

tab_predict, tab_results, tab_features, tab_about = st.tabs(
    ["1 · Predict", "2 · Results & graphs", "3 · Feature choices", "4 · About"])

# ---------------------------------------------------------------- 1. Predict
with tab_predict:
    form_col, result_col = st.columns([3, 2], gap="large")
    student = {}
    with form_col:
        boxes = [*st.columns(2), *st.columns(2)]
        for box, (group, cols) in zip(boxes, GROUPS.items()):
            with box.container(border=True):
                st.markdown(f"**{group}**")
                for c in cols:
                    student[c] = field(c)

    with result_col:
        st.markdown('<div class="result-anchor"></div>', unsafe_allow_html=True)
        p = float(risk([student])[0])
        with st.container(border=True):
            st.markdown("Prediction")
            st.markdown(f'<span class="badge {"risk" if p >= THRESHOLD else "pass"}">'
                        f'{"AT RISK" if p >= THRESHOLD else "PASS"}</span>', unsafe_allow_html=True)
            st.markdown(f"Chance of failing: **{p:.0%}**  \n<span class='muted'>AT RISK if 50% or more."
                        f"{' Close to 50%, so treat it as uncertain.' if abs(p - THRESHOLD) < 0.15 else ''}</span>",
                        unsafe_allow_html=True)
            st.progress(p)
            if student["absences"] > train["absences"].max():
                st.warning(f"No training student had more than {train['absences'].max()} absences, so this is less reliable.")

        with st.container(border=True):
            st.markdown("**What pushed this result**")
            # For each input the model uses: set it back to a typical value and see how much the chance changes.
            changed = [c for c in used if student[c] != typical[c]]
            effects = dict(zip(changed, p - risk([student | {c: typical[c]} for c in changed]))) if changed else {}
            shown = [(c, d) for c, d in sorted(effects.items(), key=lambda kv: -abs(kv[1])) if abs(d) >= 0.01][:4]
            if not shown:
                st.markdown("This student looks like a typical student on everything the model uses.")
            for c, d in shown:
                st.markdown(f"- **{DESCRIPTIONS[c]}** = {student[c]} (typical: {typical[c]}) "
                            f"{'raises' if d > 0 else 'lowers'} the chance of failing by {abs(d):.0%}")
            st.markdown("<span class='muted'>Each line sets one input back to a typical value and measures the change.</span>",
                        unsafe_allow_html=True)
        st.caption(f"The model uses {len(used)} of the 32 inputs: {', '.join(used)}.")

# ---------------------------------------------------------------- 2. Results & graphs
with tab_results:
    dnn = report["models"]["DNN (neural network)"]["test"]
    (tn, fp), (fn, tp) = report["confusion_matrix"]
    st.markdown("**Final scores on the 130 test students** (kept aside; the model never saw them while learning or choosing settings)")
    c = st.columns(4)
    c[0].metric("Accuracy", f"{dnn['accuracy']:.1%}", help="Share of all students predicted correctly")
    c[1].metric("Precision", f"{dnn['precision']:.1%}", help="Of the students we flagged AT RISK, how many really failed")
    c[2].metric("Recall", f"{dnn['recall']:.1%}", help="Of the students who really failed, how many we flagged")
    c[3].metric("F1 score", f"{dnn['f1']:.3f}", help="Balance of precision and recall")
    st.info("Only 15% of students fail, so a model that always says 'pass' already gets 84.6% accuracy. "
            "That is why we also look at recall and F1: they check that we actually find the failing students.")

    left, right = st.columns([2, 3])
    left.image("results/figures/confusion.png", width=340)
    right.markdown(f"""**Confusion matrix, in words**
- **{tn}** passing students correctly predicted PASS
- **{tp}** failing students correctly flagged AT RISK
- **{fp}** false alarms (flagged AT RISK but passed)
- **{fn}** missed (predicted PASS but failed)""")

    st.markdown("**DNN vs traditional machine learning** (same students, same inputs, same 5-fold cross-validation)")
    st.image("results/figures/comparison.png")
    rows = {name: {"CV accuracy": r["cv"]["accuracy"] if r["cv"] else None, "CV F1": r["cv"]["f1"] if r["cv"] else None,
                   "Test accuracy": r["test"]["accuracy"], "Test precision": r["test"]["precision"],
                   "Test recall": r["test"]["recall"], "Test F1": r["test"]["f1"]} for name, r in report["models"].items()}
    st.dataframe(pd.DataFrame(rows).T.style.format("{:.3f}", na_rep="–"), width="stretch")

    st.markdown("**Why these choices** (each tested with 5-fold cross-validation on the training students; green = chosen)")
    st.image("results/figures/choices.png")
    st.markdown("**How the network learned**")
    st.image("results/figures/learning_curve.png", width=620)
    s = report["settings"]
    st.markdown(f"""**The final network**
| Part | Choice | Why |
|---|---|---|
| Inputs | {len(kept)} columns | Feature selection (tab 3) |
| Hidden layers | {s['hidden'][0]} then {s['hidden'][1]} neurons | Best of the grid search |
| Activation | {s['activation']} | Best in the activation test (graph above) |
| Dropout | {s['dropout']} | Best of the grid search; stops the network memorising |
| Output | 1 neuron, sigmoid | Gives a probability between 0 and 1 |
| Loss | binary cross-entropy | Standard loss for yes/no predictions |
| Optimizer | {s['optimizer']}, learning rate {s['learning_rate']} | Best in the optimizer test and grid search |
| Early stopping | stop after 20 epochs without improvement | Trained {report['epochs_trained']} epochs, kept epoch {report['best_epoch']} |""")

# ---------------------------------------------------------------- 3. Feature choices
with tab_features:
    exp = report["experiments"]["features"]
    st.markdown(f"""**How we chose the inputs**
1. All **32** inputs were considered. Text columns become 0/1 columns, so there are 45 columns in total.
2. Each column gets an **information score**: how much it tells us about pass / at risk (mutual information), calculated on the training students only.
3. We kept the **top {len(kept)}**. We tested keeping 5, 10, 20 or all 45 columns with cross-validation, and {report['k']} gave the best F1:
""")
    st.table(pd.DataFrame({k: {"CV accuracy": f"{v['accuracy']:.3f}", "CV F1": f"{v['f1']:.3f}"} for k, v in exp.items()}).T
             .rename_axis("Columns kept"))
    st.image("results/figures/feature_scores.png", width=560)

    info = report["info_scores"]
    ranked = sorted(info, key=info.get, reverse=True)

    def evidence(f):
        cols = [c for c in info if c == f or c.startswith(f + "_")]
        best = max(cols, key=info.get)
        return info[best], ranked.index(best) + 1

    pick = st.selectbox("Ask about any input", FEATURES, index=FEATURES.index("address"),
                        format_func=lambda f: f"{DESCRIPTIONS[f]} ({f})")
    score, rank = evidence(pick)
    with st.container(border=True):
        st.markdown(f"**{DESCRIPTIONS[pick]}** is **{'USED' if pick in used else 'NOT USED'}** by the model.  \n{REASONS[pick]}")
        a, b = st.columns(2)
        a.metric("Information score", f"{score:.3f}")
        b.metric("Rank", f"{rank} of 45", help=f"The top {len(kept)} are kept")
        if train[pick].nunique() <= 6:
            rates = train.groupby(pick).at_risk.agg(["mean", "size"])
            st.table(pd.DataFrame({"At-risk rate": rates["mean"].map("{:.0%}".format), "Students": rates["size"]})
                     .rename(index=lambda v: CHOICES.get(pick, {}).get(v, v)))

    st.markdown("**All 32 inputs**")
    table = pd.DataFrame([{"Input": DESCRIPTIONS[f], "Used": "yes" if f in used else "no", "Rank": evidence(f)[1],
                           "Why": REASONS[f]} for f in FEATURES]).sort_values("Rank").set_index("Input")
    st.table(table)

# ---------------------------------------------------------------- 4. About
with tab_about:
    st.markdown("""
**Project.** Student Academic Performance Prediction using Deep Neural Networks · 24CAI0306 Deep Neural Networks ·
B.E. CSE-AIML, Semester 5 · Kavin Thakur.

**Data.** UCI Student Performance dataset (Portuguese course): 649 students from two schools in Portugal, 2005–06
(P. Cortez and A. Silva, 2008, CC BY 4.0). AT RISK means a final grade G3 below 10 out of 20.

**Method.** 80% of students for training, 20% kept aside for testing. Text columns one-hot encoded, numbers standardised,
top columns chosen by information score. A Keras neural network (Dense + Dropout layers, sigmoid output) chosen by
grid search with 5-fold cross-validation, and compared with logistic regression, a decision tree and a random forest.

**Use responsibly.** Trained on two schools in one year, and some inputs are sensitive (sex, family). Use it to start a
conversation with a student, never to decide something about them.

Code: github.com/auraCodesKM/student-performance-dnn
""")
