"""Streamlit app: enter a student's details and get the DNN's prediction, Pass or At risk.

Run:  streamlit run app.py
The app never trains anything. It loads models/dnn.joblib, the full fitted pipeline (encoder, scaler, feature
selector and the 5-network ensemble) saved by src/evaluate.py, so inference uses exactly the training preprocessing.
"""
import json

import joblib
import pandas as pd
import streamlit as st

import src.model  # noqa: F401  (the saved pipeline contains DNNClassifier, so its module must be importable)
from src.data import CATEGORICAL, DESCRIPTIONS, FEATURES, GROUPS, load_data, split

THRESHOLD = 0.5
LEVEL_1_5 = {1: "1 · very low", 2: "2 · low", 3: "3 · medium", 4: "4 · high", 5: "5 · very high"}
CHOICES = {  # readable labels for coded values (data/student.txt)
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
SLIDERS = {"age": (15, 22), "G1": (0, 20), "G2": (0, 20), "failures": (0, 3)}  # codebook ranges

st.set_page_config(page_title="Student Performance Predictor", layout="wide")
st.markdown("""<style>
  .block-container {padding-top: 2rem; max-width: 1200px;}
  h1 {font-weight: 650; letter-spacing: -0.02em;}
  .badge {display: inline-block; padding: 0.35rem 0.8rem; border-radius: 6px; font-weight: 650; font-size: 1.6rem; letter-spacing: 0.02em;}
  .pass {background: #dcfce7; color: #166534;}
  .risk {background: #fee2e2; color: #991b1b;}
  .muted {color: #71717a; font-size: 0.9rem;}
  /* keep the result panel in view while scrolling the form (desktop widths only) */
  @media (min-width: 900px) {
    div[data-testid="stColumn"]:has(.result-anchor) {position: sticky; top: 3.5rem; align-self: flex-start;}
  }
  /* on phones the columns stack: show the result above the long form */
  @media (max-width: 899px) {
    div[data-testid="stHorizontalBlock"]:has(.result-anchor) {flex-direction: column-reverse;}
  }
</style>""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load("models/dnn.joblib")


@st.cache_data
def load_reference():
    train, _ = split(load_data())  # same 519 training students the model learned from
    with open("results/metrics.json") as f:
        metrics = json.load(f)
    return train, metrics


model = load_model()
train, metrics = load_reference()
dnn = metrics["dnn"]
# Map the encoded columns the model kept (e.g. "school_MS") back to the raw inputs they came from ("school").
used = [c for c in FEATURES if any(n == c or n.startswith(c + "_") for n in dnn["selected_features"])]
typical = {c: train[c].mode()[0] if c in CATEGORICAL else int(train[c].median()) for c in FEATURES}


def field(container, name):
    """One input widget, pre-filled with the most typical value among training students."""
    label = DESCRIPTIONS[name] + (" ·  used by model" if name in used else "")
    if name in SLIDERS:
        return container.slider(label, *SLIDERS[name], typical[name], key=name)
    if name == "absences":
        return container.number_input(label, 0, 93, typical[name], key=name)  # codebook allows 0-93
    options = sorted(train[name].unique())
    fmt = CHOICES.get(name, {}).get
    return container.selectbox(label, options, index=options.index(typical[name]), key=name,
                               format_func=lambda v: fmt(v) or str(v).replace("_", " ").capitalize())


def risk(rows):
    """P(at risk) for one or more students; columns are put in the exact training order."""
    return model.predict_proba(pd.DataFrame(rows)[FEATURES])[:, 1]


st.title("Student Performance Predictor")
st.markdown('<p class="muted">Predicts whether a Portuguese secondary-school student will pass the year '
            '(final grade G3 ≥ 10 / 20) or is at risk of failing, from demographic, academic, social and behavioural data.</p>',
            unsafe_allow_html=True)

predict_tab, model_tab, about_tab = st.tabs(["Predict", "Model performance", "About"])

with predict_tab:
    form_col, result_col = st.columns([3, 2], gap="large")
    student = {}
    with form_col:
        top, bottom = st.columns(2), st.columns(2)
        for box, (group, cols) in zip([*top, *bottom], GROUPS.items()):
            with box.container(border=True):
                st.markdown(f"**{group}**")
                for c in cols:
                    student[c] = field(st, c)

    with result_col:
        st.markdown('<div class="result-anchor"></div>', unsafe_allow_html=True)
        with st.container(border=True):
            p = float(risk([student])[0])
            at_risk = p >= THRESHOLD
            st.markdown("Prediction")
            st.markdown(f'<span class="badge {"risk" if at_risk else "pass"}">{"AT RISK" if at_risk else "PASS"}</span>',
                        unsafe_allow_html=True)
            st.markdown(f"Estimated probability of failing: **{p:.0%}**  \n"
                        f'<span class="muted">Decision cut-off {THRESHOLD:.0%}. '
                        f"{'Close to the cut-off: treat as uncertain.' if abs(p - THRESHOLD) < 0.15 else ''}</span>",
                        unsafe_allow_html=True)
            st.progress(p)
            if student["absences"] > train["absences"].max():
                st.warning(f"Absences above {train['absences'].max()} were never seen in training, so this estimate is less reliable.")

        with st.container(border=True):
            st.markdown("**Why this result**")
            # Occlusion: put each input the model uses back to a typical value and see how the risk changes.
            changed = [c for c in used if student[c] != typical[c]]
            effects = dict(zip(changed, p - risk([student | {c: typical[c]} for c in changed]))) if changed else {}
            shown = sorted(effects.items(), key=lambda kv: -abs(kv[1]))[:4]
            if not shown:
                st.markdown("Every input the model relies on is at the typical training value.")
            for c, delta in shown:
                arrow = "raises" if delta > 0 else "lowers"
                st.markdown(f"- **{DESCRIPTIONS[c]}** = {student[c]} (typical {typical[c]}) {arrow} the risk by {abs(delta):.0%}")
            st.markdown('<span class="muted">Each line replaces one input with the typical training value and measures how much '
                        'the predicted risk moves. Grades G1 and G2 usually dominate.</span>', unsafe_allow_html=True)

        st.caption(f"The model uses {len(used)} of the 32 inputs: {', '.join(used)}. The others are shown for completeness.")

with model_tab:
    r = metrics["models"]["DNN (MLP)"]
    (tn, fp), (fn, tp) = dnn["confusion_matrix"]
    lo, hi = dnn["wilson_95ci_accuracy"]
    st.markdown("Scores on **130 locked test students** the model never saw during training, tuning or model selection.")
    k = st.columns(5)
    k[0].metric("Accuracy", f"{r['test_accuracy']:.1%}", help=f"95% interval {lo:.1%}–{hi:.1%}")
    k[1].metric("Precision (at risk)", f"{r['test_precision_at_risk']:.1%}", help="Of students flagged at risk, share who really failed")
    k[2].metric("Recall (at risk)", f"{r['test_recall_at_risk']:.1%}", help="Of students who really failed, share the model flagged")
    k[3].metric("Macro-F1", f"{r['test_f1_macro']:.3f}")
    k[4].metric("ROC-AUC", f"{r['test_roc_auc']:.3f}")
    st.markdown(f"Confusion matrix: **{tp}** at-risk students caught, **{fn}** missed, **{fp}** false alarms, **{tn}** passes correctly predicted. "
                f"Over 5 retrained copies the test accuracy is {dnn['seed_mean']['accuracy']:.1%} ± {dnn['seed_std']['accuracy']:.1%}. "
                "Predicting 'pass' for everyone would already score 84.6%, so recall and macro-F1 matter more than accuracy.")
    left, right = st.columns(2)
    left.image("results/figures/confusion.png", width=380)
    right.image("results/figures/roc.png", width=420)

    st.markdown("**Comparison with traditional machine learning** (same preprocessing, same cross-validation)")
    names = {"cv_accuracy": "CV accuracy", "cv_f1_macro": "CV macro-F1", "test_accuracy": "Test accuracy",
             "test_recall_at_risk": "Test recall", "test_f1_macro": "Test macro-F1", "test_roc_auc": "Test ROC-AUC"}
    st.dataframe(pd.DataFrame(metrics["models"]).T[list(names)].rename(columns=names).style.format("{:.3f}"),
                 width="stretch")
    st.image("results/figures/comparison.png")
    st.caption("McNemar tests on the 130 test students find no significant difference between the DNN and the rule, "
               "logistic regression or XGBoost (p > 0.7): the DNN is as good as the best traditional models, not clearly better.")
    st.markdown("**Architecture**  \n"
                f"{len(dnn['selected_features'])} inputs → " + " → ".join(str(u) for u in model.named_steps["model"].hidden)
                + " → 1 output. Each hidden layer: Linear → BatchNorm → ReLU → Dropout. "
                "He initialisation, binary cross-entropy, AdamW, early stopping, average of 5 networks. Tuned with Optuna.")

with about_tab:
    st.markdown("""
**Project.** Student Academic Performance Prediction using Deep Neural Networks. Course 24CAI0306 Deep Neural Networks,
B.E. CSE-AIML, Semester 5. Author: Kavin Thakur.

**Data.** UCI Student Performance dataset, Portuguese course: 649 students from two secondary schools in Portugal, 2005–06
(P. Cortez and A. Silva, 2008; CC BY 4.0). Target: at risk if the final grade G3 is below 10 out of 20.

**Method.** Stratified 80/20 split with the test set locked. One-hot encoding and standardisation, mutual-information
feature selection and a PyTorch multilayer perceptron, all fitted inside one scikit-learn pipeline on training data only.
Hyperparameters tuned with Optuna by cross-validation, then compared with a naive rule, logistic regression,
random forest and XGBoost.

**Use responsibly.** The model was trained on two schools in one year and includes sensitive attributes. It is a prompt for
a conversation with a student, not a decision about them. Source code and documentation: github.com/auraCodesKM/student-performance-dnn.
""")
