"""Streamlit app: enter a student's details, get the DNN's predicted performance category.

Run:  streamlit run app.py
"""
import json

import joblib
import pandas as pd
import streamlit as st

import src.model  # noqa: F401  (the saved pipelines contain DNNClassifier, so its module must be importable)
from src.data import DATA_PATH, load_data

SCENARIOS = {
    "Early warning: start of year, no grades yet": "early",
    "Late term: after the 2nd grading period (uses G1, G2)": "late",
}
LEVEL_1_5 = {1: "1 - very low", 2: "2 - low", 3: "3 - medium", 4: "4 - high", 5: "5 - very high"}
EDU = {0: "None", 1: "Primary (4th grade)", 2: "5th-9th grade", 3: "Secondary", 4: "Higher education"}
TRAVEL = {1: "< 15 min", 2: "15-30 min", 3: "30 min - 1 h", 4: "> 1 h"}
STUDY = {1: "< 2 h", 2: "2-5 h", 3: "5-10 h", 4: "> 10 h"}


@st.cache_resource
def load(scenario):
    return joblib.load(f"models/dnn_{scenario}.joblib")


@st.cache_data
def defaults():
    df = load_data(DATA_PATH)
    return df, json.load(open("results/metrics.json"))


st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")
st.title("Student Performance Predictor")
st.caption("Deep neural network (PyTorch MLP) trained on 649 Portuguese secondary-school students (UCI Student Performance). "
           "Predicts whether the final grade G3 will be below 10/20 (at risk) or not.")

df, metrics = defaults()
label = st.radio("When are you predicting?", list(SCENARIOS), horizontal=True)
scenario = SCENARIOS[label]


def pick(col, name, options=None, fmt=None):
    """Dropdown pre-filled with the most common value in the dataset."""
    options = options or sorted(df[name].unique())
    mode = df[name].mode()[0]
    return col.selectbox(name, options, index=list(options).index(mode), format_func=fmt or str, key=name)


with st.form("student"):
    a, b, c, d = st.columns(4)
    a.subheader("Student")
    row = {
        "school": pick(a, "school", fmt={"GP": "Gabriel Pereira", "MS": "Mousinho da Silveira"}.get),
        "sex": pick(a, "sex"),
        "age": a.slider("age", 15, 22, 17),
        "address": pick(a, "address", fmt={"U": "Urban", "R": "Rural"}.get),
        "higher": pick(a, "higher"),
        "romantic": pick(a, "romantic"),
        "health": pick(a, "health", list(LEVEL_1_5), LEVEL_1_5.get),
    }
    b.subheader("Family")
    row |= {
        "famsize": pick(b, "famsize", fmt={"LE3": "3 or fewer", "GT3": "more than 3"}.get),
        "Pstatus": pick(b, "Pstatus", fmt={"T": "Living together", "A": "Apart"}.get),
        "Medu": pick(b, "Medu", list(EDU), EDU.get),
        "Fedu": pick(b, "Fedu", list(EDU), EDU.get),
        "Mjob": pick(b, "Mjob"),
        "Fjob": pick(b, "Fjob"),
        "guardian": pick(b, "guardian"),
        "famrel": pick(b, "famrel", list(LEVEL_1_5), LEVEL_1_5.get),
        "famsup": pick(b, "famsup"),
    }
    c.subheader("School")
    row |= {
        "reason": pick(c, "reason"),
        "traveltime": pick(c, "traveltime", list(TRAVEL), TRAVEL.get),
        "studytime": pick(c, "studytime", list(STUDY), STUDY.get),
        "failures": c.slider("failures (past classes failed)", 0, 3, 0),
        "absences": c.number_input("absences", 0, 93, 2),
        "schoolsup": pick(c, "schoolsup"),
        "paid": pick(c, "paid"),
        "activities": pick(c, "activities"),
        "nursery": pick(c, "nursery"),
    }
    d.subheader("Lifestyle")
    row |= {
        "internet": pick(d, "internet"),
        "freetime": pick(d, "freetime", list(LEVEL_1_5), LEVEL_1_5.get),
        "goout": pick(d, "goout", list(LEVEL_1_5), LEVEL_1_5.get),
        "Dalc": pick(d, "Dalc", list(LEVEL_1_5), LEVEL_1_5.get),
        "Walc": pick(d, "Walc", list(LEVEL_1_5), LEVEL_1_5.get),
    }
    if scenario == "late":
        d.subheader("Grades so far")
        row |= {"G1": d.slider("G1 (1st period, 0-20)", 0, 20, 11), "G2": d.slider("G2 (2nd period, 0-20)", 0, 20, 11)}
    submitted = st.form_submit_button("Predict", type="primary")

if submitted:
    p = load(scenario).predict_proba(pd.DataFrame([row]))[0, 1]
    left, right = st.columns([1, 2])
    if p >= 0.5:
        left.error(f"### At risk\nLikely final grade below 10/20")
    else:
        left.success(f"### On track\nLikely final grade 10/20 or above")
    left.metric("Probability of being at risk", f"{p:.0%}")
    right.progress(float(p), text="Risk score (0% = safe, 100% = certain fail). The cut-off is 50%.")

rows, extra = metrics[scenario]
m = rows["DNN (MLP)"]
with st.expander("How reliable is this model?"):
    st.write(f"Scores on 130 held-out students the model never saw during training or tuning:")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Accuracy", f"{m['test_accuracy']:.1%}")
    k2.metric("Macro-F1", f"{m['test_f1_macro']:.3f}")
    k3.metric("At-risk recall", f"{m['test_recall_at_risk']:.0%}", help="Share of truly at-risk students the model flags")
    k4.metric("ROC-AUC", f"{m['test_roc_auc']:.3f}")
    st.caption(f"Always predicting 'pass' would already score 84.6% accuracy, so judge the model on macro-F1 and recall. "
               f"Features used by this model: {', '.join(extra['selected_features'])}.")
    if scenario == "early":
        st.warning("Without any grades the signal is weak. Treat this as a prompt for a conversation, not a verdict.")
