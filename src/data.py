"""Data loading, target definition, train/test split and preprocessing.

Everything that learns from data (scaler, one-hot encoder, feature selector)
lives inside a scikit-learn Pipeline, so it is fitted on training rows only.
That is the main guard against data leakage.
"""
from functools import partial

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "data/student-por.csv"  # Portuguese course: 649 students
SEED = 42

# Nominal columns have no natural order, so they are one-hot encoded.
# Binary yes/no columns are also handled by OneHotEncoder(drop="if_binary"), which turns them into a single 0/1 column.
CATEGORICAL = [
    "school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian",
    "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic",
]
# Ordinal scales (1-5, 0-4) and counts keep their order, so they stay numeric and are standardised.
NUMERIC = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",
]
GRADES = ["G1", "G2"]

# Two prediction scenarios:
#   late  = after the 2nd grading period, G1 and G2 are known
#   early = start of the year, no grades yet (early-warning system)
FEATURES = {"late": NUMERIC + GRADES + CATEGORICAL, "early": NUMERIC + CATEGORICAL}


def load_data(path=DATA_PATH):
    """Read the CSV and add the target: at_risk = 1 if the final grade G3 < 10 (fail)."""
    df = pd.read_csv(path, sep=";")
    # Failing is the positive class because those are the students we want to find,
    # so precision/recall/F1 describe how well the model detects at-risk students.
    df["at_risk"] = (df["G3"] < 10).astype(int)
    return df


def split(df):
    """Stratified 80/20 split. The 20% test set is locked and only scored once at the end."""
    return train_test_split(df, test_size=0.2, stratify=df["at_risk"], random_state=SEED)


def make_preprocessor(scenario, k="all"):
    """One-hot encode categoricals, standardise numerics, then optionally keep the k most informative features.

    Mutual information measures how much knowing a feature reduces uncertainty about the target;
    it catches non-linear relationships that correlation misses.
    """
    numeric = [c for c in FEATURES[scenario] if c not in CATEGORICAL]
    encode = ColumnTransformer([
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("num", StandardScaler(), numeric),
    ], verbose_feature_names_out=False).set_output(transform="pandas")  # keep column names so selected features are readable
    # Columns not listed (G3, at_risk, and G1/G2 in the early scenario) are dropped by the transformer.
    select = SelectKBest(partial(mutual_info_classif, random_state=SEED), k=k)
    return [("encode", encode), ("select", select)]
