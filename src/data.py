"""Data loading, target definition, train/test split and preprocessing.

Everything that learns from data (scaler, one-hot encoder, feature selector)
lives inside a scikit-learn Pipeline, so it is fitted on training rows only.
That is the main guard against data leakage.
"""
import warnings

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

# All 32 input columns are candidates. G1 and G2 are known before G3 (after the 1st and 2nd grading
# periods), so using them is not leakage. G3 itself is the target and is never an input.
FEATURES = NUMERIC + GRADES + CATEGORICAL

# The brief groups the inputs as demographic, academic, social and behavioural. Used by the app form and the feature audit.
GROUPS = {
    "Demographic": ["sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob", "guardian"],
    "Academic": ["school", "reason", "G1", "G2", "failures", "studytime", "traveltime", "schoolsup", "famsup", "paid",
                 "nursery", "higher"],
    "Social": ["famrel", "freetime", "goout", "romantic", "activities", "internet"],
    "Behavioural": ["absences", "Dalc", "Walc", "health"],
}
# Short codebook descriptions (data/student.txt).
DESCRIPTIONS = {
    "school": "School (GP or MS)", "sex": "Sex", "age": "Age (15-22)", "address": "Home address (urban / rural)",
    "famsize": "Family size (<=3 / >3)", "Pstatus": "Parents together / apart", "Medu": "Mother's education (0-4)",
    "Fedu": "Father's education (0-4)", "Mjob": "Mother's job", "Fjob": "Father's job", "reason": "Reason for choosing school",
    "guardian": "Guardian", "traveltime": "Travel time to school (1-4)", "studytime": "Weekly study time (1-4)",
    "failures": "Past class failures (0-3)", "schoolsup": "Extra school support", "famsup": "Family educational support",
    "paid": "Extra paid classes", "activities": "Extra-curricular activities", "nursery": "Attended nursery school",
    "higher": "Wants higher education", "internet": "Internet at home", "romantic": "In a relationship",
    "famrel": "Family relationship quality (1-5)", "freetime": "Free time after school (1-5)",
    "goout": "Going out with friends (1-5)", "Dalc": "Workday alcohol use (1-5)", "Walc": "Weekend alcohol use (1-5)",
    "health": "Current health (1-5)", "absences": "School absences", "G1": "1st period grade (0-20)",
    "G2": "2nd period grade (0-20)",
}
assert sorted(sum(GROUPS.values(), [])) == sorted(FEATURES) == sorted(DESCRIPTIONS)  # every input is in exactly one group


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


def discrete_mi(X, y):
    """Mutual information between each input column and the target, treating every column as discrete.

    Every input here is an integer code, a 0/1 flag or a whole-number grade, so MI is computed from counts
    (a contingency table), not from the k-nearest-neighbour estimator meant for continuous data. Standardising
    does not change the distinct values, only relabels them, so MI is the same before and after scaling.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # sklearn warns that scaled codes "look continuous"; they are still discrete
        return mutual_info_classif(X, y, discrete_features=True)


def make_preprocessor(k="all"):
    """One-hot encode categoricals, standardise numerics, then optionally keep the k most informative features.

    Mutual information measures how much knowing a feature reduces uncertainty about the target;
    it catches non-linear relationships that correlation misses.
    """
    encode = ColumnTransformer([
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC + GRADES),
    ], verbose_feature_names_out=False).set_output(transform="pandas")  # keep column names so selected features are readable
    # Columns not listed (G3 and at_risk) are dropped by the transformer (remainder='drop'), so the target can't leak in.
    select = SelectKBest(discrete_mi, k=k)
    return [("encode", encode), ("select", select)]
