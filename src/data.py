"""Step 1: load the data, create the target (Pass / At risk), split it, and prepare the inputs for the models."""
import os
import warnings

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# The UCI file in this repo. (On Kaggle the notebook points this at Kaggle's copy of the same file.)
DATA_PATH = os.environ.get("STUDENT_DATA_PATH", "data/student-por.csv")
SEED = 42  # fixed random seed, so every run gives the same result

# Text columns (categories). They become 0/1 columns ("one-hot encoding").
CATEGORICAL = ["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian",
               "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"]
# Number columns. They are rescaled to a similar range ("standardisation").
NUMERIC = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel", "freetime", "goout",
           "Dalc", "Walc", "health", "absences", "G1", "G2"]
FEATURES = NUMERIC + CATEGORICAL  # all 32 inputs. G3 is NOT an input: it is what we predict.

GROUPS = {  # the four groups named in the project brief
    "Demographic": ["sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob", "guardian"],
    "Academic": ["school", "reason", "G1", "G2", "failures", "studytime", "traveltime", "schoolsup", "famsup", "paid",
                 "nursery", "higher"],
    "Social": ["famrel", "freetime", "goout", "romantic", "activities", "internet"],
    "Behavioural": ["absences", "Dalc", "Walc", "health"],
}
DESCRIPTIONS = {  # readable names (from the dataset's codebook, data/student.txt)
    "school": "School (GP or MS)", "sex": "Sex", "age": "Age (15-22)", "address": "Home address (urban / rural)",
    "famsize": "Family size (<=3 / >3)", "Pstatus": "Parents together / apart", "Medu": "Mother's education (0-4)",
    "Fedu": "Father's education (0-4)", "Mjob": "Mother's job", "Fjob": "Father's job",
    "reason": "Reason for choosing school", "guardian": "Guardian", "traveltime": "Travel time to school (1-4)",
    "studytime": "Weekly study time (1-4)", "failures": "Past class failures (0-3)", "schoolsup": "Extra school support",
    "famsup": "Family educational support", "paid": "Extra paid classes", "activities": "Extra-curricular activities",
    "nursery": "Attended nursery school", "higher": "Wants higher education", "internet": "Internet at home",
    "romantic": "In a relationship", "famrel": "Family relationship quality (1-5)",
    "freetime": "Free time after school (1-5)", "goout": "Going out with friends (1-5)",
    "Dalc": "Workday alcohol use (1-5)", "Walc": "Weekend alcohol use (1-5)", "health": "Current health (1-5)",
    "absences": "School absences", "G1": "1st period grade (0-20)", "G2": "2nd period grade (0-20)",
}


def load_data(path=DATA_PATH):
    """Read the CSV and add the target column: at_risk = 1 if the final grade G3 is below 10, else 0."""
    df = pd.read_csv(path, sep=None, engine="python")  # works with ";" (UCI file) or "," (Kaggle copy)
    df["at_risk"] = (df["G3"] < 10).astype(int)
    return df


def split(df):
    """80% of students for training, 20% kept aside as the test set. 'stratify' keeps 15% at-risk in both parts."""
    return train_test_split(df, test_size=0.2, stratify=df["at_risk"], random_state=SEED)


def information_score(X, y):
    """How much each column tells us about pass / at risk (mutual information). 0 = nothing."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mutual_info_classif(X, y, discrete_features=True)  # all our columns are whole numbers or categories


def make_preprocessor(k=10):
    """Turns the raw student table into the numbers the models use. Fitted on training students only.

    1. One-hot encode the text columns and standardise the number columns  -> 45 columns
    2. Keep the k columns with the highest information score                 -> k columns
    """
    encode = ColumnTransformer([
        ("text", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("numbers", StandardScaler(), NUMERIC),
    ], verbose_feature_names_out=False).set_output(transform="pandas")
    return Pipeline([("encode", encode), ("select", SelectKBest(information_score, k=k))])
