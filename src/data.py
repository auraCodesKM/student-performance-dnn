"""Data loading, target definition, train/test split and preprocessing.

Everything that learns from data (scaler, one-hot encoder, feature selector)
lives inside a scikit-learn Pipeline, so it is fitted on training rows only.
That is the main guard against data leakage.
"""
import os
import warnings

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Portuguese course, 649 students. The repo copy is the original UCI file; on Kaggle the notebook points
# STUDENT_DATA_PATH at the attached Kaggle mirror of the same data instead.
DATA_PATH = os.environ.get("STUDENT_DATA_PATH", "data/student-por.csv")
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
    df = pd.read_csv(path, sep=None, engine="python")  # detect the separator: ";" in the UCI file, "," in the Kaggle mirror
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

# Why each input is or is not used by the final model. Facts are from the 519 training students. The selection rule
# (rank by mutual information, keep the top k = 10, k tuned by Optuna) was fixed before these reasons were written;
# the notebook (section 5) and the app's "Feature choices" tab add the add-back / remove test for every input.
REASONS = {
    # ---- used (top 10 by mutual information) ----
    "G2": "Strongest signal (rank 1 of 45). Removing it costs the most: CV macro-F1 drops by about 0.11.",
    "G1": "Second strongest (rank 2). Students with G1 of 8 or less fail 81% of the time, with G1 of 11 or more almost never.",
    "failures": "Rank 3. No past failures: 9% at risk; one failure: 45%; two or more: 57-64%.",
    "school": "Rank 4. Mousinho da Silveira (MS) students are at risk 31% of the time vs 7% at Gabriel Pereira (GP).",
    "higher": "Rank 5. Students who don't plan higher education are at risk 46% of the time vs 12% for those who do.",
    "absences": "Rank 6. More than 10 absences: 24% at risk vs 14% with none. Disengagement adds to what grades show.",
    "studytime": "Rank 7. Under 2 h/week: 26% at risk; over 10 h/week: 4%.",
    "Fedu": "Rank 8. Father with only primary education: 27% at risk vs 10% with higher education.",
    "Medu": "Rank 9. Mother with only primary education: 27% at risk vs 8% with higher education. Just inside the top 10.",
    "famrel": "Rank 10, the last slot. Very bad family relations: 42% at risk (19 students) vs 10% for good. A weak signal.",
    # ---- not used ----
    "goout": "Rank 11, just below the cut-off. U-shaped: going out very little (24%) or a lot (26%) is riskier than the middle levels (about 11%). Its information ranks just below the top 10.",
    "age": "Rank 12. Older students are at risk more mainly because they repeated years, which 'failures' already captures (students aged 19+ have about 1-3 past failures).",
    "freetime": "Rank 13. Only the 'very high free time' group stands out (29% vs 9-16%). Weak once grades are known.",
    "Walc": "Rank 14. Weekend drinking rises from 12% to 25% at risk, but the effect is small next to grades and overlaps workday drinking (r = 0.62).",
    "reason": "Rank 15. 'Other' reason 27% vs 'reputation' 8%, but each category is small and the effect mostly reflects which school.",
    "Mjob": "Rank 16. Mothers who are teachers: 5% at risk; this overlaps mother's education, which is already used.",
    "Dalc": "Rank 17. Workday drinking is rare (only 15 students at level 5) and overlaps weekend drinking.",
    "address": "Rank 19. Rural 21% vs urban 13% at risk, but rural students have lower G1/G2 and 59% attend MS (school is used). With the same G2, rural and urban students fail at almost the same rate, so grades and school already carry this.",
    "traveltime": "Rank 22. Long travel is linked to rural address and school, which are already covered.",
    "Fjob": "Rank 23. Fathers who are teachers: 4% at risk (only 26 students); overlaps father's education, which is used.",
    "guardian": "Rank 24. Father 10%, mother 17%, other 21%: small differences, small groups.",
    "romantic": "Rank 25. In a relationship 19% vs 13%: a weak link that disappears once grades are known.",
    "internet": "Rank 26. No internet 21% vs 14%, but 'no internet' is far more common among rural students (34% vs 19%), so it largely repeats address/school.",
    "activities": "Rank 27. Extra-curricular activities 13% vs 18%: weak.",
    "sex": "Rank 28. Male 19% vs female 13%: weak. Not using it is also fairer, but it was dropped on evidence, not by hand.",
    "health": "Rank 29. No consistent pattern (very bad 19%, good 12%, very good 17%).",
    "famsup": "Rank 33. Family educational support 14% vs 17%: almost no difference.",
    "famsize": "Rank 34. Families of 3 or fewer 13% vs larger 16%: almost no difference.",
    "schoolsup": "Rank 36. Extra school support 13% vs 16%. Support is given to weaker students, so it is not a clean signal.",
    "nursery": "Rank 38. Attended nursery 16% vs not 14%: no real difference.",
    "Pstatus": "Rank 39. Parents apart 17% vs together 15%: no real difference.",
    "paid": "Rank 41 (lowest). Paid classes 14% vs none 16%, and only 28 students took them.",
}
assert sorted(REASONS) == sorted(FEATURES)  # a reason for every input
