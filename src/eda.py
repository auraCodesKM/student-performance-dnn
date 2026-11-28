"""Exploratory data analysis: prints a data audit and saves figures to results/figures/.

Run:  python -m src.eda
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

from src.data import (CATEGORICAL, DESCRIPTIONS, FEATURES, GRADES, GROUPS, NUMERIC, discrete_mi, load_data,
                      make_preprocessor, split)

FIG = "results/figures"


def audit(df):
    print("Shape:", df.shape)
    print("Missing values:", int(df.isna().sum().sum()))
    print("Duplicate rows:", int(df.duplicated().sum()))
    print("At-risk (G3 < 10) share:", f"{df.at_risk.mean():.1%}", "-> majority-class accuracy would be", f"{1 - df.at_risk.mean():.1%}")
    zero = df[df.G3 == 0]
    print(f"G3 == 0: {len(zero)} students; with zero absences: {(zero.absences == 0).sum()} "
          f"(vs {(df.absences == 0).mean():.1%} of all students) -> likely dropped out / missed the final exam")
    print("Correlation with G3:", df[["G1", "G2"]].corrwith(df.G3).round(3).to_dict())
    print("'Pass if G2 >= 10' rule accuracy:", f"{((df.G2 < 10) == df.at_risk).mean():.1%}")
    print("Absences skew:", round(df.absences.skew(), 2), "max:", df.absences.max())


def figures(df):
    # 1. Target: G3 distribution with pass threshold
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(*zip(*df.G3.value_counts().sort_index().items()), color=["#eb6834" if g < 10 else "#2a78d6" for g in sorted(df.G3.unique())])
    ax.axvline(9.5, color="grey", ls="--")
    ax.set(xlabel="Final grade G3 (0-20)", ylabel="Students", title="G3 distribution (orange = at risk, G3 < 10)")
    fig.tight_layout(); fig.savefig(f"{FIG}/g3_distribution.png", dpi=150); plt.close(fig)

    # 2. G2 vs G3: why G2 dominates the late-term model
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(df.G2, df.G3, alpha=0.3, s=15, color="#2a78d6")
    zero = df[df.G3 == 0]
    ax.scatter(zero.G2, zero.G3, s=25, color="#eb6834", label="G3 = 0 (likely dropout)")
    ax.set(xlabel="G2 (second period)", ylabel="G3 (final)", title=f"G2 vs G3, r = {df.G2.corr(df.G3):.2f}")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{FIG}/g2_vs_g3.png", dpi=150); plt.close(fig)

    # 3. Spearman correlation of every numeric feature with G3 (rank-based, fine for ordinal scales)
    corr = df[NUMERIC + ["G1", "G2"]].corrwith(df.G3, method="spearman").sort_values()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(corr.index, corr.values, color=["#eb6834" if v < 0 else "#2a78d6" for v in corr.values])
    ax.set(xlabel="Spearman correlation with G3", title="Numeric features vs final grade")
    fig.tight_layout(); fig.savefig(f"{FIG}/correlations.png", dpi=150); plt.close(fig)

    # 4. At-risk rate for each categorical feature (largest gaps first)
    gaps = {c: df.groupby(c).at_risk.mean() for c in CATEGORICAL}
    order = sorted(gaps, key=lambda c: gaps[c].max() - gaps[c].min(), reverse=True)[:8]
    fig, axes = plt.subplots(2, 4, figsize=(12, 5), sharey=True)
    for ax, c in zip(axes.flat, order):
        ax.bar(gaps[c].index.astype(str), gaps[c].values * 100, color="#2a78d6")
        ax.set_title(c); ax.tick_params(axis="x", rotation=45)
    axes[0, 0].set_ylabel("% at risk"); axes[1, 0].set_ylabel("% at risk")
    fig.suptitle("At-risk rate by category (8 largest gaps)"); fig.tight_layout()
    fig.savefig(f"{FIG}/risk_by_category.png", dpi=150); plt.close(fig)

    # 5. Absences: heavy right tail, but real students, so kept
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(df.absences, bins=32, color="#2a78d6")
    ax.set(xlabel="Absences", ylabel="Students", title=f"Absences (skew {df.absences.skew():.1f})")
    fig.tight_layout(); fig.savefig(f"{FIG}/absences.png", dpi=150); plt.close(fig)


def feature_audit(train, k=10):
    """One row per input column: what it is, how it is encoded, how strongly it relates to G3, and how often
    SelectKBest keeps it across the same 15 CV folds used in evaluation. Uses training rows only."""
    group = {c: g for g, cols in GROUPS.items() for c in cols}
    counts = pd.Series(0, index=FEATURES)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=7)
    for fold, _ in cv.split(train, train.at_risk):
        part = train.iloc[fold]
        kept = Pipeline(make_preprocessor(k)).fit(part, part.at_risk).named_steps["select"].get_feature_names_out()
        for c in FEATURES:  # a one-hot column such as "Mjob_teacher" counts for its original column "Mjob"
            counts[c] += any(name == c or name.startswith(c + "_") for name in kept)
    final = Pipeline(make_preprocessor(k)).fit(train, train.at_risk).named_steps["select"].get_feature_names_out()
    # Mutual information of every encoded column on the full training set, ranked 1 (best) to 45.
    encoded = Pipeline(make_preprocessor()).fit(train, train.at_risk).named_steps["encode"].transform(train)
    mi = pd.Series(discrete_mi(encoded, train.at_risk), index=encoded.columns)
    rank = mi.rank(ascending=False, method="min").astype(int)
    rows = []
    for c in FEATURES:
        numeric = c in NUMERIC + GRADES
        rho = train[c].corr(train.G3, method="spearman") if numeric else np.nan
        rates = train.groupby(c).at_risk.mean()  # for categoricals: gap between the riskiest and safest category
        best = max((n for n in mi.index if n == c or n.startswith(c + "_")), key=mi.get)  # a one-hot feature's best column
        rows.append({"feature": c, "description": DESCRIPTIONS[c], "group": group[c],
                     "type": "grade" if c in GRADES else "numeric/ordinal" if numeric else "categorical",
                     "encoding": "standardised" if numeric else ("one 0/1 column" if train[c].nunique() == 2 else f"one-hot ({train[c].nunique()} columns)"),
                     "spearman_with_G3": rho, "at_risk_rate_gap": np.nan if numeric else rates.max() - rates.min(),
                     "mutual_information": mi[best], "mi_rank_of_45": rank[best], "kept_in_folds": f"{counts[c]}/15", "in_final_model": any(n == c or n.startswith(c + "_") for n in final)})
    return pd.DataFrame(rows).sort_values("kept_in_folds", key=lambda s: s.str.split("/").str[0].astype(int), ascending=False)


def vif(train):
    """Variance inflation factor of each numeric input: 1 / (1 - R^2) when that column is regressed on the others.
    The diagonal of the inverse correlation matrix gives all of them at once. Above ~5 means strong multicollinearity."""
    corr = train[NUMERIC + GRADES].corr().to_numpy()
    return pd.Series(np.diag(np.linalg.inv(corr)), index=NUMERIC + GRADES).sort_values(ascending=False)


if __name__ == "__main__":
    data = load_data()
    audit(data)
    figures(data)
    feature_audit(split(data)[0]).to_csv("results/feature_audit.csv", index=False)  # read by the app's Feature choices tab
    print(f"Figures saved to {FIG}/")
