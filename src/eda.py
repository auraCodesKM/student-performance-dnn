"""Exploratory data analysis (EDA): checks the data and saves 5 graphs to results/figures/.

Run:  python -m src.eda
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import CATEGORICAL, NUMERIC, load_data

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

    # 2. G2 vs G3: the 2nd period grade almost decides the final grade
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(df.G2, df.G3, alpha=0.3, s=15, color="#2a78d6")
    zero = df[df.G3 == 0]
    ax.scatter(zero.G2, zero.G3, s=25, color="#eb6834", label="G3 = 0 (likely dropout)")
    ax.set(xlabel="G2 (second period)", ylabel="G3 (final)", title=f"G2 vs G3, r = {df.G2.corr(df.G3):.2f}")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{FIG}/g2_vs_g3.png", dpi=150); plt.close(fig)

    # 3. Spearman correlation of every numeric feature with G3 (rank-based, fine for ordinal scales)
    corr = df[NUMERIC].corrwith(df.G3, method="spearman").sort_values()
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


if __name__ == "__main__":
    data = load_data()
    audit(data)
    figures(data)
    print(f"Figures saved to {FIG}/")
