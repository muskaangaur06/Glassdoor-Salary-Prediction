"""
All EDA visualisations for the Glassdoor Salary project.
Each function follows the UBM rule (Univariate / Bivariate / Multivariate).

Every plot is saved to `outputs/` and optionally displayed inline
(set SHOW_PLOTS=True when running in a notebook).
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# Global style

PALETTE   = "Set2"
FIG_SIZE  = (12, 6)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
SHOW_PLOTS = False   # set True in notebook


def _save(fig: plt.Figure, name: str) -> None:
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# UNIVARIATE ANALYSIS

def plot_salary_distribution(df: pd.DataFrame) -> None:
    """
    Chart 1 — Univariate
    Histogram + KDE of avg_salary to understand the overall
    salary distribution across all tech job postings.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Chart 1: Distribution of Average Salary (in $K)", fontsize=14, fontweight="bold")

    sal = df["avg_salary"].dropna()

    # Histogram
    axes[0].hist(sal, bins=30, color="#4C72B0", edgecolor="white", alpha=0.85)
    axes[0].axvline(sal.mean(),   color="red",    linestyle="--", label=f"Mean: ${sal.mean():.1f}K")
    axes[0].axvline(sal.median(), color="orange", linestyle="--", label=f"Median: ${sal.median():.1f}K")
    axes[0].set_xlabel("Average Salary ($K)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Histogram")
    axes[0].legend()

    # KDE
    sal.plot.kde(ax=axes[1], color="#4C72B0", linewidth=2)
    axes[1].fill_between(
        axes[1].lines[0].get_xdata(),
        axes[1].lines[0].get_ydata(),
        alpha=0.25, color="#4C72B0"
    )
    axes[1].set_xlabel("Average Salary ($K)")
    axes[1].set_title("Kernel Density Estimate")

    plt.tight_layout()
    _save(fig, "01_salary_distribution")


def plot_job_title_distribution(df: pd.DataFrame, top_n: int = 15) -> None:
    """
    Chart 2 — Univariate
    Frequency of top job categories to understand which roles
    dominate the tech job market.
    """
    counts = df["job_category"].value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    bars = ax.barh(counts.index[::-1], counts.values[::-1],
                   color=sns.color_palette(PALETTE, len(counts)))
    ax.bar_label(bars, padding=4)
    ax.set_xlabel("Number of Job Postings")
    ax.set_title("Chart 2: Top Job Categories by Posting Count", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "02_job_title_distribution")


def plot_seniority_distribution(df: pd.DataFrame) -> None:
    """
    Chart 3 — Univariate
    Pie chart of seniority levels to reveal what proportion of
    postings are junior, mid, senior, or leadership roles.
    """
    counts = df["seniority_level"].value_counts()
    colors = sns.color_palette(PALETTE, len(counts))

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        counts, labels=counts.index, autopct="%1.1f%%",
        colors=colors, startangle=140, pctdistance=0.82
    )
    for at in autotexts:
        at.set_fontsize(10)
    ax.set_title("Chart 3: Distribution of Seniority Levels", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "03_seniority_distribution")


def plot_sector_distribution(df: pd.DataFrame, top_n: int = 12) -> None:
    """
    Chart 4 — Univariate
    Bar chart of top industry sectors to show which sectors
    are hiring the most data professionals.
    """
    counts = df["Sector"].value_counts().dropna().head(top_n)

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    sns.barplot(x=counts.values, y=counts.index, palette=PALETTE, ax=ax)
    ax.set_xlabel("Number of Postings")
    ax.set_title("Chart 4: Top Industry Sectors Hiring Data Professionals",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "04_sector_distribution")


# BIVARIATE ANALYSIS

def plot_salary_by_job_category(df: pd.DataFrame) -> None:
    """
    Chart 5 — Bivariate (Categorical × Numerical)
    Box plot of avg_salary per job_category to compare salary
    ranges across different tech roles.
    """
    order = (df.groupby("job_category")["avg_salary"]
               .median().sort_values(ascending=False).index)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=df, x="job_category", y="avg_salary",
                order=order, palette=PALETTE, ax=ax)
    ax.set_xlabel("Job Category")
    ax.set_ylabel("Average Salary ($K)")
    ax.set_title("Chart 5: Salary Distribution by Job Category",
                 fontsize=13, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    _save(fig, "05_salary_by_job_category")


def plot_salary_by_state(df: pd.DataFrame, top_n: int = 15) -> None:
    """
    Chart 6 — Bivariate (Categorical × Numerical)
    Bar chart of median salary by state (top states) to highlight
    geographic salary disparities.
    """
    state_sal = (df.groupby("job_state")["avg_salary"]
                   .median().dropna().sort_values(ascending=False).head(top_n))

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    bars = ax.bar(state_sal.index, state_sal.values,
                  color=sns.color_palette(PALETTE, len(state_sal)))
    ax.bar_label(bars, fmt="$%.0fK", padding=3, fontsize=9)
    ax.set_xlabel("State")
    ax.set_ylabel("Median Salary ($K)")
    ax.set_title(f"Chart 6: Median Salary by State (Top {top_n})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "06_salary_by_state")


def plot_salary_by_company_size(df: pd.DataFrame) -> None:
    """
    Chart 7 — Bivariate (Categorical × Numerical)
    Box plot of salary across company size bands to test whether
    larger companies pay more.
    """
    size_order = [
        "1 to 50 employees", "51 to 200 employees",
        "201 to 500 employees", "501 to 1000 employees",
        "1001 to 5000 employees", "5001 to 10000 employees",
        "10000+ employees",
    ]
    valid_sizes = [s for s in size_order if s in df["Size"].values]

    fig, ax = plt.subplots(figsize=(13, 6))
    sns.boxplot(data=df[df["Size"].isin(valid_sizes)],
                x="Size", y="avg_salary", order=valid_sizes,
                palette=PALETTE, ax=ax)
    ax.set_xlabel("Company Size")
    ax.set_ylabel("Average Salary ($K)")
    ax.set_title("Chart 7: Salary Distribution by Company Size",
                 fontsize=13, fontweight="bold")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    _save(fig, "07_salary_by_company_size")


def plot_rating_vs_salary(df: pd.DataFrame) -> None:
    """
    Chart 8 — Bivariate (Numerical × Numerical)
    Scatter plot of company Rating vs avg_salary with a regression
    line to check if higher-rated companies pay better.
    """
    data = df[["Rating", "avg_salary"]].dropna()

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.scatter(data["Rating"], data["avg_salary"],
               alpha=0.4, color="#4C72B0", s=30, label="Job Posting")

    # Regression line
    slope, intercept, r, p, _ = stats.linregress(data["Rating"], data["avg_salary"])
    x_range = np.linspace(data["Rating"].min(), data["Rating"].max(), 100)
    ax.plot(x_range, slope * x_range + intercept,
            color="red", linewidth=2, label=f"Trend (r={r:.2f})")

    ax.set_xlabel("Company Rating")
    ax.set_ylabel("Average Salary ($K)")
    ax.set_title("Chart 8: Company Rating vs Average Salary",
                 fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    _save(fig, "08_rating_vs_salary")


def plot_salary_by_ownership(df: pd.DataFrame) -> None:
    """
    Chart 9 — Bivariate (Categorical × Numerical)
    Violin plot of salary by ownership type (Public, Private, Non-profit, etc.)
    to reveal structural salary differences by org type.
    """
    data = df[df["Type of ownership"].notna()].copy()
    top_types = data["Type of ownership"].value_counts().head(6).index
    data = data[data["Type of ownership"].isin(top_types)]

    order = (data.groupby("Type of ownership")["avg_salary"]
                  .median().sort_values(ascending=False).index)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(data=data, x="Type of ownership", y="avg_salary",
                   order=order, palette=PALETTE, ax=ax, inner="quartile")
    ax.set_xlabel("Type of Ownership")
    ax.set_ylabel("Average Salary ($K)")
    ax.set_title("Chart 9: Salary by Company Ownership Type",
                 fontsize=13, fontweight="bold")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    _save(fig, "09_salary_by_ownership")


def plot_major_city_premium(df: pd.DataFrame) -> None:
    """
    Chart 10 — Bivariate (Categorical × Numerical)
    Compare avg salary for major city vs non-major city postings.
    """
    data = df[["is_major_city", "avg_salary"]].dropna()
    data["Location Type"] = data["is_major_city"].map(
        {True: "Major City", False: "Other Location"}
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=data, x="Location Type", y="avg_salary",
                palette=["#55A868", "#C44E52"], ax=ax, width=0.5)

    # Annotate medians
    for i, loc_type in enumerate(["Major City", "Other Location"]):
        med = data[data["Location Type"] == loc_type]["avg_salary"].median()
        ax.text(i, med + 1, f"${med:.0f}K", ha="center", fontsize=11, color="black")

    ax.set_ylabel("Average Salary ($K)")
    ax.set_title("Chart 10: Salary — Major City vs Other Locations",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "10_major_city_premium")

# MULTIVARIATE ANALYSIS

def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Chart 11 — Multivariate
    Correlation heatmap of all numeric features to identify
    which variables are most predictive of salary.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Keep meaningful numeric columns only
    keep = [c for c in num_cols if not c.startswith("skill_") or
            df[c].sum() > 50]
    corr = df[keep].corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, linewidths=0.5,
                annot_kws={"size": 8}, ax=ax)
    ax.set_title("Chart 11: Correlation Heatmap of Numeric Features",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "11_correlation_heatmap")


def plot_skills_salary_heatmap(df: pd.DataFrame) -> None:
    """
    Chart 12 — Multivariate
    Heatmap of median salary per skill × seniority level to show
    which skill–seniority combos command the highest pay.
    """
    skill_cols = [c for c in df.columns if c.startswith("skill_")]
    records = []
    for skill in skill_cols:
        for level in df["seniority_level"].dropna().unique():
            subset = df[(df[skill] == 1) & (df["seniority_level"] == level)]
            if len(subset) > 5:
                records.append({
                    "skill": skill.replace("skill_", "").replace("_", " ").title(),
                    "seniority": level,
                    "median_salary": subset["avg_salary"].median(),
                })

    pivot = pd.DataFrame(records).pivot(index="skill", columns="seniority", values="median_salary")

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd",
                linewidths=0.4, ax=ax, cbar_kws={"label": "Median Salary ($K)"})
    ax.set_title("Chart 12: Median Salary by Skill × Seniority Level",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Seniority Level")
    ax.set_ylabel("Skill")
    plt.tight_layout()
    _save(fig, "12_skills_seniority_heatmap")


def plot_top_skills_frequency(df: pd.DataFrame) -> None:
    """
    Chart 13 — Multivariate
    Stacked bar: skill demand frequency split by seniority,
    revealing which skills are most common at each career stage.
    """
    skill_cols = [c for c in df.columns if c.startswith("skill_")]
    levels = ["Junior", "Mid-Level", "Senior", "Lead", "Director", "Manager"]
    levels = [l for l in levels if l in df["seniority_level"].values]

    freq = {}
    for level in levels:
        sub = df[df["seniority_level"] == level]
        freq[level] = sub[skill_cols].mean() * 100

    freq_df = pd.DataFrame(freq).T
    freq_df.columns = [c.replace("skill_", "").replace("_", " ").title()
                       for c in freq_df.columns]
    # Keep top 10 skills by overall frequency
    top10 = freq_df.mean().sort_values(ascending=False).head(10).index
    freq_df = freq_df[top10]

    fig, ax = plt.subplots(figsize=(14, 6))
    freq_df.T.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="white")
    ax.set_xlabel("Skill")
    ax.set_ylabel("% of Job Postings Mentioning Skill")
    ax.set_title("Chart 13: Top 10 Skills Frequency by Seniority Level",
                 fontsize=13, fontweight="bold")
    ax.legend(title="Seniority", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    _save(fig, "13_skills_frequency_by_seniority")


# Bonus: Salary Trend by Sector × Job Category

def plot_sector_category_heatmap(df: pd.DataFrame) -> None:
    """
    Chart 14 — Multivariate
    Heatmap of median salary by Sector × job_category to spot
    which role–sector combos are highest-paying.
    """
    pivot = (df.groupby(["Sector", "job_category"])["avg_salary"]
               .median().unstack())
    pivot = pivot.dropna(how="all", axis=0).dropna(how="all", axis=1)

    fig, ax = plt.subplots(figsize=(14, 9))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Blues",
                linewidths=0.3, ax=ax, cbar_kws={"label": "Median Salary ($K)"})
    ax.set_title("Chart 14: Median Salary by Sector × Job Category",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "14_sector_category_heatmap")


# Run All

def run_all_eda(df: pd.DataFrame, show: bool = False) -> None:
    """Execute every EDA plot and save outputs."""
    global SHOW_PLOTS
    SHOW_PLOTS = show

    print("Running EDA plots...")
    plot_salary_distribution(df)       ; print("  [1/14] Salary distribution")
    plot_job_title_distribution(df)    ; print("  [2/14] Job title distribution")
    plot_seniority_distribution(df)    ; print("  [3/14] Seniority distribution")
    plot_sector_distribution(df)       ; print("  [4/14] Sector distribution")
    plot_salary_by_job_category(df)    ; print("  [5/14] Salary by job category")
    plot_salary_by_state(df)           ; print("  [6/14] Salary by state")
    plot_salary_by_company_size(df)    ; print("  [7/14] Salary by company size")
    plot_rating_vs_salary(df)          ; print("  [8/14] Rating vs salary")
    plot_salary_by_ownership(df)       ; print("  [9/14] Salary by ownership")
    plot_major_city_premium(df)        ; print(" [10/14] Major city premium")
    plot_correlation_heatmap(df)       ; print(" [11/14] Correlation heatmap")
    plot_skills_salary_heatmap(df)     ; print(" [12/14] Skills × seniority heatmap")
    plot_top_skills_frequency(df)      ; print(" [13/14] Skills frequency")
    plot_sector_category_heatmap(df)   ; print(" [14/14] Sector × category heatmap")
    print(f"\nAll plots saved to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import load_data
    from preprocessing import run_pipeline

    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "glassdoor_jobs.csv")
    df = run_pipeline(load_data(DATA_PATH))
    run_all_eda(df, show=False)
