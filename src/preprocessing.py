"""
All data-cleaning and feature-engineering steps for the Glassdoor dataset.
Each function is pure: receives a DataFrame, returns a transformed copy.
Call `run_pipeline(df)` to execute every step in sequence.
"""

import re
import numpy as np
import pandas as pd
from datetime import datetime

CURRENT_YEAR = datetime.now().year

# 1. Salary Parsing

def parse_salary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse 'Salary Estimate' strings like '$53K-$91K (Glassdoor est.)'
    into numeric salary_min, salary_max, avg_salary (all in $K).
    Rows where parsing fails are set to NaN.
    """
    df = df.copy()

    def _extract(val: str):
        nums = re.findall(r"\d+", str(val))
        if len(nums) >= 2:
            lo, hi = int(nums[0]), int(nums[1])
            return lo, hi, round((lo + hi) / 2, 1)
        return np.nan, np.nan, np.nan

    parsed = df["Salary Estimate"].apply(_extract)
    df["salary_min"] = parsed.apply(lambda x: x[0])
    df["salary_max"] = parsed.apply(lambda x: x[1])
    df["avg_salary"]  = parsed.apply(lambda x: x[2])
    df["salary_range"] = df["salary_max"] - df["salary_min"]
    return df


# 2. Company Name Cleaning

def clean_company_name(df: pd.DataFrame) -> pd.DataFrame:
    """
    Company names sometimes have a trailing '\n<rating>' appended.
    Strip that suffix and whitespace.
    """
    df = df.copy()
    df["Company Name"] = df["Company Name"].apply(
        lambda x: x.split("\n")[0].strip() if isinstance(x, str) else x
    )
    return df


# 3. Rating Cleaning

def clean_rating(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Rating to float; replace -1 with NaN."""
    df = df.copy()
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df["Rating"] = df["Rating"].replace(-1, np.nan)
    return df


# 4. Location Features

MAJOR_CITIES = {
    "New York", "San Francisco", "Los Angeles", "Chicago",
    "Seattle", "Boston", "Austin", "Washington", "Houston",
    "Atlanta", "Denver", "Dallas", "San Jose", "San Diego",
}

def extract_location_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    From 'Location' ('City, ST') extract:
      - job_city   : city name
      - job_state  : two-letter state code
      - is_major_city : bool flag
    """
    df = df.copy()

    def _parse(loc: str):
        if not isinstance(loc, str) or loc == "-1":
            return np.nan, np.nan, False
        parts = loc.split(",")
        city  = parts[0].strip() if parts else np.nan
        state = parts[1].strip() if len(parts) > 1 else np.nan
        return city, state, city in MAJOR_CITIES

    parsed = df["Location"].apply(_parse)
    df["job_city"]       = parsed.apply(lambda x: x[0])
    df["job_state"]      = parsed.apply(lambda x: x[1])
    df["is_major_city"]  = parsed.apply(lambda x: x[2])
    return df


# 5. Job Title → Seniority & Category

SENIORITY_KEYWORDS = {
    "junior": "Junior", "jr": "Junior", "associate": "Junior",
    "senior": "Senior", "sr": "Senior", "lead": "Lead",
    "principal": "Lead", "staff": "Lead",
    "director": "Director", "vp": "Director", "head": "Director",
    "manager": "Manager",
}

JOB_CATEGORIES = {
    "data scientist": "Data Scientist",
    "data science": "Data Scientist",
    "machine learning": "ML Engineer",
    "ml engineer": "ML Engineer",
    "data engineer": "Data Engineer",
    "data analyst": "Data Analyst",
    "business analyst": "Business Analyst",
    "software engineer": "Software Engineer",
    "software developer": "Software Engineer",
    "devops": "DevOps Engineer",
    "research scientist": "Research Scientist",
    "statistician": "Statistician",
    "product manager": "Product Manager",
}

def parse_job_title(df: pd.DataFrame) -> pd.DataFrame:
    """Derive seniority_level and job_category from 'Job Title'."""
    df = df.copy()

    def _seniority(title: str) -> str:
        t = str(title).lower()
        for kw, level in SENIORITY_KEYWORDS.items():
            if kw in t:
                return level
        return "Mid-Level"

    def _category(title: str) -> str:
        t = str(title).lower()
        for kw, cat in JOB_CATEGORIES.items():
            if kw in t:
                return cat
        return "Other"

    df["seniority_level"] = df["Job Title"].apply(_seniority)
    df["job_category"]    = df["Job Title"].apply(_category)
    return df


# 6. Skill Flags from Job Description

SKILLS = [
    "python", "r ", "sql", "excel", "tableau", "power bi",
    "spark", "hadoop", "aws", "azure", "gcp", "tensorflow",
    "keras", "pytorch", "scikit", "nlp", "deep learning",
    "machine learning", "statistics", "java", "scala",
]

def extract_skills(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scan 'Job Description' for skill keywords and create binary flag
    columns: skill_python, skill_sql, skill_aws, etc.
    """
    df = df.copy()
    desc = df["Job Description"].str.lower().fillna("")
    for skill in SKILLS:
        col = "skill_" + skill.strip().replace(" ", "_")
        df[col] = desc.str.contains(skill, regex=False).astype(int)
    return df


# 7. Company Size → Ordinal

SIZE_ORDER = {
    "1 to 50 employees":       1,
    "51 to 200 employees":     2,
    "201 to 500 employees":    3,
    "501 to 1000 employees":   4,
    "1001 to 5000 employees":  5,
    "5001 to 10000 employees": 6,
    "10000+ employees":        7,
}

def encode_size(df: pd.DataFrame) -> pd.DataFrame:
    """Map 'Size' string categories to an ordinal integer (NaN if -1)."""
    df = df.copy()
    df["size_ordinal"] = df["Size"].map(SIZE_ORDER)
    return df


# 8. Company Age

def compute_company_age(df: pd.DataFrame) -> pd.DataFrame:
    """Compute company_age from 'Founded'. Invalid / -1 → NaN."""
    df = df.copy()
    founded = pd.to_numeric(df["Founded"], errors="coerce")
    founded = founded.replace(-1, np.nan)
    df["company_age"] = CURRENT_YEAR - founded
    df.loc[df["company_age"] < 0, "company_age"] = np.nan
    return df


# 9. Revenue → Ordinal
def encode_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Assign a rough ordinal rank to Revenue bands."""
    df = df.copy()
    rev_map = {
        "Less than $1 million (USD)":        1,
        "$1 to $5 million (USD)":            2,
        "$5 to $10 million (USD)":           3,
        "$10 to $25 million (USD)":          4,
        "$25 to $50 million (USD)":          5,
        "$50 to $100 million (USD)":         6,
        "$100 to $500 million (USD)":        7,
        "$500 million to $1 billion (USD)":  8,
        "$1 to $2 billion (USD)":            9,
        "$2 to $5 billion (USD)":           10,
        "$5 to $10 billion (USD)":          11,
        "$10+ billion (USD)":               12,
    }
    df["revenue_ordinal"] = df["Revenue"].map(rev_map)
    return df


# 10. Replace remaining -1 with NaN

def replace_sentinel(df: pd.DataFrame) -> pd.DataFrame:
    """Replace all -1 sentinel values with NaN for proper handling."""
    df = df.copy()
    df = df.replace(-1, np.nan)
    df = df.replace("-1", np.nan)
    return df

# Master Pipeline

def run_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute all preprocessing steps in order and return the
    fully cleaned + feature-engineered DataFrame.
    """
    df = parse_salary(df)
    df = clean_company_name(df)
    df = clean_rating(df)
    df = extract_location_features(df)
    df = parse_job_title(df)
    df = extract_skills(df)
    df = encode_size(df)
    df = compute_company_age(df)
    df = encode_revenue(df)
    df = replace_sentinel(df)
    return df


# Quick sanity check

if __name__ == "__main__":
    import os
    from data_loader import load_data

    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "glassdoor_jobs.csv")
    df_raw = load_data(DATA_PATH)
    df_clean = run_pipeline(df_raw)

    print(f"Shape after preprocessing: {df_clean.shape}")
    print(df_clean[["Job Title", "job_category", "seniority_level",
                     "salary_min", "salary_max", "avg_salary",
                     "job_state", "is_major_city", "size_ordinal",
                     "company_age"]].head(10).to_string())
