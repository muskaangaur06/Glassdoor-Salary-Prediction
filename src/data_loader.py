import pandas as pd
import numpy as np

# Public API

def load_data(filepath: str) -> pd.DataFrame:
    """Load the raw Glassdoor CSV and return as a DataFrame."""
    df = pd.read_csv(filepath, index_col=0)
    df.columns = [c.strip() for c in df.columns]
    return df


def initial_inspection(df: pd.DataFrame) -> dict:
    """
    Return a structured summary of the raw DataFrame.

    Returns
    -------
    dict with keys:
        shape, dtypes, head, missing_counts, missing_pct,
        duplicate_count, unique_counts, describe
    """
    missing_counts = (df == -1).sum() + df.isnull().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)

    return {
        "shape": df.shape,
        "dtypes": df.dtypes,
        "head": df.head(),
        "missing_counts": missing_counts,
        "missing_pct": missing_pct,
        "duplicate_count": df.duplicated().sum(),
        "unique_counts": df.nunique(),
        "describe": df.describe(include="all"),
    }


def print_inspection_report(df: pd.DataFrame) -> None:
    """Pretty-print the inspection report to stdout."""
    info = initial_inspection(df)

    print("=" * 60)
    print(f"  DATASET SHAPE  :  {info['shape'][0]} rows × {info['shape'][1]} columns")
    print(f"  DUPLICATES     :  {info['duplicate_count']}")
    print("=" * 60)

    print("\n--- DATA TYPES ---")
    print(info["dtypes"].to_string())

    print("\n--- MISSING / -1 VALUES ---")
    miss = pd.DataFrame({
        "count": info["missing_counts"],
        "pct (%)": info["missing_pct"],
    })
    print(miss[miss["count"] > 0].to_string() if miss["count"].sum() > 0
          else "  No missing values detected.")

    print("\n--- UNIQUE VALUE COUNTS ---")
    print(info["unique_counts"].to_string())

    print("\n--- FIRST 5 ROWS ---")
    print(info["head"].to_string())


# Quick sanity check when run directly
if __name__ == "__main__":
    import os
    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "glassdoor_jobs.csv")
    df_raw = load_data(DATA_PATH)
    print_inspection_report(df_raw)
