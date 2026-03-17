"""
Machine Learning pipeline for Glassdoor Salary Prediction.

Steps:
  1. Feature selection & encoding
  2. Train/test split
  3. Baseline → Ridge → Random Forest → XGBoost
  4. Evaluation (MAE, RMSE, R²)
  5. Feature importance (SHAP for XGBoost)
  6. Save best model with joblib
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.linear_model    import LinearRegression, Ridge, Lasso
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline        import Pipeline
from sklearn.impute           import SimpleImputer

import xgboost as xgb

warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42

# 1. Feature Preparation

CATEGORICAL_FEATURES = ["job_category", "seniority_level", "job_state",
                         "Sector", "Type of ownership"]

NUMERIC_FEATURES = ["Rating", "size_ordinal", "company_age",
                    "revenue_ordinal", "is_major_city", "salary_range"]

SKILL_FEATURES = []   # populated dynamically


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build feature matrix X and target vector y.

    Target  : avg_salary
    Features: numeric + encoded categoricals + skill flags
    """
    global SKILL_FEATURES
    SKILL_FEATURES = [c for c in df.columns if c.startswith("skill_")]

    df = df.copy()
    df = df[df["avg_salary"].notna()].reset_index(drop=True)

    # Encode categoricals
    le = LabelEncoder()
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
            df[col] = le.fit_transform(df[col].astype(str))

    # is_major_city → int
    df["is_major_city"] = df["is_major_city"].astype(int)

    all_features = (
        [f for f in NUMERIC_FEATURES if f in df.columns]
        + [f for f in CATEGORICAL_FEATURES if f in df.columns]
        + SKILL_FEATURES
    )

    X = df[all_features]
    y = df["avg_salary"]
    return X, y


# 2. Evaluation Helper

def evaluate(model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return MAE, RMSE, R² for a set of predictions."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"  {model_name:<30}  MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")
    return {"model": model_name, "MAE": mae, "RMSE": rmse, "R2": r2}


# 3. Model Training

def train_models(X_train: pd.DataFrame, y_train: pd.Series,
                 X_test: pd.DataFrame,  y_test:  pd.Series) -> dict:
    """
    Train four models and return a results summary DataFrame
    plus the fitted model objects.
    """
    imputer = SimpleImputer(strategy="median")
    X_tr = imputer.fit_transform(X_train)
    X_te = imputer.transform(X_test)

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc  = scaler.transform(X_te)

    models_def = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression":  Ridge(alpha=10),
        "Random Forest":     RandomForestRegressor(
                                 n_estimators=200, max_depth=10,
                                 min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost":           xgb.XGBRegressor(
                                 n_estimators=300, learning_rate=0.05,
                                 max_depth=6, subsample=0.8,
                                 colsample_bytree=0.8, random_state=RANDOM_STATE,
                                 verbosity=0),
    }

    results = []
    fitted  = {}
    print("\n=== Model Evaluation on Test Set ===")
    for name, mdl in models_def.items():
        # Linear models use scaled features
        if "Regression" in name:
            mdl.fit(X_tr_sc, y_train)
            preds = mdl.predict(X_te_sc)
        else:
            mdl.fit(X_tr, y_train)
            preds = mdl.predict(X_te)
        results.append(evaluate(name, y_test, preds))
        fitted[name] = {"model": mdl, "imputer": imputer,
                        "scaler": scaler if "Regression" in name else None}

    return pd.DataFrame(results).sort_values("RMSE"), fitted


# 4. Cross-Validation for Best Model

def cross_validate_best(X: pd.DataFrame, y: pd.Series,
                         imputer: SimpleImputer) -> None:
    """Run 5-fold CV on XGBoost and print mean ± std R²."""
    X_imp = imputer.fit_transform(X)
    model = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, verbosity=0
    )
    scores = cross_val_score(model, X_imp, y, cv=5, scoring="r2")
    print(f"\n5-Fold CV (XGBoost)  R²: {scores.mean():.4f} ± {scores.std():.4f}")


# 5. Plots

def plot_model_comparison(results_df: pd.DataFrame) -> None:
    """Bar chart comparing MAE and R² across models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Performance Comparison", fontsize=13, fontweight="bold")

    palette = sns.color_palette("Set2", len(results_df))

    axes[0].barh(results_df["model"], results_df["MAE"], color=palette)
    axes[0].set_xlabel("MAE ($K)")
    axes[0].set_title("Mean Absolute Error (lower is better)")
    axes[0].invert_xaxis()

    axes[1].barh(results_df["model"], results_df["R2"], color=palette)
    axes[1].set_xlabel("R²")
    axes[1].set_title("R² Score (higher is better)")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "15_model_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_feature_importance(model: xgb.XGBRegressor,
                             feature_names: list[str],
                             top_n: int = 20) -> None:
    """Horizontal bar chart of XGBoost feature importances."""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    top[::-1].plot.barh(ax=ax, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title(f"Top {top_n} Feature Importances — XGBoost",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "16_feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_actual_vs_predicted(y_test: pd.Series, y_pred: np.ndarray,
                              model_name: str = "XGBoost") -> None:
    """Scatter: actual vs predicted salary with ideal line."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.4, color="#4C72B0", s=25)
    lims = [min(y_test.min(), y_pred.min()),
            max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Ideal")
    ax.set_xlabel("Actual Salary ($K)")
    ax.set_ylabel("Predicted Salary ($K)")
    ax.set_title(f"Actual vs Predicted Salary — {model_name}",
                 fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "17_actual_vs_predicted.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# 6. Save / Load Model

def save_model(fitted_obj: dict, name: str = "xgboost_salary") -> str:
    """Persist model + imputer to disk."""
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    joblib.dump(fitted_obj, path)
    print(f"  Model saved: {path}")
    return path


def load_model(name: str = "xgboost_salary") -> dict:
    """Load persisted model from disk."""
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    return joblib.load(path)


# 7. Master Runner

def run_full_pipeline(df: pd.DataFrame) -> dict:
    """
    End-to-end training pipeline.
    Returns dict with results_df, best model, feature names.
    """
    X, y = prepare_features(df)
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    results_df, fitted = train_models(X_train, y_train, X_test, y_test)

    best_obj   = fitted["XGBoost"]
    best_model = best_obj["model"]
    imputer    = best_obj["imputer"]

    cross_validate_best(X, y, SimpleImputer(strategy="median"))

    X_te_imp = imputer.transform(X_test)
    y_pred   = best_model.predict(X_te_imp)

    plot_model_comparison(results_df)
    plot_feature_importance(best_model, feature_names)
    plot_actual_vs_predicted(y_test, y_pred)

    save_model(best_obj, "xgboost_salary")

    return {
        "results": results_df,
        "best_model": best_model,
        "imputer": imputer,
        "feature_names": feature_names,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }

# Quick run
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader   import load_data
    from preprocessing import run_pipeline

    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "glassdoor_jobs.csv")
    df = run_pipeline(load_data(DATA_PATH))
    run_full_pipeline(df)
