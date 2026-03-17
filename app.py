"""
app.py
------
Streamlit web application for Glassdoor Salary Prediction.
Features:
  - Interactive salary predictor (ML model)
  - EDA dashboard with pre-generated charts
  - Gemini API chatbot for salary Q&A
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import joblib

warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from data_loader   import load_data
from preprocessing import run_pipeline, SIZE_ORDER, JOB_CATEGORIES, SENIORITY_KEYWORDS

# Paths
DATA_PATH  = os.path.join(os.path.dirname(__file__), "data", "glassdoor_jobs.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "outputs", "models", "xgboost_salary.pkl")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

# Streamlit Page Config
st.set_page_config(
    page_title="Glassdoor Salary Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load Data & Model (cached)
@st.cache_data
def get_data():
    df_raw   = load_data(DATA_PATH)
    df_clean = run_pipeline(df_raw)
    return df_clean


@st.cache_resource
def get_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


df      = get_data()
model_obj = get_model()

# Sidebar Navigation
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/Glassdoor_logo.svg/1200px-Glassdoor_logo.svg.png",
                 width=180)
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "🔮 Salary Predictor", "📊 EDA Dashboard", "🤖 Ask Gemini"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Data: Glassdoor Jobs 2017–2018  |  Model: XGBoost")


# PAGE 1 — OVERVIEW
if page == "🏠 Overview":
    st.title("💼 Glassdoor Salary Prediction")
    st.markdown("""
    > **Goal:** Predict tech job salaries based on role, location, company size,
    > skills required, and more — using data scraped from Glassdoor (2017–2018).

    ---
    """)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Job Postings", f"{len(df):,}")
    c2.metric("Unique Job Titles",   f"{df['Job Title'].nunique():,}")
    c3.metric("States Covered",      f"{df['job_state'].nunique():,}")
    c4.metric("Avg Salary",          f"${df['avg_salary'].mean():.0f}K")

    st.markdown("---")
    st.subheader("Dataset Sample")
    st.dataframe(
        df[["Job Title", "job_category", "seniority_level",
            "avg_salary", "job_state", "Size", "Sector", "Rating"]].head(20),
        use_container_width=True,
    )

    st.subheader("Quick Stats by Job Category")
    stats = (df.groupby("job_category")["avg_salary"]
               .agg(["mean", "median", "min", "max", "count"])
               .round(1).sort_values("median", ascending=False))
    stats.columns = ["Mean ($K)", "Median ($K)", "Min ($K)", "Max ($K)", "Postings"]
    st.dataframe(stats, use_container_width=True)


# PAGE 2 — SALARY PREDICTOR
elif page == "🔮 Salary Predictor":
    st.title("🔮 Predict Your Salary")
    st.markdown("Fill in the job attributes below and get an estimated salary range.")

    if model_obj is None:
        st.warning("Model not found. Please run `src/models.py` first to train the model.")
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            job_cat   = st.selectbox("Job Category", sorted(df["job_category"].dropna().unique()))
            seniority = st.selectbox("Seniority Level",
                                     ["Junior", "Mid-Level", "Senior", "Lead", "Director", "Manager"])
            state     = st.selectbox("State", sorted(df["job_state"].dropna().unique()))

        with col2:
            size_label = st.selectbox("Company Size", list(SIZE_ORDER.keys()))
            sector     = st.selectbox("Sector", sorted(df["Sector"].dropna().unique()))
            ownership  = st.selectbox("Ownership Type",
                                      sorted(df["Type of ownership"].dropna().unique()))

        with col3:
            rating        = st.slider("Company Rating", 1.0, 5.0, 3.8, 0.1)
            company_age   = st.slider("Company Age (years)", 1, 150, 20)
            is_major_city = st.checkbox("Major City (e.g. NYC, SF, Seattle)", value=True)

        st.markdown("**Skills mentioned in job description:**")
        skill_cols  = [c for c in df.columns if c.startswith("skill_")]
        skill_names = [c.replace("skill_", "").replace("_", " ").title() for c in skill_cols]
        selected    = st.multiselect("Select skills", skill_names, default=["Python", "Sql", "Machine Learning"])

        if st.button("🚀 Predict Salary", type="primary"):
            # Build input row
            skill_flags = {sc: int(sn in selected)
                           for sc, sn in zip(skill_cols, skill_names)}

            from sklearn.preprocessing import LabelEncoder
            # Encode categoricals the same way as training
            le = LabelEncoder()
            row = {
                "Rating":            rating,
                "size_ordinal":      SIZE_ORDER.get(size_label, 4),
                "company_age":       company_age,
                "revenue_ordinal":   5,         # default mid-range
                "is_major_city":     int(is_major_city),
                "salary_range":      30,         # typical spread
                "job_category":      sorted(df["job_category"].dropna().unique()).index(job_cat),
                "seniority_level":   ["Junior","Mid-Level","Senior","Lead","Director","Manager"].index(seniority),
                "job_state":         sorted(df["job_state"].dropna().unique()).index(state)
                                     if state in df["job_state"].values else 0,
                "Sector":            sorted(df["Sector"].dropna().unique()).index(sector)
                                     if sector in df["Sector"].values else 0,
                "Type of ownership": sorted(df["Type of ownership"].dropna().unique()).index(ownership)
                                     if ownership in df["Type of ownership"].values else 0,
            }
            row.update(skill_flags)

            from sklearn.impute import SimpleImputer
            X_input = pd.DataFrame([row])
            imp = model_obj.get("imputer")
            mdl = model_obj.get("model")

            X_imp = imp.transform(X_input) if imp else X_input.values
            pred  = mdl.predict(X_imp)[0]

            st.success(f"### Estimated Average Salary: **${pred:.0f}K / year**")
            st.info(f"Expected Range: **${pred - 10:.0f}K — ${pred + 10:.0f}K**")

            st.markdown(f"""
            | Attribute | Value |
            |---|---|
            | Role | {job_cat} ({seniority}) |
            | Location | {state} ({'Major City' if is_major_city else 'Other'}) |
            | Company Size | {size_label} |
            | Sector | {sector} |
            | Rating | {rating} |
            """)


# PAGE 3 — EDA DASHBOARD
elif page == "📊 EDA Dashboard":
    st.title("📊 EDA Dashboard")
    st.markdown("Pre-generated visualisations from the analysis pipeline.")

    chart_meta = {
        "01_salary_distribution.png":        ("Salary Distribution", "Univariate"),
        "02_job_title_distribution.png":      ("Job Category Counts", "Univariate"),
        "03_seniority_distribution.png":      ("Seniority Breakdown", "Univariate"),
        "04_sector_distribution.png":         ("Top Sectors", "Univariate"),
        "05_salary_by_job_category.png":      ("Salary by Job Category", "Bivariate"),
        "06_salary_by_state.png":             ("Salary by State", "Bivariate"),
        "07_salary_by_company_size.png":      ("Salary by Company Size", "Bivariate"),
        "08_rating_vs_salary.png":            ("Rating vs Salary", "Bivariate"),
        "09_salary_by_ownership.png":         ("Salary by Ownership Type", "Bivariate"),
        "10_major_city_premium.png":          ("Major City Salary Premium", "Bivariate"),
        "11_correlation_heatmap.png":         ("Correlation Heatmap", "Multivariate"),
        "12_skills_seniority_heatmap.png":    ("Skills × Seniority Heatmap", "Multivariate"),
        "13_skills_frequency_by_seniority.png": ("Skill Demand by Seniority", "Multivariate"),
        "14_sector_category_heatmap.png":     ("Sector × Category Heatmap", "Multivariate"),
        "15_model_comparison.png":            ("Model Comparison", "ML"),
        "16_feature_importance.png":          ("Feature Importance", "ML"),
        "17_actual_vs_predicted.png":         ("Actual vs Predicted", "ML"),
    }

    filter_type = st.selectbox("Filter by Analysis Type",
                                ["All", "Univariate", "Bivariate", "Multivariate", "ML"])

    for fname, (title, analysis_type) in chart_meta.items():
        if filter_type != "All" and analysis_type != filter_type:
            continue
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(fpath):
            st.subheader(f"{title}  `[{analysis_type}]`")
            with open(fpath, "rb") as _f:
                st.image(_f.read(), use_column_width=True)
            st.markdown("---")

    if not any(os.path.exists(os.path.join(OUTPUT_DIR, f)) for f in chart_meta):
        st.info("No charts found. Run `src/eda.py` to generate them first.")


# PAGE 4 — ASK GEMINI
elif page == "🤖 Ask Gemini":
    st.title("🤖 Ask Gemini about Salaries")
    st.markdown("""
    Ask any question about tech salaries, job market trends, or career decisions.
    Powered by **Google Gemini API**.
    """)

    api_key = st.text_input("Enter your Gemini API Key", type="password",
                             help="Get your key from Google AI Studio")

    if api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-pro")

            # Context summary from data
            context = f"""
            You are a salary analysis expert. You have access to Glassdoor job posting data (2017-2018).
            Key statistics:
            - Total postings: {len(df):,}
            - Average salary: ${df['avg_salary'].mean():.0f}K
            - Highest paying role: {df.groupby('job_category')['avg_salary'].median().idxmax()}
              (median: ${df.groupby('job_category')['avg_salary'].median().max():.0f}K)
            - Top paying state: {df.groupby('job_state')['avg_salary'].median().idxmax()}
            Answer questions about salary trends, career decisions, and compensation negotiation.
            Be concise and data-driven.
            """

            user_q = st.text_area("Your question:", height=100,
                                   placeholder="e.g. What salary should I expect as a senior data scientist in New York?")

            if st.button("Ask Gemini", type="primary") and user_q:
                with st.spinner("Generating response..."):
                    response = model.generate_content(context + "\n\nUser question: " + user_q)
                    st.markdown("### Gemini's Answer")
                    st.markdown(response.text)

        except ImportError:
            st.error("Install google-generativeai: `pip install google-generativeai`")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("Enter your Gemini API key above to start chatting.")

        st.markdown("### Sample Questions You Can Ask")
        samples = [
            "What salary should I expect as a mid-level data scientist in San Francisco?",
            "How much more do senior roles pay compared to junior roles?",
            "Which tech skills have the highest salary premium?",
            "Is it worth relocating to New York for a data engineering role?",
            "How does company size affect compensation in tech?",
        ]
        for q in samples:
            st.markdown(f"- *{q}*")
