# Glassdoor Salary Prediction

An end-to-end machine learning project that predicts tech job salaries using Glassdoor job posting data (2017–2018). Includes a full data pipeline, exploratory data analysis, model training, and an interactive Streamlit web application powered by XGBoost and Google Gemini.

---

## Features

- **Salary Predictor** — estimate salary based on role, seniority, location, company size, sector, and skills
- **EDA Dashboard** — 17 pre-generated visualisations covering univariate, bivariate, multivariate, and ML analyses
- **Ask Gemini** — Gemini-powered chatbot for data-driven salary Q&A
- **Modular pipeline** — clean separation of data loading, preprocessing, EDA, and modelling

---

## Project Structure

```
Glassdoor-Salary-Prediction/
├── app.py                        # Streamlit application
├── requirements.txt
├── .env.example                  # API key template
├── data/
│   └── glassdoor_jobs.csv        # Raw dataset (262 job postings)
├── src/
│   ├── data_loader.py            # Data loading & inspection
│   ├── preprocessing.py          # 10-step feature engineering pipeline
│   ├── eda.py                    # EDA visualisation generators
│   └── models.py                 # Model training, evaluation & SHAP analysis
├── notebooks/
│   └── Glassdoor_Salary_Prediction.ipynb
└── outputs/
    ├── *.png                     # 17 EDA/ML charts (auto-generated)
    └── models/
        └── xgboost_salary.pkl    # Trained model (generated at runtime)
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Glassdoor-Salary-Prediction.git
cd Glassdoor-Salary-Prediction
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your Gemini API key (optional — only needed for Ask Gemini page)
```

### 5. Train the model & generate charts

```bash
python -c "from src.data_loader import load_data; from src.preprocessing import run_pipeline; from src.eda import run_eda; from src.models import run_models; import pandas as pd; df = run_pipeline(load_data('data/glassdoor_jobs.csv')); run_eda(df); run_models(df)"
```

### 6. Launch the Streamlit app

```bash
streamlit run app.py
```

---

## Models

Four regression models are trained and compared using 5-fold cross-validation:

| Model | Description |
|---|---|
| Linear Regression | Baseline |
| Ridge Regression | L2-regularised linear model |
| Random Forest | Ensemble of decision trees |
| **XGBoost** | **Best performer — used in production** |

Key engineered features:
- `job_category` — grouped from raw job titles (Data Science, ML Engineering, Data Engineering, etc.)
- `seniority_level` — parsed from job title keywords
- `size_ordinal` — ordinal encoding of company headcount
- `is_major_city` — flag for NYC, SF, Seattle, LA, Boston, Chicago, Austin
- `skill_*` — binary flags for 14 technical skills (Python, SQL, R, Spark, AWS, etc.)
- `company_age` — derived from founding year
- `salary_range` — spread between min/max posted salary

---

## EDA Highlights

| Chart | Insight |
|---|---|
| Salary by Job Category | Machine Learning roles command the highest median salaries |
| Major City Premium | Major-city postings pay ~15% more on average |
| Skills × Seniority Heatmap | Python and SQL appear across all levels; Spark/Cloud skew senior |
| Feature Importance | `salary_range`, `size_ordinal`, and `seniority_level` are top predictors |

---

## Dataset

- **Source:** Glassdoor job postings scraped 2017–2018 (US tech roles)
- **Rows:** 262 job postings
- **Key columns:** `Job Title`, `Salary Estimate`, `Company Name`, `Location`, `Size`, `Founded`, `Type of ownership`, `Industry`, `Sector`, `Rating`

---

## Requirements

- Python 3.8+
- See [requirements.txt](requirements.txt) for full dependency list

Key packages: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `streamlit`, `matplotlib`, `seaborn`, `google-generativeai`

---

## License

[MIT](LICENSE)
