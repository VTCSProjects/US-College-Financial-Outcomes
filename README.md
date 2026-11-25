# U.S. College Financial Outcomes

## Overview

I created this project to explore the relationship between college tuition, student debt, and career earnings. Using data from the U.S. Department of Education, I analyzed 6,272 colleges and universities across 59 states and territories.

## Tools Used

Python, SQL, Pandas, NumPy, Scikit-learn, SQLite, Matplotlib, Plotly, and Streamlit.

## What I Did

- Cleaned and analyzed a dataset containing 3,379 original variables.
- Built a SQL database to compare college costs and financial outcomes.
- Performed statistical analysis and regression modeling.
- Trained and evaluated machine learning models using five-fold cross-validation.
- Created an interactive Streamlit dashboard to compare institutions.

## Key Findings

- Median in-state tuition: $12,115
- Median student debt: $9,500
- Median 10-year earnings: $45,574

## Machine Learning Results

| Model | MAE | R² |
|---|---:|---:|
| Random Forest | $8,136 | 0.540 |
| Gradient Boosting | $8,174 | 0.555 |
| Ridge Regression | $9,170 | 0.444 |
| Baseline | $13,158 | -0.007 |

Random Forest performed best in cross-validation based on MAE, with a final test MAE of approximately $8,029.

## Running the Dashboard

Install dependencies:

`pip install -r requirements.txt`

Run the dashboard:

`streamlit run scripts/dashboard.py`

## Data Source

[U.S. Department of Education College Scorecard](https://collegescorecard.ed.gov/data/)

## Author

Aris Brunovskis — Virginia Tech, B.S. Computer Science
