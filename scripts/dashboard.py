
import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="US College Outcomes",
    page_icon="🎓",
    layout="wide"
)

@st.cache_data
def load_data():
    df = pd.read_csv(
        "data/processed/college_clean.csv"
    )
    return df

df = load_data()

st.title("🎓 US College Financial Outcomes")
st.caption(
    "Exploring tuition, student debt, and earnings "
    "using US Department of Education College Scorecard data."
)

st.sidebar.header("Filters")

states = st.sidebar.multiselect(
    "States",
    sorted(df["state"].dropna().unique())
)

types = st.sidebar.multiselect(
    "Institution types",
    sorted(df["institution_type"].dropna().unique())
)

degree_filter = st.sidebar.selectbox(
    "Predominant degree",
    ["All", "Bachelor's", "Associate", "Certificate"]
)

filtered = df.copy()

if states:
    filtered = filtered[filtered["state"].isin(states)]

if types:
    filtered = filtered[
        filtered["institution_type"].isin(types)
    ]

degree_codes = {
    "Bachelor's": 3,
    "Associate": 2,
    "Certificate": 1
}

if degree_filter != "All":
    filtered = filtered[
        filtered["predominant_degree"]
        == degree_codes[degree_filter]
    ]

# ------------------------------------------
# Overview
# ------------------------------------------

st.header("National College Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Institutions",
    f"{len(filtered):,}"
)

c2.metric(
    "Median Net Price",
    f"${filtered['net_price'].median():,.0f}"
)

c3.metric(
    "Median Institutional Earnings",
    f"${filtered['earnings_10yr'].median():,.0f}"
)

c4.metric(
    "Median Graduate Debt",
    f"${filtered['graduate_debt'].median():,.0f}"
)

st.caption(
    "Displayed medians are calculated across reporting "
    "institutions, not individual students."
)

# ------------------------------------------
# Tuition vs earnings
# ------------------------------------------

st.header("Tuition vs Earnings")

scatter_data = filtered.dropna(
    subset=["tuition_in_state", "earnings_10yr"]
)

fig = px.scatter(
    scatter_data,
    x="tuition_in_state",
    y="earnings_10yr",
    color="institution_type",
    hover_name="college_name",
    hover_data=["state"],
    opacity=0.6,
    labels={
        "tuition_in_state": "Annual Listed Tuition ($)",
        "earnings_10yr": "Median Earnings ($)",
        "institution_type": "Institution Type"
    },
    title="College Tuition vs Earnings 10 Years After Entry"
)

st.plotly_chart(fig, width="stretch")

# ------------------------------------------
# Compare institution types
# ------------------------------------------

st.header("Earnings by Institution Type")

grouped = (
    filtered.groupby("institution_type")[
        "earnings_10yr"
    ].median().reset_index()
)

fig2 = px.bar(
    grouped,
    x="institution_type",
    y="earnings_10yr",
    color="institution_type",
    title="Median Institutional Earnings by Type",
    labels={
        "institution_type": "Institution Type",
        "earnings_10yr": "Median Earnings ($)"
    }
)

st.plotly_chart(fig2, width="stretch")

# ------------------------------------------
# Debt analysis
# ------------------------------------------

st.header("Graduate Debt vs Earnings")

debt_data = filtered.dropna(
    subset=["graduate_debt", "earnings_10yr"]
)

fig3 = px.scatter(
    debt_data,
    x="graduate_debt",
    y="earnings_10yr",
    color="institution_type",
    hover_name="college_name",
    title="Graduate Debt vs Earnings",
    labels={
        "graduate_debt": "Median Graduate Debt ($)",
        "earnings_10yr": "Median Annual Earnings ($)"
    }
)

st.plotly_chart(fig3, width="stretch")

# ------------------------------------------
# College search
# ------------------------------------------

st.header("Search and Compare Colleges")

search = st.text_input(
    "Search college name",
    placeholder="Virginia Tech"
)

display = filtered.copy()

if search:
    display = display[
        display["college_name"].str.contains(
            search,
            case=False,
            na=False,
            regex=False
        )
    ]

columns = [
    "college_name",
    "state",
    "institution_type",
    "net_price",
    "tuition_in_state",
    "graduate_debt",
    "earnings_10yr",
    "graduation_rate"
]

st.dataframe(
    display[columns],
    use_container_width=True,
    hide_index=True
)

# ------------------------------------------
# Machine learning performance
# ------------------------------------------

st.header("Machine Learning Model Comparison")

model_path = Path(
    "outputs/machine_learning/model_comparison.csv"
)

cv_path = Path(
    "outputs/machine_learning/cross_validation.csv"
)

if model_path.exists():
    models = pd.read_csv(model_path)

    fig4 = px.bar(
        models,
        x="model",
        y="MAE",
        color="model",
        title="Model Test Error (Lower Is Better)"
    )

    st.plotly_chart(fig4, width="stretch")
    st.dataframe(models, hide_index=True)

if cv_path.exists():
    st.subheader("Five-Fold Cross-Validation")
    st.dataframe(
        pd.read_csv(cv_path),
        hide_index=True
    )

st.caption(
    "Models predict institution-level median earnings, "
    "not an individual student's future salary. "
    "Comparisons are observational, not causal."
)

st.success("College outcomes dashboard loaded successfully!")
