import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Employee Attrition Analysis & Risk Prediction",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("Employee Attrition Analysis & Risk Prediction")

st.write(
    "This dashboard analyzes employee attrition patterns "
    "and predicts the potential risk of employee attrition "
    "using machine learning."
)


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(
    "WA_Fn-UseC_-HR-Employee-Attrition.csv"
)


# ============================================================
# KPI SECTION
# ============================================================

total_employees = len(df)

employees_left = len(
    df[df["Attrition"] == "Yes"]
)

attrition_rate = (
    employees_left / total_employees
) * 100


col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Employees",
        total_employees
    )

with col2:
    st.metric(
        "Employees Left",
        employees_left
    )

with col3:
    st.metric(
        "Attrition Rate",
        f"{attrition_rate:.2f}%"
    )


# ============================================================
# DEPARTMENT ATTRITION
# ============================================================

st.header("📊 Attrition by Department")

dept_attrition = (
    df[df["Attrition"] == "Yes"]["Department"]
    .value_counts()
)

fig, ax = plt.subplots(figsize=(7, 4))

dept_attrition.plot(
    kind="bar",
    ax=ax
)

ax.set_title(
    "Employees Who Left by Department"
)

ax.set_xlabel("Department")
ax.set_ylabel("Employees Left")

ax.tick_params(
    axis="x",
    rotation=0
)

st.pyplot(fig)


# ============================================================
# FACTORS ASSOCIATED WITH ATTRITION
# ============================================================

st.header("🔎 Factors Associated with Attrition")

st.info(
    "These charts show factors associated with attrition. "
    "They do not prove that a specific factor caused an employee "
    "to leave."
)

left_df = df[
    df["Attrition"] == "Yes"
]


col1, col2 = st.columns(2)


# ------------------------------------------------------------
# OVERTIME
# ------------------------------------------------------------

with col1:

    st.subheader("OverTime")

    overtime_counts = (
        left_df["OverTime"]
        .value_counts()
    )

    fig1, ax1 = plt.subplots(
        figsize=(5, 3.5)
    )

    overtime_counts.plot(
        kind="bar",
        ax=ax1
    )

    ax1.set_xlabel("OverTime")
    ax1.set_ylabel("Employees Left")

    ax1.tick_params(
        axis="x",
        rotation=0
    )

    st.pyplot(fig1)


# ------------------------------------------------------------
# JOB SATISFACTION
# ------------------------------------------------------------

with col2:

    st.subheader("Job Satisfaction")

    satisfaction_counts = (
        left_df["JobSatisfaction"]
        .value_counts()
        .sort_index()
    )

    fig2, ax2 = plt.subplots(
        figsize=(5, 3.5)
    )

    satisfaction_counts.plot(
        kind="bar",
        ax=ax2
    )

    ax2.set_xlabel(
        "Satisfaction Level"
    )

    ax2.set_ylabel(
        "Employees Left"
    )

    ax2.tick_params(
        axis="x",
        rotation=0
    )

    st.pyplot(fig2)


# ------------------------------------------------------------
# WORK LIFE BALANCE
# ------------------------------------------------------------

st.subheader("Work-Life Balance")

balance_counts = (
    left_df["WorkLifeBalance"]
    .value_counts()
    .sort_index()
)

fig3, ax3 = plt.subplots(
    figsize=(7, 4)
)

balance_counts.plot(
    kind="bar",
    ax=ax3
)

ax3.set_xlabel(
    "Work-Life Balance Level"
)

ax3.set_ylabel(
    "Employees Left"
)

ax3.tick_params(
    axis="x",
    rotation=0
)

st.pyplot(fig3)


# ============================================================
# MACHINE LEARNING SECTION
# ============================================================

st.header("🤖 Employee Attrition Risk Prediction")


# ============================================================
# FEATURES USED BY THE MODEL
# ============================================================

features = [
    "Age",
    "MonthlyIncome",
    "OverTime",
    "JobSatisfaction",
    "WorkLifeBalance",
    "EnvironmentSatisfaction",
    "JobLevel",
    "JobInvolvement",
    "YearsAtCompany",
    "YearsSinceLastPromotion",
    "NumCompaniesWorked",
    "Department",
    "JobRole"
]


# ============================================================
# CREATE ML DATASET
# ============================================================

ml_df = df[
    features + ["Attrition"]
].copy()


# ============================================================
# SEPARATE INPUT AND TARGET
# ============================================================

X = ml_df[features]

y = ml_df["Attrition"].map({
    "No": 0,
    "Yes": 1
})


# ============================================================
# ENCODE CATEGORICAL VARIABLES
# ============================================================

X_encoded = pd.get_dummies(
    X,
    columns=[
        "OverTime",
        "Department",
        "JobRole"
    ],
    drop_first=True
)


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    min_samples_leaf=3
)


model.fit(
    X_train,
    y_train
)


# ============================================================
# MODEL EVALUATION
# ============================================================

y_pred = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    y_pred
)


st.subheader("Model Performance")

st.metric(
    "Test Accuracy",
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.subheader(
    "Top Factors Used by the Model"
)

importance_df = pd.DataFrame({
    "Feature": X_encoded.columns,
    "Importance": model.feature_importances_
})


importance_df = (
    importance_df
    .sort_values(
        "Importance",
        ascending=False
    )
    .head(10)
)


fig4, ax4 = plt.subplots(
    figsize=(8, 5)
)

importance_df.sort_values(
    "Importance"
).plot(
    kind="barh",
    x="Feature",
    y="Importance",
    ax=ax4,
    legend=False
)

ax4.set_title(
    "Top 10 Attrition Risk Factors"
)

ax4.set_xlabel(
    "Importance"
)

ax4.set_ylabel(
    "Feature"
)

st.pyplot(fig4)


# ============================================================
# INTERACTIVE EMPLOYEE RISK PREDICTOR
# ============================================================

st.header("🔍 Interactive Employee Risk Predictor")

st.write(
    "Enter the employee's information below. "
    "The trained Random Forest model will estimate "
    "the probability of attrition."
)


# ============================================================
# INPUT FIELDS
# ============================================================

col1, col2, col3 = st.columns(3)


# ------------------------------------------------------------
# COLUMN 1
# ------------------------------------------------------------

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=65,
        value=30
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=1000,
        max_value=50000,
        value=5000
    )

    years_at_company = st.number_input(
        "Years at Company",
        min_value=0,
        max_value=40,
        value=3
    )

    years_since_promotion = st.number_input(
        "Years Since Last Promotion",
        min_value=0,
        max_value=15,
        value=1
    )


# ------------------------------------------------------------
# COLUMN 2
# ------------------------------------------------------------

with col2:

    overtime = st.selectbox(
        "OverTime",
        ["Yes", "No"]
    )

    job_satisfaction = st.selectbox(
        "Job Satisfaction",
        [1, 2, 3, 4],
        index=2
    )

    work_life_balance = st.selectbox(
        "Work-Life Balance",
        [1, 2, 3, 4],
        index=2
    )

    environment_satisfaction = st.selectbox(
        "Environment Satisfaction",
        [1, 2, 3, 4],
        index=2
    )


# ------------------------------------------------------------
# COLUMN 3
# ------------------------------------------------------------

with col3:

    job_level = st.selectbox(
        "Job Level",
        [1, 2, 3, 4, 5],
        index=1
    )

    job_involvement = st.selectbox(
        "Job Involvement",
        [1, 2, 3, 4],
        index=2
    )

    num_companies_worked = st.number_input(
        "Number of Companies Worked",
        min_value=0,
        max_value=10,
        value=1
    )

    department = st.selectbox(
        "Department",
        sorted(df["Department"].unique())
    )

    job_role = st.selectbox(
        "Job Role",
        sorted(df["JobRole"].unique())
    )


# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button(
    "Predict Attrition Risk",
    type="primary"
):

    # --------------------------------------------------------
    # CREATE EMPLOYEE DATAFRAME
    # --------------------------------------------------------

    employee = pd.DataFrame({
        "Age": [age],
        "MonthlyIncome": [monthly_income],
        "OverTime": [overtime],
        "JobSatisfaction": [job_satisfaction],
        "WorkLifeBalance": [work_life_balance],
        "EnvironmentSatisfaction": [
            environment_satisfaction
        ],
        "JobLevel": [job_level],
        "JobInvolvement": [job_involvement],
        "YearsAtCompany": [years_at_company],
        "YearsSinceLastPromotion": [
            years_since_promotion
        ],
        "NumCompaniesWorked": [
            num_companies_worked
        ],
        "Department": [department],
        "JobRole": [job_role]
    })


    # --------------------------------------------------------
    # ENCODE EMPLOYEE DATA
    # --------------------------------------------------------

    employee_encoded = pd.get_dummies(
        employee,
        columns=[
            "OverTime",
            "Department",
            "JobRole"
        ],
        drop_first=True
    )


    # --------------------------------------------------------
    # MATCH TRAINING COLUMNS
    # --------------------------------------------------------

    employee_encoded = employee_encoded.reindex(
        columns=X_encoded.columns,
        fill_value=0
    )


    # --------------------------------------------------------
    # GET ATTRITION PROBABILITY
    # --------------------------------------------------------

    probability = model.predict_proba(
        employee_encoded
    )[0][1]


    probability_percent = (
        probability * 100
    )


    # ========================================================
    # RISK CLASSIFICATION
    # ========================================================

    if probability >= 0.70:

        risk_level = "HIGH"

    elif probability >= 0.40:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    st.subheader(
        "Employee Attrition Risk"
    )


    if risk_level == "HIGH":

        st.error(
            f"🔴 HIGH ATTRITION RISK\n\n"
            f"Probability of Attrition: "
            f"{probability_percent:.2f}%"
        )

        st.write(
            "This employee profile has a relatively "
            "high predicted probability of attrition "
            "according to the trained model."
        )


    elif risk_level == "MEDIUM":

        st.warning(
            f"🟠 MEDIUM ATTRITION RISK\n\n"
            f"Probability of Attrition: "
            f"{probability_percent:.2f}%"
        )

        st.write(
            "This employee profile has a moderate "
            "predicted probability of attrition."
        )


    else:

        st.success(
            f"🟢 LOW ATTRITION RISK\n\n"
            f"Probability of Attrition: "
            f"{probability_percent:.2f}%"
        )

        st.write(
            "This employee profile has a relatively "
            "low predicted probability of attrition."
        )


    # ========================================================
    # PROBABILITY
    # ========================================================

    st.metric(
        "Predicted Attrition Probability",
        f"{probability_percent:.2f}%"
    )


    # ========================================================
    # INTERPRETATION
    # ========================================================

    st.info(
        "The prediction is a statistical estimate based on "
        "patterns learned from historical HR data. It should "
        "be used as a decision-support signal and not as a "
        "definitive judgment about an individual employee."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.write(
    "Employee Attrition Analysis & Risk Prediction Dashboard"
)
