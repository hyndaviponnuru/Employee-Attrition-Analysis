import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


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
    "Analyze employee attrition patterns and predict the "
    "potential risk of employee attrition using machine learning."
)


# ============================================================
# LOAD DATA
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
# ATTRITION BY DEPARTMENT
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

ax.set_xlabel("Department")
ax.set_ylabel("Employees Left")
ax.set_title("Employees Who Left by Department")

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
    "These factors show patterns associated with employee "
    "attrition. They should not be interpreted as proven "
    "reasons why an individual employee left."
)

left_df = df[
    df["Attrition"] == "Yes"
]


# ============================================================
# OVERTIME
# ============================================================

col1, col2 = st.columns(2)

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


# ============================================================
# JOB SATISFACTION
# ============================================================

with col2:

    st.subheader(
        "Job Satisfaction"
    )

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


# ============================================================
# WORK-LIFE BALANCE
# ============================================================

st.subheader(
    "Work-Life Balance"
)

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
# MACHINE LEARNING
# ============================================================

st.header(
    "🤖 Employee Attrition Risk Prediction"
)

st.write(
    "A Random Forest classification model is trained using "
    "historical employee data to predict the probability "
    "of employee attrition."
)


# ============================================================
# PREPARE DATA
# ============================================================

ml_df = df.copy()


# Columns that should not be used

columns_to_drop = [
    "Attrition",
    "EmployeeNumber",
    "EmployeeCount",
    "Over18",
    "StandardHours"
]


X = ml_df.drop(
    columns=columns_to_drop
)

y = ml_df["Attrition"].map({
    "No": 0,
    "Yes": 1
})


# ============================================================
# ENCODE CATEGORICAL FEATURES
# ============================================================

X = pd.get_dummies(
    X,
    drop_first=True
)


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(
    X_train,
    y_train
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

y_pred = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    y_pred
)


st.subheader(
    "Model Performance"
)

st.metric(
    "Model Accuracy",
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.subheader(
    "Top Attrition Risk Factors"
)

importance_df = pd.DataFrame({
    "Feature": X.columns,
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

ax4.set_xlabel(
    "Importance"
)

ax4.set_ylabel(
    "Feature"
)

ax4.set_title(
    "Top 10 Factors Used by the Model"
)

st.pyplot(fig4)


# ============================================================
# INTERACTIVE RISK PREDICTOR
# ============================================================

st.header(
    "🔍 Employee Risk Predictor"
)

st.write(
    "Enter employee information to estimate their "
    "potential attrition risk."
)


col1, col2, col3 = st.columns(3)


# ============================================================
# EMPLOYEE INPUTS
# ============================================================

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
        max_value=20000,
        value=5000
    )

    years_at_company = st.number_input(
        "Years at Company",
        min_value=0,
        max_value=40,
        value=3
    )

    total_working_years = st.number_input(
        "Total Working Years",
        min_value=0,
        max_value=40,
        value=5
    )


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


with col3:

    job_level = st.number_input(
        "Job Level",
        min_value=1,
        max_value=5,
        value=2
    )

    job_involvement = st.selectbox(
        "Job Involvement",
        [1, 2, 3, 4],
        index=2
    )

    years_since_promotion = st.number_input(
        "Years Since Last Promotion",
        min_value=0,
        max_value=15,
        value=1
    )

    companies_worked = st.number_input(
        "Number of Companies Worked",
        min_value=0,
        max_value=10,
        value=1
    )


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "Predict Attrition Risk",
    type="primary"
):

    # Create input dataframe
    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=X.columns
    )


    # Numerical features

    numerical_values = {

        "Age": age,

        "MonthlyIncome": monthly_income,

        "YearsAtCompany": years_at_company,

        "TotalWorkingYears":
            total_working_years,

        "JobLevel": job_level,

        "JobInvolvement":
            job_involvement,

        "JobSatisfaction":
            job_satisfaction,

        "WorkLifeBalance":
            work_life_balance,

        "EnvironmentSatisfaction":
            environment_satisfaction,

        "YearsSinceLastPromotion":
            years_since_promotion,

        "NumCompaniesWorked":
            companies_worked
    }


    for feature, value in numerical_values.items():

        if feature in input_data.columns:

            input_data.loc[
                0,
                feature
            ] = value


    # Overtime

    if (
        overtime == "Yes"
        and "OverTime_Yes" in input_data.columns
    ):

        input_data.loc[
            0,
            "OverTime_Yes"
        ] = 1


    # ========================================================
    # PREDICT PROBABILITY
    # ========================================================

    probability = model.predict_proba(
        input_data
    )[0][1]

    probability_percent = (
        probability * 100
    )


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    st.subheader(
        "Attrition Risk Result"
    )


    if probability >= 0.70:

        st.error(
            f"🔴 HIGH RISK — "
            f"{probability_percent:.2f}%"
        )

    elif probability >= 0.40:

        st.warning(
            f"🟠 MEDIUM RISK — "
            f"{probability_percent:.2f}%"
        )

    else:

        st.success(
            f"🟢 LOW RISK — "
            f"{probability_percent:.2f}%"
        )


    st.metric(
        "Probability of Attrition",
        f"{probability_percent:.2f}%"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.write(
    "Employee Attrition Analysis & Risk Prediction Dashboard"
)
