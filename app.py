import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


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
    "This dashboard analyzes employee attrition patterns and uses "
    "machine learning to estimate employee attrition risk."
)


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")


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
# DATASET PREVIEW
# ============================================================

st.subheader("Dataset Preview")

st.dataframe(
    df.head()
)


# ============================================================
# ATTRITION BY DEPARTMENT
# ============================================================

st.subheader("Attrition by Department")

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
ax.set_ylabel("Number of Employees")

ax.tick_params(
    axis="x",
    rotation=0
)

st.pyplot(fig)


# ============================================================
# FACTORS ASSOCIATED WITH ATTRITION
# ============================================================

st.subheader("Factors Associated with Attrition")

st.info(
    "These charts show factors associated with attrition. "
    "They should not be interpreted as proven reasons why an "
    "individual employee left."
)

left_df = df[df["Attrition"] == "Yes"]


reason_col1, reason_col2 = st.columns(2)


# -----------------------------
# OVERTIME
# -----------------------------

with reason_col1:

    st.markdown("### OverTime")

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


# -----------------------------
# JOB SATISFACTION
# -----------------------------

with reason_col2:

    st.markdown(
        "### Job Satisfaction Level (1=Low, 4=High)"
    )

    js_counts = (
        left_df["JobSatisfaction"]
        .value_counts()
        .sort_index()
    )

    fig2, ax2 = plt.subplots(
        figsize=(5, 3.5)
    )

    js_counts.plot(
        kind="bar",
        ax=ax2
    )

    ax2.set_xlabel("Job Satisfaction")
    ax2.set_ylabel("Employees Left")

    ax2.tick_params(
        axis="x",
        rotation=0
    )

    st.pyplot(fig2)


# -----------------------------
# WORK LIFE BALANCE
# -----------------------------

reason_col3, reason_col4 = st.columns(2)


with reason_col3:

    st.markdown(
        "### Work-Life Balance (1=Bad, 4=Best)"
    )

    wlb_counts = (
        left_df["WorkLifeBalance"]
        .value_counts()
        .sort_index()
    )

    fig3, ax3 = plt.subplots(
        figsize=(5, 3.5)
    )

    wlb_counts.plot(
        kind="bar",
        ax=ax3
    )

    ax3.set_xlabel("Work-Life Balance")
    ax3.set_ylabel("Employees Left")

    ax3.tick_params(
        axis="x",
        rotation=0
    )

    st.pyplot(fig3)


# -----------------------------
# YEARS AT COMPANY
# -----------------------------

with reason_col4:

    st.markdown(
        "### Average Years at Company"
    )

    avg_years = (
        df.groupby("Attrition")["YearsAtCompany"]
        .mean()
    )

    fig4, ax4 = plt.subplots(
        figsize=(5, 3.5)
    )

    avg_years.plot(
        kind="bar",
        ax=ax4
    )

    ax4.set_xlabel("Attrition")
    ax4.set_ylabel("Average Years")

    ax4.tick_params(
        axis="x",
        rotation=0
    )

    st.pyplot(fig4)


# ============================================================
# ATTRITION BY AGE GROUP
# ============================================================

st.subheader("Attrition by Age Group")

bins = [18, 25, 35, 45, 55, 65]

labels = [
    "18-25",
    "26-35",
    "36-45",
    "46-55",
    "56-65"
]

df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=bins,
    labels=labels,
    right=True,
    include_lowest=True
)

age_group_attrition = (
    df[df["Attrition"] == "Yes"]["AgeGroup"]
    .value_counts()
    .sort_index()
)

fig5, ax5 = plt.subplots(
    figsize=(7, 4)
)

age_group_attrition.plot(
    kind="bar",
    ax=ax5
)

ax5.set_title(
    "Employees Who Left by Age Group"
)

ax5.set_xlabel("Age Group")
ax5.set_ylabel("Number of Employees")

ax5.tick_params(
    axis="x",
    rotation=0
)

st.pyplot(fig5)


# ============================================================
# MACHINE LEARNING MODEL
# ============================================================

st.header("🤖 Employee Attrition Risk Prediction")

st.write(
    "A Random Forest classification model is trained using "
    "historical employee data to estimate the probability "
    "of employee attrition."
)


# ============================================================
# PREPARE DATA FOR MACHINE LEARNING
# ============================================================

ml_df = df.copy()


# Remove columns that are not useful for prediction

columns_to_drop = [
    "Attrition",
    "EmployeeNumber",
    "EmployeeCount",
    "Over18",
    "StandardHours",
    "AgeGroup"
]

X = ml_df.drop(
    columns=columns_to_drop
)

y = ml_df["Attrition"]


# ============================================================
# ENCODE CATEGORICAL VARIABLES
# ============================================================

X = pd.get_dummies(
    X,
    drop_first=True
)


# Convert target

y = y.map({
    "No": 0,
    "Yes": 1
})


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
# RANDOM FOREST MODEL
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
# MODEL EVALUATION
# ============================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)


st.subheader("Model Performance")

st.metric(
    "Model Accuracy",
    f"{accuracy * 100:.2f}%"
)

st.caption(
    "Accuracy is measured on the held-out test dataset. "
    "For attrition prediction, precision, recall and class balance "
    "should also be considered rather than relying only on accuracy."
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.subheader(
    "Important Factors Used by the Model"
)

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = (
    feature_importance
    .sort_values(
        "Importance",
        ascending=False
    )
    .head(10)
)


fig6, ax6 = plt.subplots(
    figsize=(8, 5)
)

feature_importance.sort_values(
    "Importance"
).plot(
    kind="barh",
    x="Feature",
    y="Importance",
    ax=ax6,
    legend=False
)

ax6.set_title(
    "Top 10 Factors Influencing Model Predictions"
)

ax6.set_xlabel(
    "Importance"
)

ax6.set_ylabel(
    "Feature"
)

st.pyplot(fig6)


# ============================================================
# INTERACTIVE EMPLOYEE RISK PREDICTOR
# ============================================================

st.header("🔍 Interactive Employee Risk Predictor")

st.write(
    "Enter employee information below to estimate the "
    "employee's probability of attrition."
)


# ============================================================
# INPUT SECTION
# ============================================================

input_col1, input_col2, input_col3 = st.columns(3)


with input_col1:

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


with input_col2:

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


with input_col3:

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

    years_since_last_promotion = st.number_input(
        "Years Since Last Promotion",
        min_value=0,
        max_value=15,
        value=1
    )

    num_companies_worked = st.number_input(
        "Number of Companies Worked",
        min_value=0,
        max_value=10,
        value=1
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button(
    "Predict Attrition Risk",
    type="primary"
):

    # Create empty dataframe
    input_data = pd.DataFrame(
        columns=X.columns
    )

    # Create one row filled with zero
    input_data.loc[0] = 0


    # --------------------------------------------------------
    # NUMERICAL FEATURES
    # --------------------------------------------------------

    numerical_values = {

        "Age": age,

        "MonthlyIncome": monthly_income,

        "YearsAtCompany": years_at_company,

        "TotalWorkingYears": total_working_years,

        "JobLevel": job_level,

        "JobInvolvement": job_involvement,

        "JobSatisfaction": job_satisfaction,

        "WorkLifeBalance": work_life_balance,

        "EnvironmentSatisfaction":
            environment_satisfaction,

        "YearsSinceLastPromotion":
            years_since_last_promotion,

        "NumCompaniesWorked":
            num_companies_worked
    }


    for feature, value in numerical_values.items():

        if feature in input_data.columns:

            input_data.loc[0, feature] = value


    # --------------------------------------------------------
    # OVERTIME
    # --------------------------------------------------------

    overtime_column = "OverTime_Yes"

    if overtime_column in input_data.columns:

        if overtime == "Yes":

            input_data.loc[
                0,
                overtime_column
            ] = 1


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    probability = model.predict_proba(
        input_data
    )[0][1]


    prediction = model.predict(
        input_data
    )[0]


    probability_percent = probability * 100


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    st.subheader(
        "Attrition Risk Result"
    )


    if probability >= 0.70:

        st.error(
            f"🔴 HIGH ATTRITION RISK — "
            f"{probability_percent:.2f}%"
        )

        st.write(
            "This employee has a relatively high predicted "
            "probability of attrition based on the trained model."
        )


    elif probability >= 0.40:

        st.warning(
            f"🟠 MEDIUM ATTRITION RISK — "
            f"{probability_percent:.2f}%"
        )

        st.write(
            "This employee has a moderate predicted "
            "probability of attrition."
        )


    else:

        st.success(
            f"🟢 LOW ATTRITION RISK — "
            f"{probability_percent:.2f}%"
        )

        st.write(
            "This employee has a relatively low predicted "
            "probability of attrition."
        )


    # --------------------------------------------------------
    # PROBABILITY METRIC
    # --------------------------------------------------------

    st.metric(
        "Predicted Probability of Leaving",
        f"{probability_percent:.2f}%"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.write(
    "Employee Attrition Analysis & Risk Prediction Dashboard"
)
