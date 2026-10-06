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
    page_title="Employee Attrition Analysis",
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

try:

    df = pd.read_csv(
        "IBM_HR_Attrition_Dataset.csv"
    )

except FileNotFoundError:

    st.error(
        "IBM_HR_Attrition_Dataset.csv was not found. "
        "Please make sure the CSV file is in the same folder "
        "as this Python file."
    )

    st.stop()


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "EmployeeID",
    "Attrition",
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

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    st.error(
        "The following required columns are missing from "
        "IBM_HR_Attrition_Dataset.csv:"
    )

    st.write(missing_columns)

    st.stop()


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
# ATTRITION ANALYSIS
# ============================================================

st.header("Attrition Analysis")

st.write(
    "The following charts show the relationship between "
    "selected employee factors and employees who left the organization."
)


left_df = df[
    df["Attrition"] == "Yes"
]


# ============================================================
# OVERTIME BAR CHART
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

    ax1.set_title(
        "Employees Who Left by OverTime"
    )

    ax1.set_xlabel(
        "OverTime"
    )

    ax1.set_ylabel(
        "Employees Left"
    )

    ax1.tick_params(
        axis="x",
        rotation=0
    )

    st.pyplot(fig1)

    plt.close(fig1)


# ============================================================
# JOB SATISFACTION BAR CHART
# ============================================================

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

    ax2.set_title(
        "Employees Who Left by Job Satisfaction"
    )

    ax2.set_xlabel(
        "Job Satisfaction Level"
    )

    ax2.set_ylabel(
        "Employees Left"
    )

    ax2.tick_params(
        axis="x",
        rotation=0
    )

    st.pyplot(fig2)

    plt.close(fig2)


# ============================================================
# MACHINE LEARNING SECTION
# ============================================================

st.header("Employee Attrition Risk Prediction")


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

X = ml_df[
    features
]

y = ml_df[
    "Attrition"
].map({
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


# ============================================================
# TRAIN MODEL
# ============================================================

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


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.subheader("Model Performance")

st.metric(
    "Test Accuracy",
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# EMPLOYEE ID BASED RISK PREDICTOR
# ============================================================

st.header("Employee Risk Predictor")

st.write(
    "Enter an Employee ID from the dataset to automatically "
    "analyze the employee and estimate their attrition risk."
)


# ============================================================
# EMPLOYEE ID INPUT
# ============================================================

employee_id = st.text_input(
    "Enter Employee ID",
    placeholder="Example: 1001"
)


# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button(
    "Predict Attrition Risk",
    type="primary"
):

    # --------------------------------------------------------
    # CHECK EMPTY ID
    # --------------------------------------------------------

    if employee_id.strip() == "":

        st.warning(
            "Please enter an Employee ID."
        )


    else:

        # ----------------------------------------------------
        # FIND EMPLOYEE
        # ----------------------------------------------------

        employee_data = df[
            df["EmployeeID"].astype(str).str.strip()
            == employee_id.strip()
        ]


        # ----------------------------------------------------
        # EMPLOYEE NOT FOUND
        # ----------------------------------------------------

        if employee_data.empty:

            st.error(
                f"Employee ID {employee_id} was not found "
                "in the dataset."
            )

            st.info(
                "Please enter a valid Employee ID from "
                "IBM_HR_Attrition_Dataset.csv."
            )


        # ----------------------------------------------------
        # EMPLOYEE FOUND
        # ----------------------------------------------------

        else:

            employee = employee_data.iloc[[0]]


            # =================================================
            # DISPLAY EMPLOYEE DETAILS
            # =================================================

            st.subheader("Employee Details")


            col1, col2, col3 = st.columns(3)


            with col1:

                st.write(
                    f"**Employee ID:** "
                    f"{employee['EmployeeID'].iloc[0]}"
                )

                st.write(
                    f"**Age:** "
                    f"{employee['Age'].iloc[0]}"
                )

                st.write(
                    f"**Monthly Income:** "
                    f"{employee['MonthlyIncome'].iloc[0]}"
                )


            with col2:

                st.write(
                    f"**OverTime:** "
                    f"{employee['OverTime'].iloc[0]}"
                )

                st.write(
                    f"**Job Satisfaction:** "
                    f"{employee['JobSatisfaction'].iloc[0]}"
                )

                st.write(
                    f"**Work-Life Balance:** "
                    f"{employee['WorkLifeBalance'].iloc[0]}"
                )


            with col3:

                st.write(
                    f"**Department:** "
                    f"{employee['Department'].iloc[0]}"
                )

                st.write(
                    f"**Job Role:** "
                    f"{employee['JobRole'].iloc[0]}"
                )

                st.write(
                    f"**Years at Company:** "
                    f"{employee['YearsAtCompany'].iloc[0]}"
                )


            # =================================================
            # PREPARE EMPLOYEE DATA FOR MODEL
            # =================================================

            employee_features = employee[
                features
            ].copy()


            # =================================================
            # ENCODE EMPLOYEE DATA
            # =================================================

            employee_encoded = pd.get_dummies(
                employee_features,
                columns=[
                    "OverTime",
                    "Department",
                    "JobRole"
                ],
                drop_first=True
            )


            # =================================================
            # MATCH TRAINING COLUMNS
            # =================================================

            employee_encoded = employee_encoded.reindex(
                columns=X_encoded.columns,
                fill_value=0
            )


            # =================================================
            # PREDICT ATTRITION PROBABILITY
            # =================================================

            probability = model.predict_proba(
                employee_encoded
            )[0][1]


            probability_percent = (
                probability * 100
            )


            # =================================================
            # RISK CLASSIFICATION
            # =================================================

            if probability >= 0.70:

                risk_level = "HIGH"

            elif probability >= 0.40:

                risk_level = "MEDIUM"

            else:

                risk_level = "LOW"


            # =================================================
            # DISPLAY RISK RESULT
            # =================================================

            st.subheader(
                "Employee Attrition Risk"
            )


            if risk_level == "HIGH":

                st.error(
                    f"HIGH ATTRITION RISK\n\n"
                    f"Probability of Attrition: "
                    f"{probability_percent:.2f}%"
                )


            elif risk_level == "MEDIUM":

                st.warning(
                    f"MEDIUM ATTRITION RISK\n\n"
                    f"Probability of Attrition: "
                    f"{probability_percent:.2f}%"
                )


            else:

                st.success(
                    f"LOW ATTRITION RISK\n\n"
                    f"Probability of Attrition: "
                    f"{probability_percent:.2f}%"
                )


            # =================================================
            # PROBABILITY METRIC
            # =================================================

            st.metric(
                "Predicted Attrition Probability",
                f"{probability_percent:.2f}%"
            )


            # =================================================
            # RISK INTERPRETATION
            # =================================================

            if risk_level == "HIGH":

                st.write(
                    "This employee profile has a relatively "
                    "high predicted probability of attrition "
                    "according to the trained Random Forest model."
                )


            elif risk_level == "MEDIUM":

                st.write(
                    "This employee profile has a moderate "
                    "predicted probability of attrition "
                    "according to the trained Random Forest model."
                )


            else:

                st.write(
                    "This employee profile has a relatively "
                    "low predicted probability of attrition "
                    "according to the trained Random Forest model."
                )


            # =================================================
            # IMPORTANT NOTE
            # =================================================

            st.info(
                "The prediction is a statistical estimate based "
                "on patterns learned from historical HR data. "
                "It should be used as a decision-support signal "
                "and not as a definitive judgment about an "
                "individual employee."
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.write(
    "Employee Attrition Analysis & Risk Prediction Dashboard"
)
