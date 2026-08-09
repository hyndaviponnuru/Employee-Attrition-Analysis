import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Employee Attrition Analysis",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Title
# -----------------------------
st.title("Employee Attrition Analysis")
st.write("This dashboard provides a basic analysis of employee attrition using HR analytics data.")

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# -----------------------------
# KPI Section
# -----------------------------
total_employees = len(df)
employees_left = len(df[df["Attrition"] == "Yes"])
attrition_rate = (employees_left / total_employees) * 100

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Employees", total_employees)
with col2:
    st.metric("Employees Left", employees_left)
with col3:
    st.metric("Attrition Rate", f"{attrition_rate:.2f}%")

# -----------------------------
# Dataset Preview
# -----------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Attrition by Department Chart
# -----------------------------
st.subheader("Attrition by Department")
dept_attrition = (
    df[df["Attrition"] == "Yes"]["Department"]
    .value_counts()
)
fig, ax = plt.subplots(figsize=(7, 4))
dept_attrition.plot(kind="bar", ax=ax, color="#4C72B0")
ax.set_title("Employees Who Left by Department")
ax.set_xlabel("Department")
ax.set_ylabel("Number of Employees")
ax.tick_params(axis='x', rotation=0)
st.pyplot(fig)

# -----------------------------
# Why They Left — Reason Analysis
# -----------------------------
# The dataset has no direct "reason" column, so we approximate the
# likely drivers of attrition using related HR features.
st.subheader("Why They Left — Likely Reasons")

left_df = df[df["Attrition"] == "Yes"]

reason_col1, reason_col2 = st.columns(2)

with reason_col1:
    st.markdown("**OverTime**")
    overtime_counts = left_df["OverTime"].value_counts()
    fig1, ax1 = plt.subplots(figsize=(5, 3.5))
    overtime_counts.plot(kind="bar", ax=ax1, color="#DD8452")
    ax1.set_xlabel("OverTime")
    ax1.set_ylabel("Employees Left")
    ax1.tick_params(axis='x', rotation=0)
    st.pyplot(fig1)

    st.markdown("**Job Satisfaction Level (1=Low, 4=High)**")
    js_counts = left_df["JobSatisfaction"].value_counts().sort_index()
    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
    js_counts.plot(kind="bar", ax=ax2, color="#55A868")
    ax2.set_xlabel("Job Satisfaction")
    ax2.set_ylabel("Employees Left")
    ax2.tick_params(axis='x', rotation=0)
    st.pyplot(fig2)

with reason_col2:
    st.markdown("**Work-Life Balance (1=Bad, 4=Best)**")
    wlb_counts = left_df["WorkLifeBalance"].value_counts().sort_index()
    fig3, ax3 = plt.subplots(figsize=(5, 3.5))
    wlb_counts.plot(kind="bar", ax=ax3, color="#C44E52")
    ax3.set_xlabel("Work-Life Balance")
    ax3.set_ylabel("Employees Left")
    ax3.tick_params(axis='x', rotation=0)
    st.pyplot(fig3)

    st.markdown("**Years at Company (avg: left vs stayed)**")
    avg_years = df.groupby("Attrition")["YearsAtCompany"].mean()
    fig4, ax4 = plt.subplots(figsize=(5, 3.5))
    avg_years.plot(kind="bar", ax=ax4, color="#8172B2")
    ax4.set_xlabel("Attrition")
    ax4.set_ylabel("Avg Years at Company")
    ax4.tick_params(axis='x', rotation=0)
    st.pyplot(fig4)

st.info(
    "These charts approximate 'why' employees left using related factors "
    "in the dataset (overtime, satisfaction, work-life balance, tenure), "
    "since the raw data does not include an explicit exit-reason field."
)

# -----------------------------
# Attrition by Age Group (extra feature)
# -----------------------------
st.subheader("Attrition by Age Group")

bins = [18, 25, 35, 45, 55, 65]
labels = ["18-25", "26-35", "36-45", "46-55", "56-65"]
df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels, right=True, include_lowest=True)

age_group_attrition = (
    df[df["Attrition"] == "Yes"]["AgeGroup"]
    .value_counts()
    .sort_index()
)

fig5, ax5 = plt.subplots(figsize=(7, 4))
age_group_attrition.plot(kind="bar", ax=ax5, color="#64B5CD")
ax5.set_title("Employees Who Left by Age Group")
ax5.set_xlabel("Age Group")
ax5.set_ylabel("Number of Employees")
ax5.tick_params(axis='x', rotation=0)
st.pyplot(fig5)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.write("Employee Attrition Analysis Dashboard")
