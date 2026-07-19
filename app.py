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
st.subheader(" Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Attrition by Department Chart
# -----------------------------
st.subheader(" Attrition by Department")

dept_attrition = (
    df[df["Attrition"] == "Yes"]["Department"]
    .value_counts()
)

fig, ax = plt.subplots(figsize=(7, 4))
dept_attrition.plot(kind="bar", ax=ax)

ax.set_title("Employees Who Left by Department")
ax.set_xlabel("Department")
ax.set_ylabel("Number of Employees")
ax.tick_params(axis='x', rotation=0)

st.pyplot(fig)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.write("Developed using Streamlit | Employee Attrition Analysis Dashboard")
