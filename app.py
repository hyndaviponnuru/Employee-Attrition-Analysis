import streamlit as st
import pandas as pd

st.title("Employee Attrition Analysis")

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

st.write("Dataset Preview")
st.dataframe(df.head())

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Employee Attrition Analysis")

# Read dataset
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# ==========================
# KPI Section
# ==========================
total_emp = len(df)
employees_left = len(df[df["Attrition"] == "Yes"])
attrition_rate = (employees_left / total_emp) * 100

col1, col2, col3 = st.columns(3)

col1.metric("Total Employees", total_emp)
col2.metric("Employees Left", employees_left)
col3.metric("Attrition Rate", f"{attrition_rate:.2f}%")

# Dataset Preview
st.subheader("Dataset Preview")
st.dataframe(df)
# ==========================
# Attrition by Department
# ==========================
st.subheader("Attrition by Department")

dept_attrition = (
    df[df["Attrition"] == "Yes"]["Department"]
    .value_counts()
)

fig, ax = plt.subplots(figsize=(6,4))
dept_attrition.plot(kind="bar", ax=ax)

ax.set_xlabel("Department")
ax.set_ylabel("Employees Left")

st.pyplot(fig)
