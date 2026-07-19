import streamlit as st
import pandas as pd

st.title("Employee Attrition Analysis")

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

st.write("Dataset Preview")
st.dataframe(df.head())
