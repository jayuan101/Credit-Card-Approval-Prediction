import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Title
st.title("📊 Credit Card Approval Prediction")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to:", ["Home", "Application Data", "Credit Record"])

# Load Data
@st.cache_data
def load_application_data():
    return pd.read_csv("application_record.csv")

@st.cache_data
def load_credit_data():
    return pd.read_csv("credit_record.csv")

# Home Page
if page == "Home":
    st.write("""
    This app explores **Credit Card Approval Prediction** data.
    - `application_record.csv` contains applicant demographics and employment info.
    - `credit_record.csv` contains monthly repayment history.

    Use the sidebar to explore the data.
    """)

# Application Data Page
elif page == "Application Data":
    st.subheader("Application Record")
    try:
        app_data = load_application_data()
        st.write("Shape:", app_data.shape)
        st.dataframe(app_data.head())

        # Example plot
        fig, ax = plt.subplots()
        app_data['CODE_GENDER'].value_counts().plot(kind="bar", ax=ax)
        ax.set_title("Gender Distribution")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error loading application_record.csv: {e}")

# Credit Record Page
elif page == "Credit Record":
    st.subheader("Credit Record")
    try:
        credit_data = load_credit_data()
        st.write("Shape:", credit_data.shape)
        st.dataframe(credit_data.head())

        # Example overdue status counts
        fig, ax = plt.subplots()
        credit_data['STATUS'].value_counts().sort_index().plot(kind="bar", ax=ax)
        ax.set_title("Credit Status Distribution")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error loading credit_record.csv: {e}")
