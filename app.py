import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -------------------------------
# Title
# -------------------------------
st.title("💳 Credit Card Approval Prediction")

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("application_record.csv")
    return df

df = load_data()
st.write("Dataset Shape:", df.shape)
st.write(df.head())

# -------------------------------
# Preprocessing
# -------------------------------
# Example: Encode categorical columns
df = df.copy()
label_encoders = {}
for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Define features (X) and target (y)
# ⚠️ Replace 'TARGET' with your actual label column name
if "TARGET" in df.columns:
    X = df.drop(columns=["TARGET"])
    y = df["TARGET"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.success(f"Model Accuracy: {acc:.2f}")

    # -------------------------------
    # User Input for Prediction
    # -------------------------------
    st.subheader("🔮 Try Your Own Input")

    # Build inputs dynamically
    input_data = {}
    for col in X.columns:
        val = st.number_input(f"{col}", value=float(df[col].mean()))
        input_data[col] = val

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        st.write("✅ Prediction:", "Approved" if prediction == 1 else "Denied")

else:
    st.error("Dataset does not contain a 'TARGET' column for labels.")
