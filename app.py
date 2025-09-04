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
# Load & Prepare Data
# -------------------------------
@st.cache_data
def load_and_prepare():
    # Load datasets
    app = pd.read_csv("application_record.csv")
    credit = pd.read_csv("credit_record.csv")

    # Define bad clients if STATUS >= 3
    credit["BAD_CLIENT"] = credit["STATUS"].apply(lambda x: 1 if str(x) in ["3","4","5"] else 0)

    # Aggregate at client level
    target = credit.groupby("ID")["BAD_CLIENT"].max().reset_index()
    target.rename(columns={"BAD_CLIENT": "TARGET"}, inplace=True)

    # Merge with application data
    df = app.merge(target, on="ID", how="inner")

    return df

df = load_and_prepare()
st.write("Dataset Shape:", df.shape)
st.write(df.head())

# -------------------------------
# Preprocessing
# -------------------------------
df = df.copy()

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Features and Target
X = df.drop(columns=["ID", "TARGET"])
y = df["TARGET"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
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

input_data = {}
for col in X.columns:
    val = st.number_input(f"{col}", value=float(df[col].mean()))
    input_data[col] = val

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    st.write("✅ Prediction:", "Approved (Good Client)" if prediction == 0 else "Denied (Bad Client)")
