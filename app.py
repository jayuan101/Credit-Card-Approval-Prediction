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
st.title("💳 Credit Card Approval Prediction (User-Friendly)")

st.write("Answer a few questions about yourself and the app will predict if a credit card is likely to be approved.")

# -------------------------------
# Load & Prepare Data
# -------------------------------
@st.cache_data
def load_and_prepare():
    app = pd.read_csv("application_record.csv")
    credit = pd.read_csv("credit_record.csv")

    # Create target from credit_record
    credit["BAD_CLIENT"] = credit["STATUS"].apply(lambda x: 1 if str(x) in ["3","4","5"] else 0)
    target = credit.groupby("ID")["BAD_CLIENT"].max().reset_index()
    target.rename(columns={"BAD_CLIENT": "TARGET"}, inplace=True)

    # Merge
    df = app.merge(target, on="ID", how="inner")
    return df

df = load_and_prepare()

# -------------------------------
# Preprocess Data for Model
# -------------------------------
df_model = df.copy()

# Convert categorical to numeric
label_encoders = {}
for col in df_model.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    label_encoders[col] = le

X = df_model.drop(columns=["ID","TARGET"])
y = df_model["TARGET"]

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
# User-Friendly Input Form
# -------------------------------
st.subheader("📝 Your Information")

# Mapping human-friendly questions to column names
input_data = {}

# Gender
gender = st.radio("What is your gender?", ("Male", "Female"))
input_data["CODE_GENDER"] = 0 if gender=="Male" else 1

# Car ownership
car = st.radio("Do you own a car?", ("Yes", "No"))
input_data["FLAG_OWN_CAR"] = 1 if car=="Yes" else 0

# Real estate ownership
realty = st.radio("Do you own real estate?", ("Yes", "No"))
input_data["FLAG_OWN_REALTY"] = 1 if realty=="Yes" else 0

# Children
children = st.number_input("Number of children", min_value=0, max_value=20, value=0)
input_data["CNT_CHILDREN"] = children

# Income
income = st.number_input("Annual Income (USD)", min_value=0, value=50000)
input_data["AMT_INCOME_TOTAL"] = income

# Education
education = st.selectbox("Education Level", ["Secondary / High School", "Higher Education", "Incomplete Higher", "Lower Secondary", "Academic Degree"])
# encode with LabelEncoder from dataset
le_edu = label_encoders.get("NAME_EDUCATION_TYPE")
if le_edu:
    input_data["NAME_EDUCATION_TYPE"] = le_edu.transform([education])[0]
else:
    input_data["NAME_EDUCATION_TYPE"] = 0

# Family status
family = st.selectbox("Family Status", ["Single / not married", "Married", "Civil marriage", "Separated", "Widow"])
le_fam = label_encoders.get("NAME_FAMILY_STATUS")
if le_fam:
    input_data["NAME_FAMILY_STATUS"] = le_fam.transform([family])[0]
else:
    input_data["NAME_FAMILY_STATUS"] = 0

# Housing type
housing = st.selectbox("Housing Type", ["House / apartment", "Municipal apartment", "Rented apartment", "With parents", "Co-op apartment", "Office apartment", "Hotel"])
le_housing = label_encoders.get("NAME_HOUSING_TYPE")
if le_housing:
    input_data["NAME_HOUSING_TYPE"] = le_housing.transform([housing])[0]
else:
    input_data["NAME_HOUSING_TYPE"] = 0

# Age (in years)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
input_data["DAYS_BIRTH"] = -age*365  # dataset uses negative days

# Years employed
employed = st.number_input("Years employed", min_value=0, max_value=50, value=5)
input_data["DAYS_EMPLOYED"] = -employed*365

# Family members
fam_members = st.number_input("Number of family members", min_value=1, max_value=20, value=1)
input_data["CNT_FAM_MEMBERS"] = fam_members

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Approval"):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    
    if prediction == 0:
        st.success("✅ Likely to be Approved (Good Client)")
    else:
        st.error("❌ Likely to be Denied (Bad Client)")
