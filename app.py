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

# Encode categorical variables and store encoders
label_encoders = {}
for col in df_model.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    label_encoders[col] = le

# Features and Target
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

input_data = {}

# Gender
gender_options = df["CODE_GENDER"].unique().tolist()
gender_map = {0:"Male",1:"Female"}  # map numeric to human readable
gender_reverse = {v:k for k,v in gender_map.items()}
gender = st.radio("What is your gender?", list(gender_reverse.keys()))
input_data["CODE_GENDER"] = gender_reverse[gender]

# Car ownership
car_options = df["FLAG_OWN_CAR"].unique().tolist()
car_map = {0:"No",1:"Yes"}
car_reverse = {v:k for k,v in car_map.items()}
car = st.radio("Do you own a car?", list(car_reverse.keys()))
input_data["FLAG_OWN_CAR"] = car_reverse[car]

# Real estate ownership
realty_options = df["FLAG_OWN_REALTY"].unique().tolist()
realty_map = {0:"No",1:"Yes"}
realty_reverse = {v:k for k,v in realty_map.items()}
realty = st.radio("Do you own real estate?", list(realty_reverse.keys()))
input_data["FLAG_OWN_REALTY"] = realty_reverse[realty]

# Children
children = st.number_input("Number of children", min_value=0, max_value=20, value=0)
input_data["CNT_CHILDREN"] = children

# Income
income = st.number_input("Annual Income (USD)", min_value=0, value=50000)
input_data["AMT_INCOME_TOTAL"] = income

# Education
education_options = df["NAME_EDUCATION_TYPE"].unique().tolist()
education_map = {i:label for i,label in enumerate(label_encoders["NAME_EDUCATION_TYPE"].classes_)}
education_reverse = {v:k for k,v in education_map.items()}
education = st.selectbox("Education Level", list(education_reverse.keys()))
input_data["NAME_EDUCATION_TYPE"] = education_reverse[education]

# Family status
family_options = df["NAME_FAMILY_STATUS"].unique().tolist()
family_map = {i:label for i,label in enumerate(label_encoders["NAME_FAMILY_STATUS"].classes_)}
family_reverse = {v:k for k,v in family_map.items()}
family = st.selectbox("Family Status", list(family_reverse.keys()))
input_data["NAME_FAMILY_STATUS"] = family_reverse[family]

# Housing type
housing_options = df["NAME_HOUSING_TYPE"].unique().tolist()
housing_map = {i:label for i,label in enumerate(label_encoders["NAME_HOUSING_TYPE"].classes_)}
housing_reverse = {v:k for k,v in housing_map.items()}
housing = st.selectbox("Housing Type", list(housing_reverse.keys()))
input_data["NAME_HOUSING_TYPE"] = housing_reverse[housing]

# Age (in years)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
input_data["DAYS_BIRTH"] = -age*365

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
