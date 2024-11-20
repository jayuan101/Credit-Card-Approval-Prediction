# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load datasets
app_data = pd.read_csv("application_record.csv")
credit_data = pd.read_csv("credit_record.csv")

# Exploratory Data Analysis
print("Application Data Overview:")
print(app_data.info())
print("\nCredit Data Overview:")
print(credit_data.info())

print("\nApplication Data Shape:", app_data.shape)
print("Credit Data Shape:", credit_data.shape)

print("\nApplication Data Description:")
print(app_data.describe())
print("\nCredit Data Description:")
print(credit_data.describe())

# Selecting a random ID from credit_data for filtering
selected_id = 5069280
print("\nFiltered Credit Data for Selected ID:")
print(credit_data[credit_data['ID'] == selected_id])

# Merging datasets on 'ID' column
data = pd.merge(app_data, credit_data, on='ID', how='inner')
print("\nDuplicate Records in Merged Data:", data.duplicated().sum())

# Checking for duplicates in individual datasets
print("\nDuplicate Records in Application Data:", app_data.duplicated().sum())
print("Duplicate Records in Credit Data:", credit_data.duplicated().sum())

# Function to calculate null values percentage
def count_null_data(df):
    return round(df.isna().sum() / df.shape[0] * 100, 2)

print("\nNull Records in Merged Data:")
print(count_null_data(data))

# Dropping less relevant columns (optional)
data.drop(['OCCUPATION_TYPE'], axis=1, inplace=True)

# Checking dataset statistics after merging
print("\nMerged Data Columns:")
print(data.columns)
print("\nMerged Data Description:")
print(data.describe())

# Renaming values in 'STATUS' field for clarity
data['STATUS'].value_counts()
