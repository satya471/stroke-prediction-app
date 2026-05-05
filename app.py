import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("data.csv")

if "id" in df.columns:
    df = df.drop("id", axis=1)

df["bmi"] = df["bmi"].fillna(df["bmi"].median())
df = pd.get_dummies(df, drop_first=True)

X = df.drop("stroke", axis=1)
y = df["stroke"]

# Handle imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_resampled, y_resampled)

# UI
st.title("🧠 Stroke Prediction App")

st.write("Enter patient details:")

age = st.number_input("Age", 0, 120)
glucose = st.number_input("Glucose Level", 0, 300)
bmi = st.number_input("BMI", 0.0, 100.0)

if st.button("Predict"):
    user_data = X.iloc[0:1].copy()
    user_data["age"] = age
    user_data["avg_glucose_level"] = glucose
    user_data["bmi"] = bmi

    prediction = model.predict(user_data)

    if prediction[0] == 1:
        st.error("⚠️ High Stroke Risk")
    else:
        st.success("✅ Low Stroke Risk")