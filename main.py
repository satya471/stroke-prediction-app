import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("data.csv")

# Drop unnecessary column
if "id" in df.columns:
    df = df.drop("id", axis=1)

# Handle missing values
df["bmi"] = df["bmi"].fillna(df["bmi"].median())

# Convert categorical to numeric
df = pd.get_dummies(df, drop_first=True)

# Split features and target
X = df.drop("stroke", axis=1)
y = df["stroke"]

# 🔥 Handle imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# --------------------------
# User Input Section
# --------------------------
print("\nEnter Patient Details:")

def get_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except:
            print("Enter valid number")

age = get_float("Age: ")
glucose = get_float("Glucose Level: ")
bmi = get_float("BMI: ")

# Create input matching columns
user_data = X.iloc[0:1].copy()

user_data["age"] = age
user_data["avg_glucose_level"] = glucose
user_data["bmi"] = bmi

# Prediction
prediction = model.predict(user_data)

print("\n--- Result ---")
if prediction[0] == 1:
    print("⚠️ High Stroke Risk")
else:
    print("✅ Low Stroke Risk")