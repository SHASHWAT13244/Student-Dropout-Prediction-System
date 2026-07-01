import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("student_dropout_dataset_v3.csv")

# Remove missing values
df = df.dropna()

# Encode categorical columns
encoder = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = encoder.fit_transform(df[col])

# Features and target
X = df.drop("Dropout", axis=1)
y = df["Dropout"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Random Forest Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluation
prediction = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, prediction))
print(classification_report(y_test, prediction))

# Save model and scaler
joblib.dump(model, "student_dropout_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model Saved Successfully")