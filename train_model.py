import pandas as pd
import numpy as np
import joblib
import warnings
import os
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report, 
    confusion_matrix,
    roc_auc_score
)

print("="*60)
print("🎓 STUDENT DROPOUT PREDICTION - TRAINING SCRIPT")
print("="*60)

# ============================================================================
# 0. CREATE MODELS DIRECTORY
# ============================================================================

print("\n📁 Creating models directory...")
os.makedirs("models", exist_ok=True)
print("   ✅ Models directory created/verified")

# ============================================================================
# 1. LOAD AND PREPROCESS DATA
# ============================================================================

print("\n📂 Loading dataset...")
df = pd.read_csv("student_dropout_dataset_v3.csv")
print(f"   Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

# Remove missing values
print("\n🧹 Cleaning data...")
initial_shape = df.shape[0]
df = df.dropna()
print(f"   Removed {initial_shape - df.shape[0]} rows with missing values")
print(f"   New shape: {df.shape}")

# ============================================================================
# 2. ENCODE CATEGORICAL VARIABLES
# ============================================================================

print("\n🔢 Encoding categorical variables...")
encoder = LabelEncoder()
categorical_cols = df.select_dtypes(include="object").columns

for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])
    print(f"   Encoded: {col}")

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================

print("\n🔧 Feature engineering...")

# Remove Student_ID if it exists (adds noise to the model)
if 'Student_ID' in df.columns:
    df = df.drop('Student_ID', axis=1)
    print("   Removed 'Student_ID' (identifier column adds noise)")

# Display target distribution
print(f"\n📊 Target Distribution:")
dropout_count = df['Dropout'].sum()
total = len(df)
print(f"   Dropout: {dropout_count} ({dropout_count/total*100:.1f}%)")
print(f"   No Dropout: {total - dropout_count} ({(total-dropout_count)/total*100:.1f}%)")

# ============================================================================
# 4. FEATURES AND TARGET
# ============================================================================

print("\n📌 Preparing features and target...")
X = df.drop("Dropout", axis=1)
y = df["Dropout"]

print(f"   Features: {X.shape[1]} columns")
print(f"   Target: {y.name}")

# ============================================================================
# 5. TRAIN-TEST SPLIT
# ============================================================================

print("\n🔄 Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # Maintain class distribution
)

print(f"   Training set: {len(X_train)} samples")
print(f"   Test set: {len(X_test)} samples")
print(f"   Training dropout rate: {y_train.mean():.2%}")
print(f"   Test dropout rate: {y_test.mean():.2%}")

# ============================================================================
# 6. FEATURE SCALING
# ============================================================================

print("\n📏 Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("   ✅ Features scaled successfully!")

# ============================================================================
# 7. TRAIN MODELS WITH HYPERPARAMETER TUNING
# ============================================================================

print("\n" + "="*60)
print("🤖 TRAINING MODELS")
print("="*60)

models = {}
results = []

# --------------------------------------------------------------------------
# 7.1 Logistic Regression (Baseline)
# --------------------------------------------------------------------------
print("\n📌 Training Logistic Regression...")
lr = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced'
)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
models['Logistic Regression'] = lr

# --------------------------------------------------------------------------
# 7.2 Decision Tree (Tuned)
# --------------------------------------------------------------------------
print("\n📌 Training Decision Tree...")
dt = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced'
)
dt.fit(X_train_scaled, y_train)
dt_pred = dt.predict(X_test_scaled)
models['Decision Tree'] = dt

# --------------------------------------------------------------------------
# 7.3 Random Forest (with Hyperparameter Tuning)
# --------------------------------------------------------------------------
print("\n📌 Training Random Forest with Hyperparameter Tuning...")
print("   ⏳ This may take 2-3 minutes...")

# Parameter grid for tuning
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Base Random Forest
rf_base = RandomForestClassifier(
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

# Randomized Search
grid_search = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

grid_search.fit(X_train_scaled, y_train)

# Best model
rf = grid_search.best_estimator_
rf_pred = rf.predict(X_test_scaled)
models['Random Forest (Tuned)'] = rf

print(f"\n   ✅ Best Parameters: {grid_search.best_params_}")
print(f"   ✅ Best CV Score: {grid_search.best_score_:.4f}")

# ============================================================================
# 8. MODEL EVALUATION
# ============================================================================

print("\n" + "="*60)
print("📊 MODEL EVALUATION")
print("="*60)

def evaluate_model(name, y_true, y_pred):
    """Evaluate a single model"""
    return {
        'Model': name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred)
    }

# Evaluate all models
for name, model in models.items():
    pred = model.predict(X_test_scaled)
    results.append(evaluate_model(name, y_test, pred))

# Create results DataFrame
results_df = pd.DataFrame(results)
print("\n📊 Model Performance Comparison:")
print("-" * 80)
print(results_df.to_string(index=False))

# --------------------------------------------------------------------------
# 8.1 Random Forest - Detailed Evaluation
# --------------------------------------------------------------------------
print("\n" + "="*60)
print("🔍 RANDOM FOREST - DETAILED EVALUATION")
print("="*60)

print("\n📌 Classification Report:")
print("-" * 40)
print(classification_report(y_test, rf_pred, target_names=['No Dropout', 'Dropout']))

print("\n📌 Confusion Matrix:")
print("-" * 40)
cm = confusion_matrix(y_test, rf_pred)
tn, fp, fn, tp = cm.ravel()
print(f"   True Negatives: {tn}")
print(f"   False Positives: {fp}")
print(f"   False Negatives: {fn}")
print(f"   True Positives: {tp}")
print(f"\n   Sensitivity (Recall): {tp/(tp+fn):.4f}")
print(f"   Specificity: {tn/(tn+fp):.4f}")
print(f"   Precision: {tp/(tp+fp):.4f}")

# --------------------------------------------------------------------------
# 8.2 Cross-Validation
# --------------------------------------------------------------------------
print("\n📌 Cross-Validation (5-fold):")
print("-" * 40)
cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=5, scoring='accuracy')
cv_f1 = cross_val_score(rf, X_train_scaled, y_train, cv=5, scoring='f1_macro')
print(f"   CV Accuracy: {cv_scores}")
print(f"   Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
print(f"   Mean CV F1-Score: {cv_f1.mean():.4f}")

# --------------------------------------------------------------------------
# 8.3 Feature Importance
# --------------------------------------------------------------------------
print("\n📌 Feature Importance (Top 10):")
print("-" * 40)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance.head(10).to_string(index=False))

# ============================================================================
# 9. SAVE MODEL AND PREPROCESSORS
# ============================================================================

print("\n" + "="*60)
print("💾 SAVING MODEL")
print("="*60)

# Ensure models directory exists before saving
os.makedirs("models", exist_ok=True)

# Save the best model (Random Forest)
joblib.dump(rf, "models/student_dropout_model.pkl")
print("   ✅ Model saved as 'models/student_dropout_model.pkl'")

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")
print("   ✅ Scaler saved as 'models/scaler.pkl'")

# Save feature names
joblib.dump(X.columns.tolist(), "models/feature_names.pkl")
print("   ✅ Feature names saved as 'models/feature_names.pkl'")

# Save model metadata
metadata = {
    'model_type': 'Random Forest (Tuned)',
    'features': X.columns.tolist(),
    'n_features': X.shape[1],
    'n_classes': len(np.unique(y)),
    'class_names': ['No Dropout', 'Dropout'],
    'accuracy': results_df.loc[2, 'Accuracy'],
    'precision': results_df.loc[2, 'Precision'],
    'recall': results_df.loc[2, 'Recall'],
    'f1_score': results_df.loc[2, 'F1-Score'],
    'cv_accuracy_mean': cv_scores.mean(),
    'feature_importance': rf.feature_importances_.tolist(),
    'best_params': grid_search.best_params_
}
joblib.dump(metadata, "models/model_metadata.pkl")
print("   ✅ Model metadata saved as 'models/model_metadata.pkl'")

# ============================================================================
# 10. TEST PREDICTION
# ============================================================================

print("\n" + "="*60)
print("🔮 TEST PREDICTION")
print("="*60)

# Take first student as sample
sample = X.iloc[[0]]
sample_scaled = scaler.transform(sample)
prediction = rf.predict(sample_scaled)[0]
probability = rf.predict_proba(sample_scaled)[0][1]

print(f"\n📌 Sample Student Features:")
for col in X.columns:
    print(f"   {col}: {sample.iloc[0][col]}")

print(f"\n📌 Prediction Result:")
print(f"   Status: {'⚠️ At Dropout Risk' if prediction == 1 else '✅ Safe'}")
print(f"   Dropout Probability: {probability:.2%}")
print(f"   Safe Probability: {1-probability:.2%}")

# ============================================================================
# 11. SUMMARY
# ============================================================================

print("\n" + "="*60)
print("✅ TRAINING COMPLETE")
print("="*60)

print(f"\n📊 Best Model: {metadata['model_type']}")
print(f"   Test Accuracy: {metadata['accuracy']:.4f}")
print(f"   Test F1-Score: {metadata['f1_score']:.4f}")
print(f"   CV Accuracy: {metadata['cv_accuracy_mean']:.4f}")

print(f"\n📁 Files Saved:")
print("   1. models/student_dropout_model.pkl - Trained model")
print("   2. models/scaler.pkl - StandardScaler for features")
print("   3. models/feature_names.pkl - Feature names in correct order")
print("   4. models/model_metadata.pkl - Model performance metrics")
