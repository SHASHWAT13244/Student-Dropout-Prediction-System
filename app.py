from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import traceback

app = Flask(__name__)
app.secret_key = os.urandom(24)

# ============================================================================
# 0. CREATE MODELS DIRECTORY
# ============================================================================

os.makedirs("models", exist_ok=True)

# ============================================================================
# 1. LOAD MODEL AND PREPROCESSORS
# ============================================================================

print("="*60)
print("🚀 LOADING STUDENT DROPOUT PREDICTION SYSTEM")
print("="*60)

# Try loading from models directory first, then root directory
model_paths = [
    "models/student_dropout_model.pkl",
    "student_dropout_model.pkl"
]

scaler_paths = [
    "models/scaler.pkl",
    "scaler.pkl"
]

feature_paths = [
    "models/feature_names.pkl",
    "feature_names.pkl"
]

metadata_paths = [
    "models/model_metadata.pkl",
    "model_metadata.pkl"
]

# Load model
model = None
for path in model_paths:
    try:
        model = joblib.load(path)
        print(f"✅ Model loaded from: {path}")
        break
    except FileNotFoundError:
        continue

if model is None:
    print("❌ Model file not found! Please run train_model.py first")

# Load scaler
scaler = None
for path in scaler_paths:
    try:
        scaler = joblib.load(path)
        print(f"✅ Scaler loaded from: {path}")
        break
    except FileNotFoundError:
        continue

if scaler is None:
    print("❌ Scaler file not found! Please run train_model.py first")

# Load feature names
feature_names = None
for path in feature_paths:
    try:
        feature_names = joblib.load(path)
        print(f"✅ Feature names loaded from: {path}")
        break
    except FileNotFoundError:
        continue

if feature_names is None:
    # Fallback to default feature names
    feature_names = [
        'Age', 'Gender', 'Family_Income', 'Internet_Access',
        'Study_Hours_per_Day', 'Attendance_Rate', 'Assignment_Delay_Days',
        'Travel_Time_Minutes', 'Part_Time_Job', 'Scholarship',
        'Stress_Index', 'GPA', 'Semester_GPA', 'CGPA',
        'Semester', 'Department', 'Parental_Education'
    ]
    print("⚠️ Using default feature names")

# Load metadata
metadata = None
for path in metadata_paths:
    try:
        metadata = joblib.load(path)
        print(f"✅ Metadata loaded from: {path}")
        print(f"   Model Type: {metadata['model_type']}")
        print(f"   Accuracy: {metadata['accuracy']:.4f}")
        print(f"   F1-Score: {metadata['f1_score']:.4f}")
        break
    except FileNotFoundError:
        continue

if metadata is None:
    print("⚠️ Model metadata not found")

print("="*60)

# ============================================================================
# 2. HELPER FUNCTIONS
# ============================================================================

def safe_float(value, default=0.0):
    """Safely convert value to float"""
    try:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return default
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value, default=0):
    """Safely convert value to int"""
    try:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return default
        return int(float(value))
    except (ValueError, TypeError):
        return default

def prepare_prediction_data(form_data):
    """
    Prepare form data for model prediction
    
    Parameters:
    -----------
    form_data : dict
        Form data from request
    
    Returns:
    --------
    pd.DataFrame: Prepared data for prediction
    """
    # Get form values with safe defaults
    data_dict = {
        "Age": [safe_float(form_data.get("age", 20.0))],
        "Gender": [safe_int(form_data.get("gender", 1))],
        "Family_Income": [safe_float(form_data.get("income", 30000.0))],
        "Internet_Access": [safe_int(form_data.get("internet", 1))],
        "Study_Hours_per_Day": [safe_float(form_data.get("study", 4.0))],
        "Attendance_Rate": [safe_float(form_data.get("attendance", 80.0))],
        "Assignment_Delay_Days": [safe_int(form_data.get("delay", 0))],
        "Travel_Time_Minutes": [safe_float(form_data.get("travel", 25.0))],
        "Part_Time_Job": [safe_int(form_data.get("job", 0))],
        "Scholarship": [safe_int(form_data.get("scholarship", 0))],
        "Stress_Index": [safe_float(form_data.get("stress", 5.0))],
        "GPA": [safe_float(form_data.get("gpa", 2.5))],
        "Semester_GPA": [safe_float(form_data.get("sgpa", 2.5))],
        "CGPA": [safe_float(form_data.get("cgpa", 2.5))],
        "Semester": [safe_int(form_data.get("semester", 1))],
        "Department": [safe_int(form_data.get("department", 0))],
        "Parental_Education": [safe_int(form_data.get("parent", 1))]
    }
    
    # Create DataFrame with correct feature order
    df = pd.DataFrame(data_dict)
    
    # Ensure columns match feature order
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0
    
    df = df[feature_names]
    
    return df

def get_risk_level(probability):
    """Determine risk level based on probability"""
    if probability >= 0.7:
        return 'High Risk'
    elif probability >= 0.4:
        return 'Medium Risk'
    else:
        return 'Low Risk'

def get_recommendations(prediction, probability):
    """Generate recommendations based on prediction"""
    recommendations = []
    
    if prediction == 1:
        recommendations.append("📚 Schedule academic counseling session")
        recommendations.append("📅 Monitor attendance regularly")
        recommendations.append("🧘 Refer to student wellness program")
        
        if probability >= 0.7:
            recommendations.append("🚨 Immediate intervention recommended")
            recommendations.append("👨‍👩‍👧 Contact parents/guardians")
    else:
        recommendations.append("✅ Student is on track")
        recommendations.append("📈 Continue monitoring progress")
        recommendations.append("🎯 Encourage maintaining current performance")
    
    return recommendations

# ============================================================================
# 3. ROUTES
# ============================================================================

@app.route("/")
def home():
    """Home page with prediction form"""
    return render_template(
        "index.html",
        prediction=None,
        metadata=metadata
    )

@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction request"""
    try:
        # Get form data
        form_data = request.form
        
        # Prepare data for prediction
        df = prepare_prediction_data(form_data)
        
        # Check if model and scaler are loaded
        if model is None or scaler is None:
            return render_template(
                "index.html",
                prediction={"error": "Model not loaded. Please run train_model.py first."},
                metadata=metadata
            )
        
        # Scale features
        scaled_data = scaler.transform(df)
        
        # Make prediction
        prediction = model.predict(scaled_data)[0]
        probabilities = model.predict_proba(scaled_data)[0]
        
        # Get probabilities
        prob_dropout = probabilities[1]
        prob_safe = probabilities[0]
        
        # Get student data for display
        student_data = df.iloc[0].to_dict()
        
        # Get risk level
        risk_level = get_risk_level(prob_dropout)
        
        # Get recommendations
        recommendations = get_recommendations(prediction, prob_dropout)
        
        # Prepare result
        result = {
            'prediction': int(prediction),
            'prediction_label': 'At Risk' if prediction == 1 else 'Safe',
            'probability_dropout': float(prob_dropout),
            'probability_safe': float(prob_safe),
            'risk_level': risk_level,
            'recommendations': recommendations,
            'student_data': student_data
        }
        
        # Render template with results
        return render_template(
            "index.html",
            prediction=result,
            metadata=metadata
        )
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(f"Error: {traceback.format_exc()}")
        
        return render_template(
            "index.html",
            prediction={'error': error_msg},
            metadata=metadata
        )

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    API endpoint for programmatic predictions
    Expects JSON with student features
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Prepare data
        df = prepare_prediction_data(data)
        
        # Check if model and scaler are loaded
        if model is None or scaler is None:
            return jsonify({'error': 'Model not loaded'}), 503
        
        # Scale features
        scaled_data = scaler.transform(df)
        
        # Make prediction
        prediction = model.predict(scaled_data)[0]
        probabilities = model.predict_proba(scaled_data)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'prediction_label': 'Dropout' if prediction == 1 else 'Safe',
            'probability_dropout': float(probabilities[1]),
            'probability_safe': float(probabilities[0]),
            'risk_level': get_risk_level(probabilities[1])
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if model is not None else 'unhealthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'metadata_available': metadata is not None
    })

@app.route("/metadata")
def get_metadata():
    """Get model metadata"""
    if metadata:
        return jsonify(metadata)
    return jsonify({'error': 'Metadata not available'}), 404

# ============================================================================
# 4. MAIN - UPDATED FOR RENDER DEPLOYMENT
# ============================================================================

if __name__ == "__main__":
    # Get port from environment variable (Render sets this automatically)
    port = int(os.environ.get('PORT', 5000))
    
    # Don't auto-open browser in production
    # Only open browser if running locally
    if os.environ.get('FLASK_ENV') != 'production':
        import webbrowser
        webbrowser.open(f"http://127.0.0.1:{port}")
    
    app.run(
        debug=False,  # Always False in production
        host='0.0.0.0',  # Required for Render
        port=port
    )
