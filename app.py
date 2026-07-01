from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load model and scaler
model = joblib.load("student_dropout_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        def safe_float(key, default=0.0):
            val = request.form.get(key, "").strip()
            return float(val) if val else default
        
        def safe_int(key, default=0):
            val = request.form.get(key, "").strip()
            return int(float(val)) if val else default
        
        data = pd.DataFrame({
            "Student_ID": [safe_int("id")],
            "Age": [safe_float("age", 20.0)],
            "Gender": [safe_int("gender")],
            "Family_Income": [safe_float("income", 30000.0)],
            "Internet_Access": [safe_int("internet", 1)],
            "Study_Hours_per_Day": [safe_float("study", 4.0)],
            "Attendance_Rate": [safe_float("attendance", 80.0)],
            "Assignment_Delay_Days": [safe_int("delay")],
            "Travel_Time_Minutes": [safe_float("travel", 25.0)],
            "Part_Time_Job": [safe_int("job")],
            "Scholarship": [safe_int("scholarship")],
            "Stress_Index": [safe_float("stress", 5.0)],
            "GPA": [safe_float("gpa", 2.5)],
            "Semester_GPA": [safe_float("sgpa", 2.5)],
            "CGPA": [safe_float("cgpa", 2.5)],
            "Semester": [safe_int("semester", 1)],
            "Department": [safe_int("department")],
            "Parental_Education": [safe_int("parent", 1)]
        })
        
        data_scaled = scaler.transform(data)
        result = model.predict(data_scaled)
        probability = model.predict_proba(data_scaled)[0][1]
        
        if result[0] == 1:
            message = f"⚠️ Student is at Risk of Dropout (Confidence: {probability:.2%})"
        else:
            message = f"✅ Student will Continue (Confidence: {1-probability:.2%})"
        
        return render_template("index.html", prediction=message)
    
    except Exception as e:
        return render_template("index.html", prediction=f"❌ Error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
