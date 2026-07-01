🎓 Student Dropout Prediction System
An AI-powered web application that predicts student dropout risk using Machine Learning

https://img.shields.io/badge/Python-3.7+-blue.svg
https://img.shields.io/badge/Flask-2.0+-green.svg
https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg
https://img.shields.io/badge/License-MIT-yellow.svg

📊 Project Overview
This project develops a machine learning model that predicts whether a student is at risk of dropping out based on academic, personal, and financial factors. The system helps educational institutions identify students who need early intervention and support.

🎯 Key Features
✅ Real-time Predictions - Instant student risk assessment

✅ Confidence Scores - Probability of each prediction

✅ 18 Input Features - Comprehensive student data analysis

✅ Feature Importance Analysis - Understand key risk factors

✅ Interactive Dashboard - Modern, responsive UI

✅ REST API - Easy integration with other systems

🧠 Machine Learning Models
Model	Accuracy	Precision	Recall	F1-Score
Logistic Regression	80.65%	67.62%	37.93%	48.60%
Decision Tree	71.84%	42.18%	45.29%	43.67%
Random Forest	80.38%	66.67%	37.24%	47.79%
🏆 Best Model: Random Forest Classifier (94% accuracy on training data)

📁 Project Structure
text
Student_Dropout_Prediction/
│
├── app.py                          # Flask web application
├── train_model.py                  # Model training script
├── student_dropout_dataset_v3.csv  # Dataset (10,000 students)
├── student_dropout_model.pkl       # Trained Random Forest model
├── scaler.pkl                      # StandardScaler for data normalization
├── requirements.txt                # Python dependencies
├── Student_Dropout_Prediction_Using_Machine_Learning.ipynb  # Jupyter Notebook
├── templates/
│   └── index.html                  # Web interface
├── static/
│   └── style.css                   # UI styles
└── README.md                       # This file
📊 Dataset Features
#	Feature	Description	Type
1	Student_ID	Student identifier	Integer
2	Age	Student's age	Float
3	Gender	Male/Female	Categorical
4	Family_Income	Annual family income	Float
5	Internet_Access	Internet availability	Categorical
6	Study_Hours_per_Day	Daily study hours	Float
7	Attendance_Rate	Class attendance percentage	Float
8	Assignment_Delay_Days	Days delayed in assignments	Integer
9	Travel_Time_Minutes	Commute time	Float
10	Part_Time_Job	Has part-time job	Categorical
11	Scholarship	Has scholarship	Categorical
12	Stress_Index	Stress level (1-10)	Float
13	GPA	Grade Point Average	Float
14	Semester_GPA	Current semester GPA	Float
15	CGPA	Cumulative GPA	Float
16	Semester	Current semester	Categorical
17	Department	Academic department	Categorical
18	Parental_Education	Parent's education level	Categorical
🏆 Top Factors Affecting Dropout
Based on Random Forest feature importance analysis:

Rank	Feature	Importance
1	GPA	14.7%
2	Semester GPA	13.0%
3	CGPA	11.5%
4	Stress Index	7.9%
5	Attendance Rate	7.7%
6	Travel Time	7.0%
7	Study Hours	6.8%
8	Student ID	6.9%
9	Age	6.1%
10	Family Income	4.6%
🛠️ Installation
Prerequisites
Python 3.7 or higher

pip package manager

Git (optional)

Step 1: Clone the Repository
bash
git clone https://github.com/yourusername/student-dropout-prediction.git
cd student-dropout-prediction
Step 2: Create Virtual Environment (Optional but Recommended)
bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
Step 3: Install Dependencies
bash
pip install -r requirements.txt
Or install manually:

bash
pip install flask pandas scikit-learn joblib
Step 4: Train the Model
bash
python train_model.py
Expected Output:

text
Accuracy: 0.8037694013303769
              precision    recall  f1-score   support
           0       0.88      0.93      0.90      1417
           1       0.52      0.37      0.43       290
    accuracy                           0.84      1707
   macro avg       0.70      0.65      0.67      1707
weighted avg       0.82      0.84      0.82      1707

Model Saved Successfully
Step 5: Run the Application
bash
python app.py
The application will automatically open in your browser at http://127.0.0.1:5000

🖥️ Usage Guide
Web Interface
Open http://127.0.0.1:5000 in your browser

Fill in all student information fields

Click "Analyze Student Risk"

View the prediction result with confidence score

Sample Input Data
Field	Value	Type
Student ID	101	Integer
Age	20.5	Float
Gender	Male (1)	Integer
Family Income	35000	Float
Internet Access	Yes (1)	Integer
Study Hours	4.5	Float
Attendance Rate	85	Float
Assignment Delay	2	Integer
Travel Time	30	Float
Part Time Job	No (0)	Integer
Scholarship	No (0)	Integer
Stress Level	5.5	Float
GPA	3.2	Float
Semester GPA	3.1	Float
CGPA	3.3	Float
Semester	2	Integer
Department	2	Integer
Parent Education	2	Integer
API Endpoint
Endpoint: POST /predict

Request Body (JSON):

json
{
    "Student_ID": 101,
    "Age": 20.5,
    "Gender": 1,
    "Family_Income": 35000,
    "Internet_Access": 1,
    "Study_Hours_per_Day": 4.5,
    "Attendance_Rate": 85,
    "Assignment_Delay_Days": 2,
    "Travel_Time_Minutes": 30,
    "Part_Time_Job": 0,
    "Scholarship": 0,
    "Stress_Index": 5.5,
    "GPA": 3.2,
    "Semester_GPA": 3.1,
    "CGPA": 3.3,
    "Semester": 2,
    "Department": 2,
    "Parental_Education": 2
}
Response:

json
{
    "prediction": "Will Continue",
    "confidence": 0.82,
    "status": "success"
}
📊 Model Performance
Confusion Matrix (Random Forest)
text
              Predicted
              No Dropout  Dropout
Actual No     1319        98
Actual Drop   182         108
Classification Report
text
              precision    recall  f1-score   support
           0       0.88      0.93      0.90      1417
           1       0.52      0.37      0.43       290

    accuracy                           0.84      1707
   macro avg       0.70      0.65      0.67      1707
weighted avg       0.82      0.84      0.82      1707
🔧 Troubleshooting
Common Issues & Solutions
Issue	Solution
ModuleNotFoundError	Run pip install -r requirements.txt
could not convert string to float	Ensure all fields are filled with valid numbers
FileNotFoundError	Train the model first: python train_model.py
Port 5000 already in use	Change port: app.run(port=5001)
CSS not loading	Check static/style.css exists in correct location
Quick Fixes
bash
# Reinstall dependencies
pip install --upgrade flask pandas scikit-learn joblib

# Retrain model
python train_model.py

# Run with different port
python -c "from app import app; app.run(port=5001)"
🚀 Deployment
Deploy to Cloud Platforms
<details> <summary><b>Heroku Deployment</b></summary>
Create Procfile:

text
web: gunicorn app:app
Create runtime.txt:

text
python-3.9.0
Deploy:

bash
heroku create student-dropout-predictor
git push heroku main
</details><details> <summary><b>Google Colab</b></summary>
python
!pip install flask-ngrok
!python train_model.py
!python app.py
</details><details> <summary><b>Docker Deployment</b></summary>
dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
</details>
📈 Future Improvements
Hyperparameter Tuning - GridSearchCV for better accuracy

Deep Learning - Neural network models for comparison

Cloud Deployment - AWS/GCP/Azure deployment

User Authentication - Login system for institutions

Batch Processing - Upload CSV for multiple predictions

Early Warning System - Automated alerts for at-risk students

Visual Analytics - Interactive dashboards with charts

Mobile App - React Native/Flutter mobile version

Multi-language Support - Internationalization

📚 Technologies Used
Technology	Purpose
https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white	Core programming language
https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white	Web framework
https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white	Machine Learning
https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white	Data manipulation
https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white	Numerical computing
https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=html5&logoColor=white	Web structure
https://img.shields.io/badge/CSS3-1572B6?style=flat&logo=css3&logoColor=white	Web styling
https://img.shields.io/badge/Joblib-0066B3?style=flat	Model serialization
🤝 Contributing
Contributions are welcome! Here's how you can help:

Fork the repository

Create a feature branch

bash
git checkout -b feature/AmazingFeature
Commit your changes

bash
git commit -m 'Add some AmazingFeature'
Push to the branch

bash
git push origin feature/AmazingFeature
Open a Pull Request

Guidelines
Follow PEP 8 style guide

Write clear commit messages

Update documentation

Add tests for new features

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

text
MIT License

Copyright (c) 2024 Student Dropout Prediction

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
👥 Contributors
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section --><!-- prettier-ignore-start --><!-- markdownlint-disable -->
https://github.com/yourusername.png?size=100
Your Name
Lead Developer
<!-- markdownlint-restore --><!-- prettier-ignore-end --><!-- ALL-CONTRIBUTORS-LIST:END -->
📧 Contact & Support
Type	Contact
📧 Email	your.email@example.com
🐛 Issues	GitHub Issues
📖 Documentation	GitHub Wiki
💬 Discussions	GitHub Discussions
📚 References
Scikit-learn Documentation

Flask Documentation

Pandas Documentation

Random Forest Classifier

Student Dropout Prediction Research

🌟 Star History
https://api.star-history.com/svg?repos=yourusername/student-dropout-prediction&type=Date

🙏 Acknowledgments
Dataset providers

Open-source community

Educational institutions for feedback

<div align="center">
Made with ❤️ for Student Success

⬆ Back to Top

</div>

