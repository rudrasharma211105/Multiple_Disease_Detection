import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# Load trained models
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, "../models")

model_diabetes = joblib.load(os.path.join(models_dir, "model_diabetes.pkl"))
model_heart = joblib.load(os.path.join(models_dir, "model_heart.pkl"))
model_cancer = joblib.load(os.path.join(models_dir, "model_cancer.pkl"))

# Title
st.title("🧠 Multi-Disease Prediction System")
st.write("Predict Diabetes, Heart Disease, or Cancer based on user health information.")

# Sidebar for disease selection
disease = st.sidebar.selectbox("Select Disease to Predict", ["Diabetes", "Heart Disease", "Cancer"])

# ================= Diabetes ==================
if disease == "Diabetes":
    st.header("🩸 Diabetes Prediction")

    Pregnancies = st.number_input("Pregnancies", 0, 20)
    Glucose = st.number_input("Glucose", 0.0)
    BloodPressure = st.number_input("Blood Pressure", 0.0)
    SkinThickness = st.number_input("Skin Thickness", 0.0)
    Insulin = st.number_input("Insulin", 0.0)
    BMI = st.number_input("BMI", 0.0)
    DPF = st.number_input("Diabetes Pedigree Function", 0.0)
    Age = st.number_input("Age", 1, 120)

    if st.button("Predict Diabetes"):
        input_data = pd.DataFrame([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                                Insulin, BMI, DPF, Age]],
                                columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
        result = model_diabetes.predict(input_data)[0]
        st.success("✅ Diabetic" if result == 1 else "❎ Not Diabetic")

# ================= Heart Disease ==================
elif disease == "Heart Disease":
    st.header("❤️ Heart Disease Prediction")

    Age = st.number_input("Age", 1, 120)
    Sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
    cp = st.number_input("Chest Pain Type (0–3)", 0, 3)
    trestbps = st.number_input("Resting Blood Pressure", 0.0)
    chol = st.number_input("Cholesterol", 0.0)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.number_input("Rest ECG (0–2)", 0, 2)
    thalach = st.number_input("Max Heart Rate Achieved", 0.0)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("Oldpeak (ST depression)", 0.0)
    slope = st.number_input("Slope of Peak Exercise ST Segment", 0, 2)
    ca = st.number_input("Number of Major Vessels (0–3)", 0, 3)
    thal = st.number_input("Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)", 1, 3)

    if st.button("Predict Heart Disease"):
        input_data = pd.DataFrame([[Age, Sex, cp, trestbps, chol, fbs, restecg,
                                thalach, exang, oldpeak, slope, ca, thal]],
                                columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
        result = model_heart.predict(input_data)[0]
        st.success("⚠️ Heart Disease Detected" if result == 1 else "💚 No Heart Disease")

# ================= Cancer ==================
elif disease == "Cancer":
    st.header("🧬 Cancer Prediction")

    Age = st.number_input("Age", 1, 120)
    Gender = st.selectbox("Gender (0 = Female, 1 = Male)", [0, 1])
    BMI = st.number_input("BMI", 0.0)
    Smoking = st.selectbox("Smoking (0 = No, 1 = Yes)", [0, 1])
    GeneticRisk = st.selectbox("Genetic Risk (0 = Low, 1 = Medium, 2 = High)", [0, 1, 2])
    PhysicalActivity = st.number_input("Physical Activity (hours/week)", 0.0)
    AlcoholIntake = st.number_input("Alcohol Intake (units/week)", 0.0)
    CancerHistory = st.selectbox("Cancer History (0 = No, 1 = Yes)", [0, 1])

    if st.button("Predict Cancer"):
        input_data = pd.DataFrame([[Age, Gender, BMI, Smoking, GeneticRisk,
                                PhysicalActivity, AlcoholIntake, CancerHistory]],
                                columns=['Age', 'Gender', 'BMI', 'Smoking', 'GeneticRisk', 'PhysicalActivity', 'AlcoholIntake', 'CancerHistory'])
        result = model_cancer.predict(input_data)[0]
        st.success("❗ Cancer Detected" if result == 1 else "✅  No Cancer")
