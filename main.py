from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

# Load models
diabetes_model = joblib.load("diabetes_pipeline.pkl")
heart_model = joblib.load("heart_pipeline.pkl")
stroke_model = joblib.load("stroke_pipeline.pkl")

app = FastAPI()

# -------------------------
# SIMPLE CORS (Allow All)
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Allow all domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# User Input (10 Fields)
# -------------------------
class UserInput(BaseModel):
    age: int
    gender: str
    glucose: float
    blood_pressure: float
    bmi: float
    physical_activity: str
    smoking_status: str
    family_history: bool
    chest_pain: bool
    stress_level: str

# -------------------------
# Severity Calculation
# -------------------------
def compute_severity(diabetes_prob, heart_prob, stroke_prob):
    score = (2 * stroke_prob + 1.5 * heart_prob + 1 * diabetes_prob) / 4.5
    score_10 = score * 10

    if score_10 < 3:
        level = "Mild"
    elif score_10 < 7:
        level = "Moderate"
    else:
        level = "Severe"

    return round(score_10, 2), level

# -------------------------
# API Endpoint
# -------------------------
@app.post("/predict")
def predict(input: UserInput):

    # -------- Diabetes features --------
    diabetes_features = {
        "Pregnancies": 0,
        "Glucose": input.glucose,
        "BloodPressure": input.blood_pressure,
        "SkinThickness": 20,
        "Insulin": 80,
        "BMI": input.bmi,
        "DiabetesPedigreeFunction": 0.3 if input.family_history else 0.1,
        "Age": input.age,
    }

    # -------- Heart features --------
    heart_features = {
        "age": input.age,
        "sex": 1 if input.gender.lower() == "male" else 0,
        "cp": 1 if input.chest_pain else 0,
        "trestbps": input.blood_pressure,
        "chol": 200,
        "fbs": 1 if input.glucose > 126 else 0,
        "restecg": 1,
        "thalach": 200 - input.age,
        "exang": 1 if input.stress_level == "high" else 0,
        "oldpeak": 1.0,
        "slope": 1,
        "ca": 0,
        "thal": 2
    }

    # -------- Stroke features --------
    stroke_features = {
        "gender": input.gender,
        "age": input.age,
        "hypertension": 1 if input.blood_pressure > 140 else 0,
        "heart_disease": 1 if input.chest_pain else 0,
        "ever_married": "Yes",
        "work_type": "Private",
        "Residence_type": "Urban",
        "avg_glucose_level": input.glucose,
        "bmi": input.bmi,
        "smoking_status": input.smoking_status
    }

    # -------- Predictions --------
    diabetes_prob = float(diabetes_model.predict_proba([diabetes_features])[0][1])
    heart_prob = float(heart_model.predict_proba([heart_features])[0][1])
    stroke_prob = float(stroke_model.predict_proba([stroke_features])[0][1])

    diabetes = int(diabetes_prob >= 0.5)
    heart = int(heart_prob >= 0.5)
    stroke = int(stroke_prob >= 0.5)

    # -------- Severity --------
    severity_score, severity_level = compute_severity(
        diabetes_prob, heart_prob, stroke_prob
    )

    # -------- Mild condition if no disease --------
    suggestion = None
    if diabetes == 0 and heart == 0 and stroke == 0:
        suggestion = "Likely minor issue (e.g., acidity, dehydration, viral cold)."

    return {
        "diabetes": diabetes,
        "diabetes_probability": diabetes_prob,
        "heart_disease": heart,
        "heart_probability": heart_prob,
        "stroke": stroke,
        "stroke_probability": stroke_prob,
        "severity_score": severity_score,
        "severity_level": severity_level,
        "mild_condition_suggestion": suggestion,
    }
