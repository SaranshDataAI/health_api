# main.py - COMPLETE FIXED VERSION
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import traceback
import os

# Check if model files exist
print("🔍 Checking for model files...")
model_files = ["diabetes_pipeline.pkl", "heart_pipeline.pkl", "stroke_pipeline.pkl"]
for file in model_files:
    if os.path.exists(file):
        print(f"✅ Found {file}")
    else:
        print(f"❌ Missing {file}")

# Load models with better error handling
try:
    diabetes_model = joblib.load("diabetes_pipeline.pkl")
    print("✅ Diabetes model loaded successfully")
except Exception as e:
    print(f"❌ Error loading diabetes model: {e}")
    diabetes_model = None

try:
    heart_model = joblib.load("heart_pipeline.pkl")
    print("✅ Heart model loaded successfully")
except Exception as e:
    print(f"❌ Error loading heart model: {e}")
    heart_model = None

try:
    stroke_model = joblib.load("stroke_pipeline.pkl")
    print("✅ Stroke model loaded successfully")
except Exception as e:
    print(f"❌ Error loading stroke model: {e}")
    stroke_model = None

app = FastAPI(title="Health Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.post("/predict")
def predict(input: UserInput):
    try:
        print(f"📥 Received input: {input.dict()}")

        # Check if models are loaded
        if not all([diabetes_model, heart_model, stroke_model]):
            raise HTTPException(
                status_code=500, 
                detail="One or more models failed to load. Check server logs."
            )

        print("🔍 Preparing features for prediction...")

        # DIABETES FEATURES
        diabetes_features = [[
            0,                          # Pregnancies
            input.glucose,              # Glucose
            input.blood_pressure,       # BloodPressure
            20,                         # SkinThickness
            80,                         # Insulin
            input.bmi,                  # BMI
            0.3 if input.family_history else 0.1,  # DiabetesPedigreeFunction
            input.age                   # Age
        ]]
        print(f"Diabetes features: {diabetes_features}")

        # HEART DISEASE FEATURES
        heart_features = [[
            input.age,                          # age
            1 if input.gender.lower() == "male" else 0,  # sex (1=male, 0=female)
            1 if input.chest_pain else 0,      # cp (chest pain)
            input.blood_pressure,               # trestbps
            200,                               # chol
            1 if input.glucose > 126 else 0,   # fbs
            0,                                 # restecg
            180 - (0.5 * input.age),           # thalach (max heart rate)
            1 if input.stress_level.lower() == "high" else 0,  # exang
            1.0,                               # oldpeak
            2,                                 # slope
            0,                                 # ca
            2                                  # thal
        ]]
        print(f"Heart features: {heart_features}")

        # STROKE FEATURES
        stroke_features = [[
            1 if input.gender.lower() == "male" else 0,  # gender (1=male, 0=female)
            input.age,                          # age
            1 if input.blood_pressure > 140 else 0,  # hypertension
            1 if input.chest_pain else 0,      # heart_disease
            1,                                 # ever_married (1=Yes)
            2,                                 # work_type (2=Private)
            1,                                 # Residence_type (1=Urban)
            input.glucose,                     # avg_glucose_level
            input.bmi,                         # bmi
            1 if input.smoking_status.lower() == "current" else 
            2 if input.smoking_status.lower() == "former" else 0  # smoking_status
        ]]
        print(f"Stroke features: {stroke_features}")

        print("🎯 Making predictions...")
        
        # Get predictions
        try:
            diabetes_proba = diabetes_model.predict_proba(diabetes_features)
            diabetes_prob = float(diabetes_proba[0][1])
            print(f"Diabetes probability: {diabetes_prob}")
        except Exception as e:
            print(f"❌ Diabetes prediction error: {e}")
            diabetes_prob = 0.1

        try:
            heart_proba = heart_model.predict_proba(heart_features)
            heart_prob = float(heart_proba[0][1])
            print(f"Heart probability: {heart_prob}")
        except Exception as e:
            print(f"❌ Heart prediction error: {e}")
            heart_prob = 0.1

        try:
            stroke_proba = stroke_model.predict_proba(stroke_features)
            stroke_prob = float(stroke_proba[0][1])
            print(f"Stroke probability: {stroke_prob}")
        except Exception as e:
            print(f"❌ Stroke prediction error: {e}")
            stroke_prob = 0.1

        # Convert to binary predictions
        diabetes = int(diabetes_prob >= 0.5)
        heart = int(heart_prob >= 0.5)
        stroke = int(stroke_prob >= 0.5)

        # Calculate severity
        severity_score, severity_level = compute_severity(diabetes_prob, heart_prob, stroke_prob)

        # Mild condition if no disease
        suggestion = None
        if diabetes == 0 and heart == 0 and stroke == 0:
            suggestion = "Likely minor issue (e.g., acidity, dehydration, viral cold)."

        result = {
            "diabetes": diabetes,
            "diabetes_probability": round(diabetes_prob, 4),
            "heart_disease": heart,
            "heart_probability": round(heart_prob, 4),
            "stroke": stroke,
            "stroke_probability": round(stroke_prob, 4),
            "severity_score": severity_score,
            "severity_level": severity_level,
            "mild_condition_suggestion": suggestion,
        }

        print(f"✅ Prediction successful: {result}")
        return result

    except Exception as e:
        print(f"❌ Error in prediction: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Health Prediction API is running!"}

@app.get("/health")
def health_check():
    models_loaded = all([diabetes_model, heart_model, stroke_model])
    return {
        "status": "healthy" if models_loaded else "degraded",
        "models_loaded": models_loaded,
        "diabetes_model": diabetes_model is not None,
        "heart_model": heart_model is not None,
        "stroke_model": stroke_model is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
