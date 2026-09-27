import os
import sys
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List

# Safe Windows console encoding
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Initialize FastAPI App
app = FastAPI(
    title="DiaGuard AI Inference API",
    description="Production-grade API for Diabetes Risk Assessment & Counterfactual Lifestyle Guidance based on CDC BRFSS epidemiological data.",
    version="1.0.0"
)

# Load Model Bundle
MODEL_PATH = os.path.join("models", "diaguard_model.joblib")
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Trained model not found at {MODEL_PATH}. Run src/train.py first.")

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
feature_names = bundle["feature_names"]
feature_importances = bundle["feature_importances"]

class PatientVitals(BaseModel):
    HighBP: int = Field(..., ge=0, le=1, description="High Blood Pressure (1 = Yes, 0 = No)")
    HighChol: int = Field(..., ge=0, le=1, description="High Cholesterol (1 = Yes, 0 = No)")
    CholCheck: int = Field(1, ge=0, le=1, description="Cholesterol check in past 5 years (1 = Yes, 0 = No)")
    BMI: float = Field(..., ge=10.0, le=75.0, description="Body Mass Index (e.g. 24.5)")
    Smoker: int = Field(0, ge=0, le=1, description="Smoked at least 100 cigarettes in entire life (1 = Yes, 0 = No)")
    Stroke: int = Field(0, ge=0, le=1, description="Ever had a stroke (1 = Yes, 0 = No)")
    HeartDiseaseorAttack: int = Field(0, ge=0, le=1, description="Coronary heart disease or heart attack (1 = Yes, 0 = No)")
    PhysActivity: int = Field(1, ge=0, le=1, description="Physical activity in past 30 days (1 = Yes, 0 = No)")
    Fruits: int = Field(1, ge=0, le=1, description="Consume fruit 1 or more times per day (1 = Yes, 0 = No)")
    Veggies: int = Field(1, ge=0, le=1, description="Consume vegetables 1 or more times per day (1 = Yes, 0 = No)")
    HvyAlcoholConsump: int = Field(0, ge=0, le=1, description="Heavy alcohol consumption (1 = Yes, 0 = No)")
    AnyHealthcare: int = Field(1, ge=0, le=1, description="Has any health care coverage (1 = Yes, 0 = No)")
    NoDocbcCost: int = Field(0, ge=0, le=1, description="Could not see doctor due to cost (1 = Yes, 0 = No)")
    GenHlth: int = Field(2, ge=1, le=5, description="General health scale: 1=Excellent, 2=Very Good, 3=Good, 4=Fair, 5=Poor")
    MentHlth: int = Field(0, ge=0, le=30, description="Days with poor mental health (0-30 days)")
    PhysHlth: int = Field(0, ge=0, le=30, description="Days with physical illness/injury (0-30 days)")
    DiffWalk: int = Field(0, ge=0, le=1, description="Serious difficulty walking or climbing stairs (1 = Yes, 0 = No)")
    Sex: int = Field(..., ge=0, le=1, description="Biological Sex (0 = Female, 1 = Male)")
    Age: int = Field(..., ge=1, le=13, description="13-level age category (1=18-24, 7=50-54, 9=60-64, 13=80+)")
    Education: int = Field(4, ge=1, le=6, description="Education scale (1-6)")
    Income: int = Field(5, ge=1, le=8, description="Income scale (1-8)")

def determine_risk_tier(prob: float) -> str:
    if prob < 0.25:
        return "Low Risk"
    elif prob < 0.50:
        return "Moderate Risk"
    elif prob < 0.75:
        return "Elevated Risk"
    else:
        return "High Risk"

@app.get("/health")
def health_check():
    """Health check endpoint for Kubernetes / Docker liveness probes."""
    return {
        "status": "healthy",
        "service": "DiaGuard AI Inference Engine",
        "model_version": "1.0.0",
        "features_loaded": len(feature_names),
        "test_roc_auc": bundle["metrics"]["roc_auc"]
    }

@app.post("/api/v1/predict")
def predict_diabetes_risk(vitals: PatientVitals):
    """
    Takes patient health indicators, validates constraints, and returns
    calibrated diabetes risk probability, clinical category, and top contributing factors.
    """
    input_data = vitals.model_dump()
    df_input = pd.DataFrame([input_data])[feature_names]
    
    # Calculate probability
    prob = float(model.predict_proba(df_input)[0, 1])
    is_diabetic_risk = bool(prob >= 0.50)
    risk_tier = determine_risk_tier(prob)
    
    # Extract top patient-specific active risk indicators
    patient_risk_factors = []
    if input_data["HighBP"] == 1:
        patient_risk_factors.append("High Blood Pressure")
    if input_data["HighChol"] == 1:
        patient_risk_factors.append("High Cholesterol")
    if input_data["BMI"] >= 30.0:
        patient_risk_factors.append(f"Elevated BMI ({input_data['BMI']})")
    if input_data["GenHlth"] >= 4:
        patient_risk_factors.append("Sub-optimal General Health")
    if input_data["DiffWalk"] == 1:
        patient_risk_factors.append("Mobility Limitations (Difficulty Walking)")
    if input_data["Smoker"] == 1:
        patient_risk_factors.append("Tobacco Use")
    if input_data["PhysActivity"] == 0:
        patient_risk_factors.append("Sedentary Lifestyle (Lack of Physical Activity)")

    return {
        "diabetes_probability": round(prob, 4),
        "risk_percentage": f"{prob * 100:.1f}%",
        "risk_tier": risk_tier,
        "classification": "Diabetic / Pre-diabetic Tendency" if is_diabetic_risk else "Healthy / Low Risk",
        "active_risk_factors": patient_risk_factors if patient_risk_factors else ["No major high-risk indicators flagged"],
        "top_model_drivers": list(feature_importances.keys())[:5]
    }

@app.post("/api/v1/simulate-what-if")
def simulate_counterfactual(current_vitals: PatientVitals, target_vitals: PatientVitals):
    """
    Counterfactual Guidance Simulator:
    Compares patient's current risk with hypothetical lifestyle changes
    (e.g., losing weight, lowering BP, exercising) and returns exact risk reduction.
    """
    df_curr = pd.DataFrame([current_vitals.model_dump()])[feature_names]
    df_targ = pd.DataFrame([target_vitals.model_dump()])[feature_names]
    
    curr_prob = float(model.predict_proba(df_curr)[0, 1])
    targ_prob = float(model.predict_proba(df_targ)[0, 1])
    
    risk_difference = curr_prob - targ_prob
    relative_reduction = (risk_difference / curr_prob * 100) if curr_prob > 0 else 0.0
    
    return {
        "current_risk": f"{curr_prob * 100:.1f}%",
        "simulated_risk": f"{targ_prob * 100:.1f}%",
        "absolute_reduction": f"{max(0.0, risk_difference * 100):.1f}%",
        "relative_improvement": f"{max(0.0, relative_reduction):.1f}%",
        "current_tier": determine_risk_tier(curr_prob),
        "simulated_tier": determine_risk_tier(targ_prob)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
