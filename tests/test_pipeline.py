import os
import pytest
import joblib
import pandas as pd
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_model_artifact_exists():
    """Verify that the model artifact is saved and contains expected metadata."""
    model_path = os.path.join("models", "diaguard_model.joblib")
    assert os.path.exists(model_path), "Model artifact does not exist!"
    
    bundle = joblib.load(model_path)
    assert "model" in bundle
    assert "feature_names" in bundle
    assert len(bundle["feature_names"]) == 21
    assert "metrics" in bundle
    assert bundle["metrics"]["roc_auc"] > 0.75

def test_api_health():
    """Verify health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_version"] == "1.0.0"

def test_api_predict_valid():
    """Verify prediction endpoint with valid patient inputs."""
    payload = {
        "HighBP": 1,
        "HighChol": 1,
        "CholCheck": 1,
        "BMI": 32.5,
        "Smoker": 1,
        "Stroke": 0,
        "HeartDiseaseorAttack": 0,
        "PhysActivity": 0,
        "Fruits": 0,
        "Veggies": 1,
        "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1,
        "NoDocbcCost": 0,
        "GenHlth": 4,
        "MentHlth": 5,
        "PhysHlth": 10,
        "DiffWalk": 1,
        "Sex": 1,
        "Age": 9,
        "Education": 4,
        "Income": 5
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "diabetes_probability" in data
    assert 0.0 <= data["diabetes_probability"] <= 1.0
    assert "risk_tier" in data
    assert "active_risk_factors" in data
    assert len(data["active_risk_factors"]) > 0

def test_api_predict_invalid_data():
    """Verify that Pydantic rejects invalid biological data (e.g. negative BMI)."""
    invalid_payload = {
        "HighBP": 1,
        "HighChol": 0,
        "BMI": -10.0, # Impossible BMI
        "Sex": 1,
        "Age": 5
    }
    response = client.post("/api/v1/predict", json=invalid_payload)
    assert response.status_code == 422 # Pydantic validation error

def test_api_what_if_simulator():
    """Verify that simulated healthy interventions reduce risk."""
    current = {
        "HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 34.0, "Smoker": 1,
        "Stroke": 0, "HeartDiseaseorAttack": 0, "PhysActivity": 0, "Fruits": 0,
        "Veggies": 0, "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0,
        "GenHlth": 4, "MentHlth": 0, "PhysHlth": 0, "DiffWalk": 1, "Sex": 1,
        "Age": 9, "Education": 4, "Income": 5
    }
    # Simulate: Normalized BP, lower BMI, and daily exercise
    simulated = current.copy()
    simulated["HighBP"] = 0
    simulated["BMI"] = 25.0
    simulated["PhysActivity"] = 1
    simulated["GenHlth"] = 2
    
    response = client.post("/api/v1/simulate-what-if", json={
        "current_vitals": current,
        "target_vitals": simulated
    })
    assert response.status_code == 200
    data = response.json()
    assert float(data["current_risk"].replace("%", "")) > float(data["simulated_risk"].replace("%", ""))
