import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier

# Safe Windows console encoding
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def train_diaguard_model(data_path="data/raw/cdc_diabetes_data.csv", output_dir="models"):
    """
    Trains an XGBoost Classifier on the CDC Diabetes dataset with class weighting,
    evaluates performance on a holdout test set, and exports the model package.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from: {data_path}...")
    df = pd.read_csv(data_path)
    
    target_col = "Diabetes_binary"
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    
    feature_names = list(X.columns)
    print(f"Total rows: {len(df):,}, Features: {len(feature_names)}")
    
    # Stratified 80/20 train-test split
    print("Splitting into Train (80%) and Test (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Calculate class weight to account for 14% positive class ratio
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    pos_weight = neg_count / pos_count
    print(f"Negative samples: {neg_count:,}, Positive samples: {pos_count:,}")
    print(f"Using scale_pos_weight: {pos_weight:.2f}")
    
    print("\nTraining XGBoost Classifier...")
    model = XGBClassifier(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.08,
        scale_pos_weight=pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("Model training complete!")
    
    # Evaluation
    print("\nEvaluating on holdout test set (50,736 patients)...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"⭐ ROC-AUC Score: {roc_auc:.4f}")
    print(f"Accuracy:        {acc:.4f}")
    print(f"Recall:          {rec:.4f}  (Critical for disease detection: catches {rec*100:.1f}% of diabetic cases!)")
    print(f"Precision:       {prec:.4f}")
    print(f"F1-Score:        {f1:.4f}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Healthy", "Diabetic / Pre-diabetic"]))
    
    # Feature Importances
    importances = model.feature_importances_
    feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    
    print("\nTop 5 Most Influential Risk Factors:")
    for rank, (feat, score) in enumerate(feat_imp[:5], 1):
        print(f"  {rank}. {feat}: {score*100:.2f}%")
        
    # Save Model Artifact Bundle
    model_save_path = os.path.join(output_dir, "diaguard_model.joblib")
    metrics_save_path = os.path.join(output_dir, "evaluation_metrics.json")
    
    model_package = {
        "model": model,
        "feature_names": feature_names,
        "threshold": 0.5,
        "feature_importances": dict(feat_imp),
        "target_col": target_col,
        "metrics": {
            "roc_auc": round(float(roc_auc), 4),
            "accuracy": round(float(acc), 4),
            "recall": round(float(rec), 4),
            "precision": round(float(prec), 4),
            "f1_score": round(float(f1), 4)
        }
    }
    
    joblib.dump(model_package, model_save_path)
    print(f"\nModel bundle saved to: {model_save_path}")
    
    with open(metrics_save_path, "w") as f:
        json.dump(model_package["metrics"], f, indent=2)
    print(f"Metrics saved to: {metrics_save_path}")
    
    return model_package

if __name__ == "__main__":
    train_diaguard_model()
