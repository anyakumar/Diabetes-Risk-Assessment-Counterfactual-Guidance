import os
import sys
import json
import pandas as pd
import numpy as np
from scipy import stats

# Safe Windows console encoding
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def calculate_ks_drift(reference_series: pd.Series, current_series: pd.Series, alpha: float = 0.05):
    """
    Computes 2-sample Kolmogorov-Smirnov test for continuous/discrete distributions.
    Returns p-value and boolean indicating whether significant distribution drift occurred.
    """
    res = stats.ks_2samp(reference_series, current_series)
    is_drifted = bool(res.pvalue < alpha)
    return {
        "statistic": round(float(res.statistic), 4),
        "p_value": round(float(res.pvalue), 5),
        "drift_detected": is_drifted
    }

def run_drift_analysis(
    baseline_path="data/raw/cdc_diabetes_data.csv",
    output_report="models/drift_report.json"
):
    """
    Simulates new incoming clinical batches and tests for demographic / feature drift
    against the baseline CDC epidemiological training dataset.
    """
    print(f"Loading reference baseline: {baseline_path}...")
    baseline_df = pd.read_csv(baseline_path)
    
    # Simulate a new incoming hospital stream with slight population shifts (e.g. older demographic, higher BMI)
    np.random.seed(42)
    sample_current = baseline_df.sample(5000, replace=True).copy()
    
    # Introduce controlled synthetic drift to demonstrate monitoring
    sample_current["BMI"] = sample_current["BMI"] + np.random.normal(3.5, 1.0, size=len(sample_current))
    sample_current["HighBP"] = np.random.choice([0, 1], size=len(sample_current), p=[0.25, 0.75])
    
    features_to_monitor = ["BMI", "HighBP", "HighChol", "GenHlth", "PhysActivity", "Age"]
    drift_results = {}
    total_drifted = 0
    
    print("\nRunning Kolmogorov-Smirnov Drift Tests across key clinical indicators...")
    for feature in features_to_monitor:
        analysis = calculate_ks_drift(baseline_df[feature], sample_current[feature])
        drift_results[feature] = analysis
        if analysis["drift_detected"]:
            total_drifted += 1
            status = "🚨 DRIFT DETECTED"
        else:
            status = "✅ STABLE"
        print(f"  {feature:<15} | p-value: {analysis['p_value']:<8} | {status}")
        
    overall_status = "ALERT: Retraining Recommended" if total_drifted >= 2 else "HEALTHY: No Retraining Needed"
    
    report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "baseline_sample_size": len(baseline_df),
        "current_batch_size": len(sample_current),
        "features_monitored": len(features_to_monitor),
        "features_drifted": total_drifted,
        "system_status": overall_status,
        "feature_details": drift_results
    }
    
    os.makedirs(os.path.dirname(output_report), exist_ok=True)
    with open(output_report, "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"\nDrift evaluation report saved to: {output_report}")
    print(f"Overall System Health: {overall_status}")
    return report

if __name__ == "__main__":
    run_drift_analysis()
