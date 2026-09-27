import os
import sys
import pandas as pd
from ucimlrepo import fetch_ucirepo

# Ensure safe printing on Windows consoles
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def download_and_save_data(output_dir="data/raw"):
    """
    Downloads the official CDC BRFSS Diabetes Health Indicators dataset
    from UCI Machine Learning Repository (ID: 891) and saves it locally.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "cdc_diabetes_data.csv")
    
    if os.path.exists(csv_path):
        print(f"Dataset already exists at: {csv_path}")
        df = pd.read_csv(csv_path)
        return df

    print("Downloading CDC Diabetes dataset from UCI repository (ID: 891)...")
    dataset = fetch_ucirepo(id=891)
    
    X = dataset.data.features
    y = dataset.data.targets
    
    # Combine features and target into one clean DataFrame
    df = pd.concat([X, y], axis=1)
    df.to_csv(csv_path, index=False)
    
    print(f"Success! Saved {len(df):,} patient records to: {csv_path}")
    return df

if __name__ == "__main__":
    download_and_save_data()
