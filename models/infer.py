import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Ground-Truth / Model Limitation Notice:
# This inference pipeline loads the trained PyTorch model to evaluate climate suitability for vector transmission.
# Output scores range 0.0 - 1.0 (Low, Medium, High risk tiers).
# This measures environmental suitability for outbreak risk, not clinical diagnostic counts.

class BioWeatherRiskModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32):
        super(BioWeatherRiskModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, 16)
        self.out = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return self.sigmoid(self.out(x))

def normalize_features(df):
    """Normalize input dataframe metrics to match training distribution."""
    temp = np.clip(df["mean_temp_14d"].values / 40.0, 0.0, 1.0)
    min_temp = np.clip(df["min_temp_14d"].values / 40.0, 0.0, 1.0)
    max_temp = np.clip(df["max_temp_14d"].values / 40.0, 0.0, 1.0)
    humidity = np.clip(df["mean_humidity_14d"].values / 100.0, 0.0, 1.0)
    vc_proxy = np.clip(df["vectorial_capacity_proxy"].values, 0.0, 1.0)
    return np.column_stack([temp, min_temp, max_temp, humidity, vc_proxy])

def assign_risk_tier(score):
    if score >= 0.65:
        return "High"
    elif score >= 0.35:
        return "Medium"
    else:
        return "Low"

def run_inference(feature_csv="data/processed/features_latest.csv", model_path="models/bioweather_model.pt", output_dir="data/processed"):
    """Load latest processed features, execute PyTorch inference, and save predictions."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("docs/data/processed", exist_ok=True)

    if not os.path.exists(feature_csv):
        raise FileNotFoundError(f"Feature CSV {feature_csv} not found. Run features.py first.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model artifact {model_path} not found. Run train.py first.")

    df = pd.read_csv(feature_csv)
    
    # Normalize features for neural network
    X_norm = normalize_features(df)
    X_tensor = torch.tensor(X_norm, dtype=torch.float32)

    # Load trained model
    checkpoint = torch.load(model_path)
    model = BioWeatherRiskModel(input_dim=5, hidden_dim=32)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        raw_scores = model(X_tensor).squeeze(1).numpy()

    df["risk_score"] = [round(float(s), 4) for s in raw_scores]
    df["risk_tier"] = df["risk_score"].apply(assign_risk_tier)

    # Save outputs
    output_csv = os.path.join(output_dir, "risk_predictions.csv")
    output_json = os.path.join(output_dir, "risk_predictions.json")
    docs_json = os.path.join("docs/data/processed", "risk_predictions.json")
    
    df.to_csv(output_csv, index=False)
    
    records = df.to_dict(orient="records")
    payload = {"total_regions": len(records), "predictions": records}
    
    with open(output_json, "w") as f:
        json.dump(payload, f, indent=2)
    with open(docs_json, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Successfully generated outbreak risk predictions for {len(df)} regions.")
    print(f"Saved: {output_csv}, {output_json}, & {docs_json}")
    return df

if __name__ == "__main__":
    run_inference()
