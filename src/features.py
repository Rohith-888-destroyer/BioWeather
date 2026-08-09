import json
import os
import numpy as np
import pandas as pd

def calculate_vectorial_capacity_proxy(temp_mean, humidity_mean):
    """
    Simplified Mordecai et al. (2016, 2019) thermal response curve for vector-borne transmission.
    Optimal transmission range for Aedes aegypti / Anopheles is ~25°C - 29°C.
    Transmission suitability drops off sharply below 16°C and above 38°C.
    Humidity > 60% enhances vector longevity.
    """
    if np.isnan(temp_mean) or temp_mean < 15.0 or temp_mean > 40.0:
        thermal_factor = 0.0
    else:
        # Biquadratic thermal response curve centered at 27.5°C
        thermal_factor = max(0.0, 1.0 - ((temp_mean - 27.5) / 12.5) ** 2)

    # Relative humidity multiplier (vector survival increases above 60%)
    if np.isnan(humidity_mean):
        humidity_factor = 0.5
    else:
        humidity_factor = min(1.0, max(0.1, humidity_mean / 100.0))

    return thermal_factor * humidity_factor

def extract_features_from_region(region_entry):
    """Extract rolling epidemiological climate features for a single region."""
    region_id = region_entry.get("id")
    name = region_entry.get("name")
    country = region_entry.get("country")
    lat = region_entry.get("lat")
    lon = region_entry.get("lon")
    focus = region_entry.get("endemic_focus", "Unknown")

    climate = region_entry.get("climate", {})
    temp_list = climate.get("temp_mean", [])
    humidity_list = climate.get("humidity_mean", [])
    precip_list = climate.get("precip_sum", [])

    if not temp_list:
        return None

    # Filter non-None values
    valid_temps = [t for t in temp_list if t is not None]
    valid_humidity = [h for h in humidity_list if h is not None]
    valid_precip = [p for p in precip_list if p is not None]

    mean_temp = float(np.mean(valid_temps)) if valid_temps else 25.0
    mean_humidity = float(np.mean(valid_humidity)) if valid_humidity else 70.0
    total_precip = float(np.sum(valid_precip)) if valid_precip else 0.0
    max_temp = float(np.max(valid_temps)) if valid_temps else mean_temp
    min_temp = float(np.min(valid_temps)) if valid_temps else mean_temp

    # Compute epidemiological proxies
    vc_proxy = calculate_vectorial_capacity_proxy(mean_temp, mean_humidity)
    precip_anomaly = max(0.0, total_precip - 30.0) / 100.0  # Excessive rainfall breeding site availability

    return {
        "region_id": region_id,
        "name": name,
        "country": country,
        "lat": lat,
        "lon": lon,
        "endemic_focus": focus,
        "mean_temp_14d": round(mean_temp, 2),
        "min_temp_14d": round(min_temp, 2),
        "max_temp_14d": round(max_temp, 2),
        "mean_humidity_14d": round(mean_humidity, 2),
        "total_precip_14d": round(total_precip, 2),
        "vectorial_capacity_proxy": round(vc_proxy, 4),
        "breeding_site_index": round(precip_anomaly, 4)
    }

def run_feature_engineering(raw_file="data/raw/latest.json", output_dir="data/processed"):
    """Load latest raw climate json, produce engineered feature dataset, save CSV."""
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(raw_file):
        raise FileNotFoundError(f"Raw data file {raw_file} not found. Run ingest.py first.")

    with open(raw_file, "r") as f:
        raw_payload = json.load(f)

    regions_data = raw_payload.get("data", [])
    feature_records = []

    for reg in regions_data:
        feat = extract_features_from_region(reg)
        if feat:
            feature_records.append(feat)

    df = pd.DataFrame(feature_records)
    output_path = os.path.join(output_dir, "features_latest.csv")
    df.to_csv(output_path, index=False)
    print(f"Successfully processed {len(df)} region features into {output_path}")
    return output_path

if __name__ == "__main__":
    run_feature_engineering()
