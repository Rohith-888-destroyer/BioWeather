import json
import logging
import os
import time
from datetime import datetime, timezone, timedelta
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Representative global regions weighted toward Dengue / Malaria / Chikungunya endemic zones
REGIONS = [
    # India & South Asia (High-Density Vector Transmission Zones)
    {"id": "DEL", "name": "New Delhi", "country": "India", "lat": 28.6139, "lon": 77.2090, "endemic_focus": "Dengue/Chikungunya"},
    {"id": "BOM", "name": "Mumbai", "country": "India", "lat": 19.0760, "lon": 72.8777, "endemic_focus": "Dengue/Malaria"},
    {"id": "CCU", "name": "Kolkata", "country": "India", "lat": 22.5726, "lon": 88.3639, "endemic_focus": "Dengue/Malaria"},
    {"id": "MAA", "name": "Chennai", "country": "India", "lat": 13.0827, "lon": 80.2707, "endemic_focus": "Dengue/Chikungunya"},
    {"id": "BLR", "name": "Bengaluru", "country": "India", "lat": 12.9716, "lon": 77.5946, "endemic_focus": "Dengue"},
    {"id": "HYD", "name": "Hyderabad", "country": "India", "lat": 17.3850, "lon": 78.4867, "endemic_focus": "Dengue/Malaria"},
    {"id": "AMD", "name": "Ahmedabad", "country": "India", "lat": 23.0225, "lon": 72.5714, "endemic_focus": "Dengue/Malaria"},
    {"id": "PNQ", "name": "Pune", "country": "India", "lat": 18.5204, "lon": 73.8567, "endemic_focus": "Dengue/Zika"},
    {"id": "JAI", "name": "Jaipur", "country": "India", "lat": 26.9124, "lon": 75.7873, "endemic_focus": "Dengue/Malaria"},
    {"id": "LKO", "name": "Lucknow", "country": "India", "lat": 26.8467, "lon": 80.9462, "endemic_focus": "Dengue/JE"},
    {"id": "PAT", "name": "Patna", "country": "India", "lat": 25.5941, "lon": 85.1376, "endemic_focus": "Dengue/Kala-azar"},
    {"id": "GAU", "name": "Guwahati", "country": "India", "lat": 26.1445, "lon": 91.7362, "endemic_focus": "Malaria/JE"},
    {"id": "TRV", "name": "Thiruvananthapuram", "country": "India", "lat": 8.5241, "lon": 76.9366, "endemic_focus": "Dengue/Chikungunya"},
    {"id": "BBI", "name": "Bhubaneswar", "country": "India", "lat": 20.2961, "lon": 85.8245, "endemic_focus": "Malaria/Dengue"},

    # Southeast Asia
    {"id": "BKK", "name": "Bangkok", "country": "Thailand", "lat": 13.7563, "lon": 100.5018, "endemic_focus": "Dengue"},
    {"id": "JKT", "name": "Jakarta", "country": "Indonesia", "lat": -6.2088, "lon": 106.8456, "endemic_focus": "Dengue"},
    {"id": "MNL", "name": "Manila", "country": "Philippines", "lat": 14.5995, "lon": 120.9842, "endemic_focus": "Dengue"},
    {"id": "DAC", "name": "Dhaka", "country": "Bangladesh", "lat": 23.8103, "lon": 90.4125, "endemic_focus": "Dengue"},
    {"id": "CMB", "name": "Colombo", "country": "Sri Lanka", "lat": 6.9271, "lon": 79.8612, "endemic_focus": "Dengue"},
    {"id": "SGN", "name": "Ho Chi Minh City", "country": "Vietnam", "lat": 10.8231, "lon": 106.6297, "endemic_focus": "Dengue"},
    {"id": "SIN", "name": "Singapore", "country": "Singapore", "lat": 1.3521, "lon": 103.8198, "endemic_focus": "Dengue"},
    {"id": "RGN", "name": "Yangon", "country": "Myanmar", "lat": 16.8661, "lon": 96.1951, "endemic_focus": "Malaria/Dengue"},

    # Sub-Saharan Africa
    {"id": "LOS", "name": "Lagos", "country": "Nigeria", "lat": 6.5244, "lon": 3.3792, "endemic_focus": "Malaria"},
    {"id": "FIH", "name": "Kinshasa", "country": "DR Congo", "lat": -4.4419, "lon": 15.2663, "endemic_focus": "Malaria"},
    {"id": "NBO", "name": "Nairobi", "country": "Kenya", "lat": -1.2921, "lon": 36.8219, "endemic_focus": "Malaria"},
    {"id": "ACC", "name": "Accra", "country": "Ghana", "lat": 5.6037, "lon": -0.1870, "endemic_focus": "Malaria"},
    {"id": "EBB", "name": "Kampala", "country": "Uganda", "lat": 0.3476, "lon": 32.5825, "endemic_focus": "Malaria"},
    {"id": "DAR", "name": "Dar es Salaam", "country": "Tanzania", "lat": -6.7924, "lon": 39.2083, "endemic_focus": "Malaria"},
    {"id": "MPM", "name": "Maputo", "country": "Mozambique", "lat": -25.9692, "lon": 32.5732, "endemic_focus": "Malaria"},
    {"id": "ABJ", "name": "Abidjan", "country": "Cote d'Ivoire", "lat": 5.3600, "lon": -4.0083, "endemic_focus": "Malaria"},
    {"id": "DKR", "name": "Dakar", "country": "Senegal", "lat": 14.7167, "lon": -17.4677, "endemic_focus": "Malaria"},
    {"id": "ADD", "name": "Addis Ababa", "country": "Ethiopia", "lat": 9.0300, "lon": 38.7400, "endemic_focus": "Malaria"},

    # Latin America & Caribbean
    {"id": "MAO", "name": "Manaus", "country": "Brazil", "lat": -3.1190, "lon": -60.0217, "endemic_focus": "Dengue/Malaria"},
    {"id": "GIG", "name": "Rio de Janeiro", "country": "Brazil", "lat": -22.9068, "lon": -43.1729, "endemic_focus": "Dengue"},
    {"id": "GYE", "name": "Guayaquil", "country": "Ecuador", "lat": -2.1894, "lon": -79.8891, "endemic_focus": "Dengue"},
    {"id": "SJU", "name": "San Juan", "country": "Puerto Rico", "lat": 18.4655, "lon": -66.1057, "endemic_focus": "Dengue"},
    {"id": "TGU", "name": "Tegucigalpa", "country": "Honduras", "lat": 14.0723, "lon": -87.1921, "endemic_focus": "Dengue"},
    {"id": "CTG", "name": "Cartagena", "country": "Colombia", "lat": 10.3910, "lon": -75.4794, "endemic_focus": "Dengue"},
    {"id": "VER", "name": "Veracruz", "country": "Mexico", "lat": 19.1738, "lon": -96.1342, "endemic_focus": "Dengue"},
    {"id": "PTY", "name": "Panama City", "country": "Panama", "lat": 8.9824, "lon": -79.5199, "endemic_focus": "Dengue"},
    {"id": "SDQ", "name": "Santo Domingo", "country": "Dominican Republic", "lat": 18.4861, "lon": -69.9312, "endemic_focus": "Dengue"},
    {"id": "HAV", "name": "Havana", "country": "Cuba", "lat": 23.1136, "lon": -82.3666, "endemic_focus": "Dengue"},

    # Control Baselines
    {"id": "MIA", "name": "Miami", "country": "United States", "lat": 25.7617, "lon": -80.1918, "endemic_focus": "Low Baseline"},
    {"id": "ROM", "name": "Rome", "country": "Italy", "lat": 41.9028, "lon": 12.4964, "endemic_focus": "Low Baseline"},
    {"id": "TYO", "name": "Tokyo", "country": "Japan", "lat": 35.6762, "lon": 139.6503, "endemic_focus": "Low Baseline"},
    {"id": "SYD", "name": "Sydney", "country": "Australia", "lat": -33.8688, "lon": 151.2093, "endemic_focus": "Low Baseline"},
    {"id": "CAI", "name": "Cairo", "country": "Egypt", "lat": 30.0444, "lon": 31.2357, "endemic_focus": "Low Baseline"}
]

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

def fetch_open_meteo(lat, lon, max_retries=3):
    """Fetch live current telemetry + 14-day historical weather metrics from Open-Meteo."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,precipitation",
        "past_days": 14,
        "forecast_days": 7,
        "daily": "temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum",
        "timezone": "UTC"
    }
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(OPEN_METEO_URL, params=params, timeout=10)
            if resp.status_code == 200:
                payload = resp.json()
                daily = payload.get("daily", {})
                current = payload.get("current", {})
                return {
                    "current_temp": current.get("temperature_2m"),
                    "current_humidity": current.get("relative_humidity_2m"),
                    "current_precip": current.get("precipitation"),
                    "time": daily.get("time", []),
                    "temp_mean": daily.get("temperature_2m_mean", []),
                    "humidity_mean": daily.get("relative_humidity_2m_mean", []),
                    "precip_sum": daily.get("precipitation_sum", []),
                    "source": "Open-Meteo Real-Time"
                }
        except Exception as e:
            logging.warning(f"Open-Meteo attempt {attempt} failed for ({lat}, {lon}): {e}")
            time.sleep(2 ** attempt)
    return None

def fetch_nasa_power_fallback(lat, lon):
    """Fallback fetch from NASA POWER API for recent 14-day window."""
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=14)
    params = {
        "parameters": "T2M,RH2M,PRECTOTCORR",
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": start_dt.strftime("%Y%m%d"),
        "end": end_dt.strftime("%Y%m%d"),
        "format": "JSON"
    }
    try:
        resp = requests.get(NASA_POWER_URL, params=params, timeout=12)
        if resp.status_code == 200:
            properties = resp.json().get("properties", {}).get("parameter", {})
            times = sorted(list(properties.get("T2M", {}).keys()))
            temp_vals = [properties.get("T2M", {}).get(t) for t in times]
            hum_vals = [properties.get("RH2M", {}).get(t) for t in times]
            return {
                "current_temp": temp_vals[-1] if temp_vals else None,
                "current_humidity": hum_vals[-1] if hum_vals else None,
                "current_precip": 0.0,
                "time": [f"{t[:4]}-{t[4:6]}-{t[6:]}" for t in times],
                "temp_mean": temp_vals,
                "humidity_mean": hum_vals,
                "precip_sum": [properties.get("PRECTOTCORR", {}).get(t) for t in times],
                "source": "NASA POWER Fallback"
            }
    except Exception as e:
        logging.error(f"NASA POWER fallback failed for ({lat}, {lon}): {e}")
    return None

def run_ingestion(output_dir="data/raw"):
    """Run full real-time ingestion pipeline for all defined regions."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    timestamped_file = os.path.join(output_dir, f"climate_raw_{timestamp}.json")
    latest_file = os.path.join(output_dir, "latest.json")

    results = []
    logging.info(f"Starting real-time climate telemetry ingestion for {len(REGIONS)} global regions...")

    for r in REGIONS:
        logging.info(f"Ingesting real-time metrics: {r['name']}, {r['country']}...")
        climate_data = fetch_open_meteo(r["lat"], r["lon"])
        if not climate_data:
            logging.warning(f"Switching to NASA POWER fallback for {r['name']}...")
            climate_data = fetch_nasa_power_fallback(r["lat"], r["lon"])

        record = {**r, "climate": climate_data or {}}
        results.append(record)
        time.sleep(0.08)  # Rate limiting optimization

    payload = {"ingested_at": timestamp, "region_count": len(results), "data": results}

    with open(timestamped_file, "w") as f:
        json.dump(payload, f, indent=2)
    with open(latest_file, "w") as f:
        json.dump(payload, f, indent=2)

    logging.info(f"Successfully saved real-time ingested data to {timestamped_file} and {latest_file}.")
    return latest_file

if __name__ == "__main__":
    run_ingestion()
