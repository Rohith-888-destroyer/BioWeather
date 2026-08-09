import json
import os
import requests

def send_discord_alert(webhook_url, high_risk_regions):
    """Dispatch alert formatted for Discord webhook integration."""
    if not high_risk_regions:
        return

    embed_fields = []
    for r in high_risk_regions[:10]:  # Limit top 10 for clean layout
        embed_fields.append({
            "name": f"🚨 {r['name']}, {r['country']}",
            "value": f"**Risk Score:** `{r['risk_score']:.2f}`\n**Endemic Focus:** {r.get('endemic_focus', 'Vector Zone')}\n**Temp:** {r['mean_temp_14d']}°C | **Humidity:** {r['mean_humidity_14d']}%",
            "inline": True
        })

    payload = {
        "username": "BioWeather Alert System",
        "avatar_url": "https://raw.githubusercontent.com/your-username/BioWeather/main/docs/latest_map.png",
        "embeds": [{
            "title": "🦟 High Vector Transmission Risk Warning",
            "description": f"BioWeather PyTorch model identified **{len(high_risk_regions)} regions** currently operating at **High Transmission Risk Tier (🔴 Score ≥ 0.65)**.",
            "color": 16730200,  # Red color integer
            "fields": embed_fields,
            "footer": {"text": "BioWeather Real-Time Autonomous Outbreak Forecaster"}
        }]
    }

    try:
        resp = requests.post(webhook_url, json=payload, timeout=8)
        print(f"Discord alert webhook status: {resp.status_code}")
    except Exception as e:
        print(f"Error sending Discord webhook alert: {e}")

def run_alert_notifier(predictions_csv="data/processed/risk_predictions.csv"):
    """Check risk predictions and trigger alert webhooks if configured."""
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL") or os.environ.get("ALERT_WEBHOOK_URL")
    if not os.path.exists(predictions_csv):
        print(f"Predictions CSV {predictions_csv} not found.")
        return

    import pandas as pd
    df = pd.read_csv(predictions_csv)
    high_risk = df[df["risk_tier"] == "High"].sort_values(by="risk_score", ascending=False).to_dict(orient="records")

    print(f"BioWeather Notifier: Detected {len(high_risk)} regions at High Outbreak Risk Tier.")
    if webhook_url:
        print(f"Sending real-time webhook notification to configured endpoint...")
        send_discord_alert(webhook_url, high_risk)
    else:
        print("No WEBHOOK_URL environment variable configured (Skipping live HTTP push notification).")

if __name__ == "__main__":
    run_alert_notifier()
