import json
import os
import re
from datetime import datetime, timezone
import pandas as pd

def format_risk_summary(predictions_csv="data/processed/risk_predictions.csv"):
    """Format summary statistics and high risk regions for README insertion."""
    if not os.path.exists(predictions_csv):
        return "<p>No forecast data available.</p>"

    df = pd.read_csv(predictions_csv)
    total = len(df)
    high_risk = df[df["risk_tier"] == "High"]
    med_risk = df[df["risk_tier"] == "Medium"]
    low_risk = df[df["risk_tier"] == "Low"]

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    summary_html = f"""
### 🌍 Weekly Vector Transmission Risk Summary
**Last Updated:** `{now_str}`  
**Monitored Regions:** `{total}` global urban & endemic centers

| Outbreak Risk Tier | Region Count | Percentage |
| :--- | :---: | :---: |
| 🔴 **High Risk** | `{len(high_risk)}` | `{round(len(high_risk)/total*100, 1)}%` |
| 🟠 **Medium Risk** | `{len(med_risk)}` | `{round(len(med_risk)/total*100, 1)}%` |
| 🟢 **Low Risk** | `{len(low_risk)}` | `{round(len(low_risk)/total*100, 1)}%` |

#### 🚨 Current High-Risk Vector Transmission Zones
"""

    if len(high_risk) == 0:
        summary_html += "\n*No regions currently meet the high-risk climate threshold.*\n"
    else:
        summary_html += "\n| Region | Country | Endemic Focus | 14-Day Temp | Humidity | Risk Score |\n"
        summary_html += "| :--- | :--- | :--- | :---: | :---: | :---: |\n"
        for _, row in high_risk.sort_values(by="risk_score", ascending=False).iterrows():
            summary_html += f"| **{row['name']}** | {row['country']} | {row['endemic_focus']} | {row['mean_temp_14d']}°C | {row['mean_humidity_14d']}% | `{row['risk_score']:.2f}` |\n"

    summary_html += f"""
![BioWeather Global Outbreak Risk Map](docs/latest_map.png)

*Interactive map view available at [BioWeather GitHub Pages / latest_map.html](docs/latest_map.html).*
"""
    return summary_html

def update_readme(readme_path="README.md", predictions_csv="data/processed/risk_predictions.csv"):
    """Update README.md content between RISK_MAP_START and RISK_MAP_END tags."""
    if not os.path.exists(readme_path):
        print(f"README file {readme_path} does not exist.")
        return

    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    summary_section = format_risk_summary(predictions_csv)
    pattern = r"(<!-- RISK_MAP_START -->)(.*?)(<!-- RISK_MAP_END -->)"
    replacement = rf"\1\n{summary_section}\n\3"

    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    print("README.md updated successfully with latest forecast metrics.")

if __name__ == "__main__":
    update_readme()
