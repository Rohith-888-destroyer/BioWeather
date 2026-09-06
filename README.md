# 🦟 BioWeather: Global Disease Outbreak Risk Forecaster

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Automation](https://img.shields.io/badge/Automation-GitHub_Actions-2088FF.svg)

**BioWeather** is a fully autonomous, self-updating global disease-outbreak-risk forecaster created by **Rohith Ashwa Vardhan**. It monitors climate-driven vector-borne diseases (such as Dengue, Malaria, Chikungunya, and Zika). 

Every hour, a scheduled **GitHub Action** automatically pulls fresh global climate telemetry from open APIs, engineers epidemiological suitability features, runs a trained **PyTorch deep learning model**, generates interactive & static risk maps, and commits the updated intelligence directly back to this repository—**requiring zero manual intervention**.

---

## 🛰️ Real-Time Outbreak Risk Forecast

<!-- RISK_MAP_START -->

### 🌍 Real-Time Vector Transmission Risk Summary
**Last Updated:** `2026-09-06 13:32:33 UTC`  
**Monitored Regions:** `47` global urban & endemic centers  
**Project Lead & Creator:** **Rohith Ashwa Vardhan**

| Outbreak Risk Tier | Region Count | Percentage |
| :--- | :---: | :---: |
| 🔴 **High Risk** | `34` | `72.3%` |
| 🟠 **Medium Risk** | `7` | `14.9%` |
| 🟢 **Low Risk** | `6` | `12.8%` |

#### 🚨 Current High-Risk Vector Transmission Zones

| Region | Country | Endemic Focus | 14-Day Temp | Humidity | Risk Score |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Yangon** | Myanmar | Malaria/Dengue | 26.39°C | 90.9% | `0.87` |
| **Lucknow** | India | Dengue/JE | 27.8°C | 89.14% | `0.87` |
| **Guwahati** | India | Malaria/JE | 27.91°C | 88.33% | `0.86` |
| **Bhubaneswar** | India | Malaria/Dengue | 27.9°C | 87.14% | `0.85` |
| **Kolkata** | India | Dengue/Malaria | 28.44°C | 87.52% | `0.85` |
| **Patna** | India | Dengue/Kala-azar | 28.65°C | 86.62% | `0.85` |
| **Lagos** | Nigeria | Malaria | 26.33°C | 85.86% | `0.84` |
| **Thiruvananthapuram** | India | Dengue/Chikungunya | 26.64°C | 84.81% | `0.84` |
| **Dhaka** | Bangladesh | Dengue | 28.56°C | 84.24% | `0.84` |
| **Ho Chi Minh City** | Vietnam | Dengue | 27.49°C | 83.52% | `0.84` |
| **Manila** | Philippines | Dengue | 27.19°C | 83.19% | `0.83` |
| **Mumbai** | India | Dengue/Malaria | 27.14°C | 83.0% | `0.83` |
| **Abidjan** | Cote d'Ivoire | Malaria | 25.29°C | 84.52% | `0.82` |
| **Bangkok** | Thailand | Dengue | 27.84°C | 81.57% | `0.82` |
| **Tokyo** | Japan | Low Baseline | 24.36°C | 85.57% | `0.82` |
| **Accra** | Ghana | Malaria | 26.1°C | 81.48% | `0.81` |
| **Colombo** | Sri Lanka | Dengue | 27.58°C | 80.19% | `0.81` |
| **Veracruz** | Mexico | Dengue | 28.26°C | 80.24% | `0.81` |
| **San Juan** | Puerto Rico | Dengue | 28.15°C | 80.1% | `0.81` |
| **Cartagena** | Colombia | Dengue | 29.2°C | 80.14% | `0.80` |
| **New Delhi** | India | Dengue/Chikungunya | 28.76°C | 79.52% | `0.80` |
| **Miami** | United States | Low Baseline | 28.49°C | 78.95% | `0.80` |
| **Singapore** | Singapore | Dengue | 27.93°C | 77.52% | `0.79` |
| **Dakar** | Senegal | Malaria | 28.83°C | 78.19% | `0.79` |
| **Pune** | India | Dengue/Zika | 24.49°C | 80.57% | `0.79` |
| **Havana** | Cuba | Dengue | 28.44°C | 76.86% | `0.78` |
| **Jaipur** | India | Dengue/Malaria | 28.02°C | 73.48% | `0.75` |
| **Guayaquil** | Ecuador | Dengue | 27.33°C | 72.43% | `0.74` |
| **Hyderabad** | India | Dengue/Malaria | 26.56°C | 71.19% | `0.72` |
| **Rio de Janeiro** | Brazil | Dengue | 22.17°C | 83.86% | `0.71` |
| **Tegucigalpa** | Honduras | Dengue | 24.37°C | 73.19% | `0.71` |
| **Bengaluru** | India | Dengue | 24.33°C | 71.81% | `0.69` |
| **Kinshasa** | DR Congo | Malaria | 25.8°C | 68.57% | `0.69` |
| **Chennai** | India | Dengue/Chikungunya | 30.14°C | 71.43% | `0.68` |

![BioWeather Global Outbreak Risk Map](docs/latest_map.png)

*Interactive map view available at [BioWeather GitHub Pages / latest_map.html](docs/latest_map.html).*

<!-- RISK_MAP_END -->

---

## 🧬 Methodology & Architecture

The BioWeather forecast pipeline follows a 5-tier architecture:

```
[Open-Meteo / NASA POWER APIs] 
            │
            ▼
[Data Ingestion (src/ingest.py)] ──> Raw JSON Cache (data/raw/)
            │
            ▼
[Feature Engineering (src/features.py)] ──> Rolling Epidemiological Metrics & R0 Proxies (data/processed/)
            │
            ▼
[PyTorch Inference (models/infer.py)] ──> Outbreak Risk Index (0.0 - 1.0) & Risk Tiers
            │
            ▼
[Visualization & Automation (src/map_generator.py + GitHub Actions)] ──> Interactive HTML & Static README Maps
```

### 1. Data Ingestion (`src/ingest.py`)
- Ingests 14-day historical and 7-day forecast climate series across **47 representative global urban centers** (including 14 major Indian cities), with heavy weighting toward Dengue/Malaria endemic regions (South & SE Asia, Sub-Saharan Africa, Latin America & Caribbean).
- Primary Source: **Open-Meteo API** (Keyless, high-resolution global telemetry).
- Fallback Source: **NASA POWER API** (Keyless solar & meteorological archive).
- Features automatic retries with exponential backoff and persistent raw payload logging in `data/raw/`.

### 2. Epidemiological Feature Engineering (`src/features.py`)
- Calculates 14-day rolling mean temperature (°C), relative humidity (%), and cumulative precipitation (mm).
- Computes **Vectorial Capacity & $R_0$ Transmission Suitability Proxies** derived from thermal response literature (*Mordecai et al. 2016, 2019*).
- Mosquito vector reproduction (*Aedes aegypti*, *Anopheles*) peaks in temperature windows between 25°C – 29°C with relative humidity > 60%, dropping sharply below 15°C and above 38°C.

### 3. Deep Learning Risk Model (`models/`)
- Implemented in **PyTorch** (`BioWeatherRiskModel`).
- Evaluates multi-dimensional climate feature vectors to output a normalized **Outbreak Transmission Risk Index (0.0 to 1.0)**:
  - 🔴 **High Risk ($\ge 0.65$)**: Optimal climate suitability for rapid vector proliferation & viral replication.
  - 🟠 **Medium Risk ($0.35 - 0.64$)**: Moderate transmission suitability; potential seasonal surge.
  - 🟢 **Low Risk ($< 0.35$)**: Sub-optimal thermal or moisture conditions for vector transmission.

---

## ⚠️ Important Scientific & Regulatory Disclaimer

> [!IMPORTANT]
> **Proxy Target & Model Limitations:**  
> Live open-access APIs for real-time epidemiological case counts (e.g. daily clinical hospital admissions) do not exist globally. Therefore, BioWeather is trained on **environmental and thermal vector suitability proxies** derived from peer-reviewed entomological research (*Mordecai et al.*), rather than clinical diagnostic records.  
>  
> **This tool is for research, educational, and environmental monitoring purposes only.** It does NOT constitute clinical advice, medical diagnosis, or official public health guidance.

---

## 💻 Local Running & Development

### Prerequisites
- Python 3.11+
- `pip`

### Setup & Execution
```bash
# Clone the repository
git clone https://github.com/your-username/BioWeather.git
cd BioWeather

# Install dependencies
pip install -r requirements.txt

# Step 1: Run climate ingestion
python src/ingest.py

# Step 2: Run feature engineering
python src/features.py

# Step 3: Train model (optional - pre-trained weights included)
python models/train.py

# Step 4: Run inference
python models/infer.py

# Step 5: Generate interactive map & README section
python src/map_generator.py
python src/update_readme.py
```

---

## 👨‍💻 Project Lead & Attribution

**Created & Developed by:** **Rohith Ashwa Vardhan**

---

## 📡 Data Sources & Attribution

- **Open-Meteo**: Weather forecast API provided under CC BY 4.0. [open-meteo.com](https://open-meteo.com/)
- **NASA POWER**: Prediction Of Worldwide Energy Resources project. [power.larc.nasa.gov](https://power.larc.nasa.gov/)

---

## 📜 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.
