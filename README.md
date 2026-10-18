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
**Last Updated:** `2026-10-18 19:11:07 UTC`  
**Monitored Regions:** `47` global urban & endemic centers  
**Project Lead & Creator:** **Rohith Ashwa Vardhan**

| Outbreak Risk Tier | Region Count | Percentage |
| :--- | :---: | :---: |
| 🔴 **High Risk** | `30` | `63.8%` |
| 🟠 **Medium Risk** | `12` | `25.5%` |
| 🟢 **Low Risk** | `5` | `10.6%` |

#### 🚨 Current High-Risk Vector Transmission Zones

| Region | Country | Endemic Focus | 14-Day Temp | Humidity | Risk Score |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Thiruvananthapuram** | India | Dengue/Chikungunya | 26.63°C | 87.67% | `0.86` |
| **Panama City** | Panama | Dengue | 26.96°C | 87.1% | `0.85` |
| **Ho Chi Minh City** | Vietnam | Dengue | 27.24°C | 86.43% | `0.85` |
| **Colombo** | Sri Lanka | Dengue | 26.75°C | 86.38% | `0.85` |
| **Cartagena** | Colombia | Dengue | 27.75°C | 85.62% | `0.85` |
| **Santo Domingo** | Dominican Republic | Dengue | 26.15°C | 85.48% | `0.84` |
| **Lagos** | Nigeria | Malaria | 27.3°C | 84.81% | `0.84` |
| **Bangkok** | Thailand | Dengue | 27.95°C | 83.19% | `0.83` |
| **Guayaquil** | Ecuador | Dengue | 27.24°C | 82.33% | `0.83` |
| **Singapore** | Singapore | Dengue | 27.89°C | 82.24% | `0.83` |
| **Yangon** | Myanmar | Malaria/Dengue | 28.65°C | 81.76% | `0.82` |
| **San Juan** | Puerto Rico | Dengue | 27.7°C | 80.76% | `0.82` |
| **Veracruz** | Mexico | Dengue | 26.23°C | 80.67% | `0.81` |
| **Havana** | Cuba | Dengue | 26.82°C | 79.76% | `0.81` |
| **Kinshasa** | DR Congo | Malaria | 25.75°C | 79.9% | `0.80` |
| **Bhubaneswar** | India | Malaria/Dengue | 27.61°C | 78.62% | `0.80` |
| **Rio de Janeiro** | Brazil | Dengue | 23.43°C | 85.29% | `0.79` |
| **Accra** | Ghana | Malaria | 27.57°C | 77.48% | `0.79` |
| **Miami** | United States | Low Baseline | 26.87°C | 77.48% | `0.79` |
| **Guwahati** | India | Malaria/JE | 26.25°C | 76.38% | `0.78` |
| **Chennai** | India | Dengue/Chikungunya | 28.74°C | 76.29% | `0.77` |
| **Patna** | India | Dengue/Kala-azar | 26.34°C | 75.29% | `0.77` |
| **Tegucigalpa** | Honduras | Dengue | 22.18°C | 88.29% | `0.75` |
| **Dar es Salaam** | Tanzania | Malaria | 26.74°C | 73.19% | `0.75` |
| **Lucknow** | India | Dengue/JE | 26.12°C | 73.24% | `0.74` |
| **Bengaluru** | India | Dengue | 23.81°C | 75.95% | `0.72` |
| **Kampala** | Uganda | Malaria | 22.23°C | 83.81% | `0.71` |
| **Kolkata** | India | Dengue/Malaria | 28.42°C | 70.57% | `0.71` |
| **Manila** | Philippines | Dengue | 29.3°C | 70.62% | `0.69` |
| **New Delhi** | India | Dengue/Chikungunya | 26.28°C | 67.43% | `0.67` |

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
