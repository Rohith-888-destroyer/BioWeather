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
**Last Updated:** `2026-10-09 20:00:07 UTC`  
**Monitored Regions:** `47` global urban & endemic centers  
**Project Lead & Creator:** **Rohith Ashwa Vardhan**

| Outbreak Risk Tier | Region Count | Percentage |
| :--- | :---: | :---: |
| 🔴 **High Risk** | `31` | `66.0%` |
| 🟠 **Medium Risk** | `11` | `23.4%` |
| 🟢 **Low Risk** | `5` | `10.6%` |

#### 🚨 Current High-Risk Vector Transmission Zones

| Region | Country | Endemic Focus | 14-Day Temp | Humidity | Risk Score |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Ho Chi Minh City** | Vietnam | Dengue | 27.15°C | 87.1% | `0.85` |
| **Yangon** | Myanmar | Malaria/Dengue | 27.99°C | 85.14% | `0.84` |
| **Cartagena** | Colombia | Dengue | 27.6°C | 85.1% | `0.84` |
| **Thiruvananthapuram** | India | Dengue/Chikungunya | 26.98°C | 84.9% | `0.84` |
| **Lagos** | Nigeria | Malaria | 26.53°C | 85.29% | `0.84` |
| **Singapore** | Singapore | Dengue | 27.31°C | 84.19% | `0.84` |
| **Colombo** | Sri Lanka | Dengue | 27.18°C | 83.86% | `0.84` |
| **Bangkok** | Thailand | Dengue | 27.72°C | 82.95% | `0.83` |
| **Guayaquil** | Ecuador | Dengue | 26.93°C | 82.95% | `0.83` |
| **Rio de Janeiro** | Brazil | Dengue | 23.67°C | 89.05% | `0.83` |
| **Dakar** | Senegal | Malaria | 27.64°C | 81.67% | `0.82` |
| **Havana** | Cuba | Dengue | 27.25°C | 80.43% | `0.81` |
| **Guwahati** | India | Malaria/JE | 27.05°C | 80.1% | `0.81` |
| **Abidjan** | Cote d'Ivoire | Malaria | 25.73°C | 81.14% | `0.81` |
| **Patna** | India | Dengue/Kala-azar | 26.96°C | 79.76% | `0.81` |
| **Lucknow** | India | Dengue/JE | 26.14°C | 80.29% | `0.81` |
| **San Juan** | Puerto Rico | Dengue | 28.0°C | 79.19% | `0.80` |
| **Miami** | United States | Low Baseline | 27.11°C | 77.71% | `0.79` |
| **Accra** | Ghana | Malaria | 26.7°C | 77.24% | `0.79` |
| **Bhubaneswar** | India | Malaria/Dengue | 28.31°C | 77.24% | `0.78` |
| **Kolkata** | India | Dengue/Malaria | 28.67°C | 76.29% | `0.77` |
| **Kinshasa** | DR Congo | Malaria | 25.75°C | 75.81% | `0.77` |
| **Tegucigalpa** | Honduras | Dengue | 22.5°C | 87.48% | `0.76` |
| **Manila** | Philippines | Dengue | 28.68°C | 75.86% | `0.76` |
| **Dar es Salaam** | Tanzania | Malaria | 26.42°C | 73.52% | `0.75` |
| **Jakarta** | Indonesia | Dengue | 28.98°C | 70.57% | `0.70` |
| **Kampala** | Uganda | Malaria | 22.32°C | 81.19% | `0.69` |
| **Maputo** | Mozambique | Malaria | 23.09°C | 75.86% | `0.68` |
| **New Delhi** | India | Dengue/Chikungunya | 27.82°C | 66.43% | `0.66` |
| **Bengaluru** | India | Dengue | 24.53°C | 69.24% | `0.66` |
| **Dhaka** | Bangladesh | Dengue | 29.67°C | 68.1% | `0.65` |

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
