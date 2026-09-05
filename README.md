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
**Last Updated:** `2026-09-05 05:10:50 UTC`  
**Monitored Regions:** `47` global urban & endemic centers  
**Project Lead & Creator:** **Rohith Ashwa Vardhan**

| Outbreak Risk Tier | Region Count | Percentage |
| :--- | :---: | :---: |
| 🔴 **High Risk** | `34` | `72.3%` |
| 🟠 **Medium Risk** | `8` | `17.0%` |
| 🟢 **Low Risk** | `5` | `10.6%` |

#### 🚨 Current High-Risk Vector Transmission Zones

| Region | Country | Endemic Focus | 14-Day Temp | Humidity | Risk Score |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Yangon** | Myanmar | Malaria/Dengue | 26.13°C | 92.1% | `0.88` |
| **Lucknow** | India | Dengue/JE | 27.78°C | 89.57% | `0.87` |
| **Panama City** | Panama | Dengue | 26.95°C | 89.38% | `0.87` |
| **Guwahati** | India | Malaria/JE | 27.95°C | 88.71% | `0.86` |
| **Bhubaneswar** | India | Malaria/Dengue | 27.83°C | 87.95% | `0.86` |
| **Kolkata** | India | Dengue/Malaria | 28.52°C | 87.43% | `0.85` |
| **Patna** | India | Dengue/Kala-azar | 28.64°C | 87.1% | `0.85` |
| **Thiruvananthapuram** | India | Dengue/Chikungunya | 26.57°C | 85.05% | `0.84` |
| **Dhaka** | Bangladesh | Dengue | 28.62°C | 84.29% | `0.84` |
| **Ho Chi Minh City** | Vietnam | Dengue | 27.65°C | 82.76% | `0.83` |
| **Manila** | Philippines | Dengue | 27.31°C | 82.57% | `0.83` |
| **Santo Domingo** | Dominican Republic | Dengue | 27.32°C | 82.43% | `0.83` |
| **Tokyo** | Japan | Low Baseline | 24.77°C | 85.1% | `0.83` |
| **Abidjan** | Cote d'Ivoire | Malaria | 25.31°C | 84.71% | `0.83` |
| **Mumbai** | India | Dengue/Malaria | 27.19°C | 82.38% | `0.83` |
| **Bangkok** | Thailand | Dengue | 27.83°C | 81.81% | `0.82` |
| **Accra** | Ghana | Malaria | 26.03°C | 81.9% | `0.82` |
| **Veracruz** | Mexico | Dengue | 28.1°C | 80.95% | `0.82` |
| **Colombo** | Sri Lanka | Dengue | 27.6°C | 80.19% | `0.81` |
| **San Juan** | Puerto Rico | Dengue | 28.2°C | 79.95% | `0.81` |
| **Cartagena** | Colombia | Dengue | 29.13°C | 80.52% | `0.80` |
| **Miami** | United States | Low Baseline | 28.55°C | 78.33% | `0.79` |
| **New Delhi** | India | Dengue/Chikungunya | 29.06°C | 79.0% | `0.79` |
| **Pune** | India | Dengue/Zika | 24.47°C | 80.81% | `0.79` |
| **Dakar** | Senegal | Malaria | 28.87°C | 78.05% | `0.78` |
| **Singapore** | Singapore | Dengue | 28.02°C | 76.71% | `0.78` |
| **Havana** | Cuba | Dengue | 28.59°C | 76.24% | `0.77` |
| **Dar es Salaam** | Tanzania | Malaria | 25.1°C | 75.95% | `0.76` |
| **Guayaquil** | Ecuador | Dengue | 27.32°C | 72.05% | `0.74` |
| **Jaipur** | India | Dengue/Malaria | 28.2°C | 72.57% | `0.73` |
| **Tegucigalpa** | Honduras | Dengue | 24.04°C | 76.05% | `0.73` |
| **Bengaluru** | India | Dengue | 24.04°C | 73.86% | `0.70` |
| **Hyderabad** | India | Dengue/Malaria | 26.76°C | 69.19% | `0.70` |
| **Rio de Janeiro** | Brazil | Dengue | 21.71°C | 85.38% | `0.69` |

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
