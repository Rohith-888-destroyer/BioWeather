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
**Last Updated:** `2026-11-18 01:34:45 UTC`  
**Monitored Regions:** `47` global urban & endemic centers  
**Project Lead & Creator:** **Rohith Ashwa Vardhan**

| Outbreak Risk Tier | Region Count | Percentage |
| :--- | :---: | :---: |
| 🔴 **High Risk** | `28` | `59.6%` |
| 🟠 **Medium Risk** | `8` | `17.0%` |
| 🟢 **Low Risk** | `11` | `23.4%` |

#### 🚨 Current High-Risk Vector Transmission Zones

| Region | Country | Endemic Focus | 14-Day Temp | Humidity | Risk Score |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Singapore** | Singapore | Dengue | 26.13°C | 90.95% | `0.87` |
| **Santo Domingo** | Dominican Republic | Dengue | 26.05°C | 86.19% | `0.85` |
| **Guayaquil** | Ecuador | Dengue | 27.05°C | 85.67% | `0.85` |
| **Colombo** | Sri Lanka | Dengue | 26.67°C | 85.24% | `0.84` |
| **Thiruvananthapuram** | India | Dengue/Chikungunya | 26.77°C | 84.29% | `0.84` |
| **San Juan** | Puerto Rico | Dengue | 27.25°C | 83.43% | `0.83` |
| **Panama City** | Panama | Dengue | 27.79°C | 82.38% | `0.83` |
| **Kinshasa** | DR Congo | Malaria | 25.58°C | 83.43% | `0.82` |
| **Lagos** | Nigeria | Malaria | 28.34°C | 82.52% | `0.82` |
| **Cartagena** | Colombia | Dengue | 28.59°C | 81.95% | `0.82` |
| **Veracruz** | Mexico | Dengue | 25.25°C | 82.19% | `0.81` |
| **Rio de Janeiro** | Brazil | Dengue | 24.9°C | 82.24% | `0.81` |
| **Abidjan** | Cote d'Ivoire | Malaria | 27.52°C | 79.43% | `0.81` |
| **Yangon** | Myanmar | Malaria/Dengue | 27.63°C | 79.14% | `0.80` |
| **Dar es Salaam** | Tanzania | Malaria | 27.61°C | 77.9% | `0.79` |
| **Havana** | Cuba | Dengue | 24.86°C | 80.29% | `0.79` |
| **Bhubaneswar** | India | Malaria/Dengue | 25.29°C | 78.52% | `0.78` |
| **Accra** | Ghana | Malaria | 28.39°C | 77.05% | `0.78` |
| **Ho Chi Minh City** | Vietnam | Dengue | 28.48°C | 76.95% | `0.78` |
| **Jakarta** | Indonesia | Dengue | 28.73°C | 76.62% | `0.77` |
| **Manaus** | Brazil | Dengue/Malaria | 29.34°C | 77.05% | `0.77` |
| **Dhaka** | Bangladesh | Dengue | 25.4°C | 74.38% | `0.75` |
| **Maputo** | Mozambique | Malaria | 25.47°C | 73.62% | `0.74` |
| **Tegucigalpa** | Honduras | Dengue | 22.13°C | 85.9% | `0.73` |
| **Miami** | United States | Low Baseline | 23.92°C | 75.67% | `0.71` |
| **Guwahati** | India | Malaria/JE | 23.04°C | 78.86% | `0.71` |
| **Kampala** | Uganda | Malaria | 21.76°C | 85.43% | `0.69` |
| **Bangkok** | Thailand | Dengue | 29.2°C | 67.19% | `0.65` |

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
