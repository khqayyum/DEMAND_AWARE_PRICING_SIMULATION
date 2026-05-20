# Demand-Aware Pricing Simulation for NYC Yellow Taxi

A machine learning system that predicts NYC Yellow Taxi fares and simulates bounded, transparent pricing adjustments based on demand, traffic, weather, and airport conditions — deployed as an interactive Streamlit application with a natural-language pricing assistant.

---

## What This Project Does

NYC Yellow Taxis operate on a mostly static fare structure that does not respond to real-world conditions like rush-hour demand, rain, or congestion. This project builds a **data-driven pricing simulation framework** that:

- Predicts taxi fares using a trained XGBoost model
- Applies small, explainable adjustments for demand, traffic, weather, and airport trips
- Visualizes route previews on an interactive map
- Answers natural-language questions via a built-in pricing assistant (GPT-powered)

The goal is not aggressive surge pricing — it is **bounded, transparent, and regulator-friendly** fare adjustment.

---

## Live Demo Preview

| Fare Estimator | Route Preview | Pricing Assistant |
|---|---|---|
| Select zones, hour, weather → get base + adjusted fare with full breakdown | Interactive dark-map route visualization with distance + duration | Ask natural-language questions about fares, routes, and best travel times |

---

## Model Performance

Three models were trained and compared on ~200,000 NYC Yellow Taxi trips (January 2024):

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| Ridge Regression | 4.62 | 2.18 | 0.91 |
| Random Forest | 3.27 | 1.34 | 0.96 |
| **XGBoost** | **2.89** | **1.01** | **0.981** |

XGBoost was selected as the final model. It predicts fares within ~$2.89 on average — strong performance for a real-world transportation dataset with tolls, surcharges, and location variability.

---

## Pricing Simulation

The pricing engine applies bounded rule-based adjustments on top of the ML prediction:

| Condition | Adjustment |
|-----------|-----------|
| DemandIndex ≥ 1.4 (high zone demand) | +$1.00 |
| High traffic intensity (low trip speed) | +$1.25 |
| Bad-weather day (rain, snow, wind) | +$0.50 |
| Airport trip (JFK, LGA, EWR) | +$0.75 |
| Post-adjustment multiplier | ×1.03 |

**Revenue impact:** The simulation produces approximately **2.6% revenue lift** compared to static pricing — achieved without aggressive surge multipliers.

Every adjustment is shown directly in the interface, making it auditable and transparent.

---

## Feature Engineering

Raw trip records were enriched with:

- **Temporal features** — pickup hour (sine/cosine encoded), weekday, rush-hour flag, night flag
- **Demand Index** — zone-hour trip count relative to median, capturing localized demand pressure
- **Traffic Proxy** — speed-based congestion score (lower speed = higher traffic intensity)
- **Weather features** — bad-weather flag from precipitation, wind gust, snow, snow depth
- **Interaction features** — demand × traffic, demand × weather severity

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Model | XGBoost, Random Forest, Ridge Regression |
| Data Processing | Python, Pandas, NumPy, Scikit-learn |
| App Framework | Streamlit |
| Mapping | GeoPandas, Folium / route visualization |
| Pricing Assistant | OpenAI GPT API |
| Model Persistence | joblib (.pkl) |
| Data Source | NYC TLC Yellow Taxi Trip Records (Jan 2024) + NYC Weather Dataset |

---

## Project Structure

```
nyc-taxi-pricing-simulation/
│
├── final_app.py              # Streamlit application (fare estimator + pricing assistant)
├── fare_model.pkl            # Trained XGBoost model (joblib serialized)
├── taxi_zone_lookup.csv      # TLC zone ID → borough/zone name mapping
│
└── README.md
```

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/khqayyum/nyc-taxi-pricing-simulation.git
cd nyc-taxi-pricing-simulation
```

### 2. Install dependencies
```bash
pip install streamlit pandas numpy scikit-learn xgboost joblib geopandas openai python-dotenv
```

### 3. Set up your OpenAI API key
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key_here
```
> The pricing assistant requires an OpenAI API key. The fare estimator tab works without it.

### 4. Run the app
```bash
streamlit run final_app.py
```

The app will open at `http://localhost:8501`

---

## How to Use

### Tab 1 — Fare Estimator
1. Select a **pickup zone** and **dropoff zone** from the TLC zone dropdown
2. Choose **pickup hour**, **weekday**, and **passenger count**
3. Select **weather condition** (Clear / Rain / Snow)
4. Click estimate — the app shows:
   - Base model fare (XGBoost prediction)
   - Final adjusted fare (with pricing simulation applied)
   - Breakdown of demand, traffic, weather, and airport adjustments
   - Estimated trip duration
   - Route preview map

### Tab 2 — Pricing Assistant
Ask natural-language questions such as:
- *"Best time to travel from Manhattan to JFK?"*
- *"How much does a trip from Midtown to Brooklyn cost at 6 PM in rain?"*
- *"Is taxi service available to Newark Airport?"*
- *"When is it cheapest to go from Times Square to the airport?"*

The assistant uses GPT + the fare model to answer with real estimates and practical suggestions.

---

## Key Design Decisions

### Why XGBoost?
Taxi fare prediction involves nonlinear interactions between distance, duration, location, time, and demand. XGBoost captures these better than linear models and outperformed Random Forest on RMSE and MAE.

### Why Bounded Adjustments Instead of Surge Pricing?
Unrestricted surge pricing creates fairness concerns in regulated transportation. This framework uses small, fixed adjustments tied to explicit conditions — making it auditable by regulators and understandable to riders.

### Why a Demand Index Instead of a Time-Series Model?
Demand was modeled at the zone-hour level (trips per zone per hour vs. median). This localizes demand pressure — high demand in Midtown at 5 PM does not mean high demand everywhere — without requiring a separate deployed forecasting model.

### Why a Natural-Language Assistant?
Users should not need to understand feature engineering to use a pricing tool. The GPT-powered assistant lets riders ask practical questions and receive plain-language explanations of fare behavior.

---

## Limitations

- Weather data is daily, not hourly — reduces storm-level precision
- Traffic estimated from trip speed, not live traffic APIs
- Route distances are zone-centroid estimates, not official TLC meter readings
- Dataset covers January 2024 only — may not fully generalize to other seasons
- Rider behavioral response to pricing changes is not modeled

---

## Future Work

- Integrate live traffic APIs (Google Maps, HERE) for real-time congestion data
- Add hourly weather data for finer weather-condition modeling
- Build a separate demand forecasting model using lagged zone-hour patterns
- Expand into a regulator-facing policy simulation dashboard
- Add fairness analysis across boroughs and pickup zones

---

## Data Sources

- [NYC TLC Yellow Taxi Trip Records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) — January 2024
- NYC Weather Dataset — daily weather observations for New York City

---

## Research Report

A full academic report covering the methodology, feature engineering, model evaluation, pricing simulation, and system-level impact analysis is included in the repository: `Capstone_Report.pdf`
