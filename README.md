# ⚡ Renewable Energy Load Balancing System

An end-to-end AI system that predicts electricity demand, forecasts solar and wind
generation, calculates the supply–demand gap, and recommends real-time grid
interventions — built on 5 years of real European power data.

---

## 🎯 Problem Statement

Germany's electricity grid must balance supply and demand every single hour.
Renewable energy (solar + wind) is intermittent — it fluctuates with weather.
Conventional power plants (gas, coal) must fill the gap.

**This system answers three questions every hour:**
1. How much electricity will people demand?
2. How much solar + wind will be available?
3. What intervention is needed to keep the grid balanced?

---

## 📊 Dataset

**Source:** [Open Power System Data (OPSD)](https://data.open-power-system-data.org/time_series/)

| Property | Value |
|---|---|
| Country | Germany |
| Period | January 2015 → September 2020 |
| Resolution | Hourly |
| Total rows | 50,400 hours (~5.75 years) |
| Key columns | Solar generation, wind generation, load demand, load forecast, day-ahead price |

---

## 🏗️ System Architecture

```
Raw OPSD Data (280+ columns)
        │
        ▼
┌─────────────────┐
│   Step 1: EDA   │  Exploratory analysis · missing value handling
│   eda.py        │  Feature engineering · ENTSOE benchmark analysis
└────────┬────────┘
         │
         ▼
┌──────────────────────────┐
│  Step 2: Demand Model    │  XGBoost + Random Forest
│  model_demand_forecast.py│  19 lag/rolling/cyclical features
│                          │  MAE: 490 MW  |  R²: 0.9956
│                          │  94.9% better than ENTSOE baseline
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Step 3: Supply Models   │  XGBoost (solar) + XGBoost (wind)
│  model_supply_gap.py     │  Solar MAE: 255 MW  |  R²: 0.9955
│                          │  Wind  MAE: 913 MW  |  R²: 0.9836
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Step 4: Gap Analysis    │  Gap = Demand − (Solar + Wind)
│  generate_full_gap.py    │  7-level intervention engine
│                          │  Quantile-based adaptive thresholds
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Step 5: Dashboard       │  Streamlit · 5 interactive tabs
│  dashboard.py            │  Live threshold sliders
│                          │  Year/month/all-time filtering
└──────────────────────────┘
```

---

## 🔑 Key Results

### Demand Prediction

| Model | MAE (MW) | R² | vs ENTSOE |
|---|---|---|---|
| ENTSOE Forecast (baseline) | 9,582 | ~0.30 | — |
| Random Forest | 518 | 0.9952 | **94.6% better** |
| **XGBoost** | **490** | **0.9956** | **94.9% better** |

### Supply Prediction

| Model | Target | MAE (MW) | R² |
|---|---|---|---|
| XGBoost | Solar generation | 255 | 0.9955 |
| XGBoost | Wind generation | 913 | 0.9836 |

### Grid Balance (2015–2019)

| Metric | Value |
|---|---|
| Average gap | +41,168 MW (deficit) |
| Renewable share of demand | ~26% |
| Hours in deficit | 100% (conventional backup always needed) |
| Best month (lowest deficit) | May–July (peak solar) |
| Worst month (highest deficit) | November–February (low solar, high demand) |

---

## 🚦 Intervention Engine

The system classifies every hour into one of 7 intervention levels using
**quantile-based thresholds** computed from your own data — no hardcoded values.

| Level | Threshold | Action |
|---|---|---|
| 🔴 CRITICAL | > p90 of deficits | Emergency import + backup generators |
| 🟠 HIGH | > p65 of deficits | Demand response + gas peakers |
| 🟡 MODERATE | > p25 of deficits | Minor demand response |
| 🟢 BALANCED | Within normal range | No action needed |
| 🔵 SURPLUS | < p75 of surpluses | Charge storage / export |
| 🟣 HIGH SURPLUS | < p35 of surpluses | Curtail wind + charge all storage |
| ⚫ CRIT. SURPLUS | < p10 of surpluses | Emergency curtailment |

---

## 📁 Project Structure

```
project/
│
├── data/
│   └── time_series_60min_singleindex.csv   ← OPSD raw data
│
├── outputs/
│   ├── germany_energy_clean.csv            ← cleaned dataset (Step 1)
│   ├── demand_predictions.csv              ← demand model output (Step 2)
│   ├── gap_interventions.csv               ← gap + labels (Step 3)
│   └── full_gap_2015_2020.csv              ← full 5-year analysis (Step 4)
│
├── models/
│   ├── xgb_demand_model.pkl                ← XGBoost demand model
│   ├── rf_demand_model.pkl                 ← Random Forest demand model
│   ├── solar_model.pkl                     ← XGBoost solar model
│   └── wind_model.pkl                      ← XGBoost wind model
│
├── eda.py                                  ← Step 1: EDA
├── model_demand_forecast.py                ← Step 2: Demand model
├── model_supply_gap.py                     ← Step 3: Supply + gap
├── generate_full_gap.py                    ← Step 4: Full range analysis
├── dashboard.py                            ← Step 5: Streamlit dashboard
│
└── README.md
```

---

## ⚙️ How to Run

### 1. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib streamlit
```

### 2. Download the dataset

Go to [OPSD Time Series](https://data.open-power-system-data.org/time_series/)
and download `time_series_60min_singleindex.csv`.

### 3. Run in order

```bash
# Step 1 — EDA and data cleaning
python eda.py

# Step 2 — Demand prediction models
python model_demand_forecast.py

# Step 3 — Supply prediction + gap analysis
python model_supply_gap.py

# Step 4 — Full 2015-2020 range analysis
python generate_full_gap.py

# Step 5 — Launch dashboard
streamlit run dashboard.py
```

---

## 📈 Feature Engineering

### Demand Model Features (19 total)

| Category | Features |
|---|---|
| Lag features | `lag_1h`, `lag_2h`, `lag_3h`, `lag_24h`, `lag_48h`, `lag_168h` |
| Rolling stats | `roll_6h`, `roll_24h`, `roll_168h`, `roll_std_24h`, `roll_std_168h` |
| Cyclical time | `hour_sin/cos`, `month_sin/cos`, `dow_sin/cos` |
| Calendar | `is_weekend`, `year` |

### Supply Model Features (13 solar / 14 wind)

| Category | Features |
|---|---|
| Lag features | `lag_1h`, `lag_24h`, `lag_168h` |
| Rolling mean | `roll_24h` (solar), `roll_24h` + `roll_72h` (wind) |
| Cyclical time | `hour_sin/cos`, `month_sin/cos`, `doy_sin/cos` |
| Capacity | `solar_cap` / `wind_cap` (installed MW — grows year-over-year) |
| Calendar | `is_weekend`, `year` |

---

## 💡 Key Technical Decisions

**Why time-based train/test split (not random)?**
Random splitting leaks future data into training. For time series, the model
must always be trained on past data and tested on future data.

**Why predict demand ourselves instead of using ENTSOE forecast?**
The ENTSOE forecast column was not in comparable MW units for this dataset.
Our XGBoost model achieves MAE 490 MW vs ENTSOE's 9,582 MW — a 94.9% improvement.

**Why quantile-based thresholds for interventions?**
Hardcoded values (e.g. 8,000 MW = "critical") are meaningless for different
grids or time periods. Quantile thresholds adapt automatically: "critical" always
means the top 10% worst hours in your actual data.

**Why cyclical encoding for hour/month?**
Without it, models treat hour 23 and hour 0 as far apart (23 units difference)
when they are actually adjacent. `sin/cos` encoding makes them neighbours.

---

## 🌍 Real-World Findings

1. **Germany's renewables covered only ~26% of demand** (2015–2019).
   Coal, gas, and nuclear filled the remaining ~74% gap.

2. **The gap is remarkably stable year-over-year** (~40,000 MW avg).
   Despite growing renewable capacity, demand also grew proportionally.

3. **Solar peaks in summer, but demand also dips** — making summer the
   *best* season for renewable coverage.

4. **January 2017 had the worst single-month deficit** (48,749 MW avg)
   due to a combination of low solar irradiance and high heating demand.

5. **Wind is the dominant renewable** — contributing ~3.7× more than solar
   on average (11,155 MW vs 3,128 MW).

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| Pandas / NumPy | Data processing |
| Scikit-learn | Random Forest, metrics |
| XGBoost | Gradient boosting models |
| Matplotlib / Seaborn | Visualisation |
| Streamlit | Interactive dashboard |
| Joblib | Model serialisation |

---

## 🚀 Future Improvements

- [ ] Add LSTM for sequential demand forecasting
- [ ] Incorporate weather data (temperature, cloud cover, wind speed)
- [ ] Extend to other European countries using OPSD multi-country data
- [ ] Add Prophet model for long-range seasonal forecasting
- [ ] Deploy dashboard to Streamlit Cloud
- [ ] Add causal inference (DoWhy) to identify root causes of deficit spikes

---

## 👤 Author

Built as a learning project in AI engineering.  
Dataset: © Open Power System Data (CC BY 4.0)
