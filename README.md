# EV Adoption Forecasting — Germany (2025–2029)

> **Portfolio Project** | Nisrina Afnan Walyadin
> Skills demonstrated: Python · Time-Series Forecasting · ARIMA · Prophet · Polynomial Regression · Power BI · Data Pipeline Design

---

## Project Overview

This project forecasts the future demand for electric vehicles (EVs) in Germany, answering the question:

> **"How many people will buy an EV in the coming years — and what does that mean for charging infrastructure demand?"**

Using 15 years of historical BEV registration data (2010–2024), three forecasting models are compared to produce demand projections through 2029. The analysis is directly relevant to EV infrastructure operators like IONITY, who need demand forecasts to plan network expansion.

---

## Key Findings

| Metric | Value |
|--------|-------|
| Historical data range | 2010–2024 (15 years) |
| Best-performing model | Prophet (Test MAE: 99,855 registrations) |
| Forecast horizon | 2025–2029 |
| Projected new BEV registrations (2025) | ~712,000 |
| Projected new BEV registrations (2029) | ~1,110,000 |
| CAGR new BEV registrations (2020–2024) | 18.3% |

---

## Project Structure

```
ev_adoption_forecast/
|
+-- utils/
|   +-- fetch_ev_registrations.py   # Builds historical EV datasets (KBA + EAFO + Statista)
|
+-- notebooks/
|   +-- ev_forecasting.py           # Main analysis: EDA, 3 forecast models, regional breakdown
|
+-- outputs/                        # Generated charts and Power BI-ready CSVs
|   +-- 01_ev_eda_overview.png      # Historical EDA (registrations, stock, YoY growth, EU benchmark)
|   +-- 02_forecast_comparison.png  # All 3 models vs actual
|   +-- 03_forecast_bar_comparison.png  # Model comparison by forecast year
|   +-- 04_regional_analysis.png    # BEV adoption by German Bundesland
|   +-- ev_forecast_powerbi.csv     # Main forecast table (load into Power BI)
|   +-- ev_regional_powerbi.csv     # Regional data for map visual
|   +-- ev_timeseries_powerbi.csv   # Full historical time series
|
+-- data/                           # Source datasets
|   +-- germany_ev_timeseries.csv   # New BEV registrations + stock 2010-2024
|   +-- germany_ev_regional.csv     # BEV stock per Bundesland (2024)
|   +-- eu_ev_comparison_2023.csv   # EU country benchmark (2023)
|
+-- requirements.txt
+-- README.md
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Build the datasets
```bash
python utils/fetch_ev_registrations.py
```
Saves three CSVs to `data/` based on public KBA, EAFO, and Statista data.

### 3. Run the forecasting analysis
```bash
python notebooks/ev_forecasting.py
```
Outputs charts and Power BI-ready CSVs to `outputs/`.

### 4. Power BI Dashboard
Load `outputs/ev_forecast_powerbi.csv` and build:
- Line chart: `year` (X) vs `value` (Y), legend = `model`
- Slicer: `model` (toggle between Linear Regression / ARIMA / Prophet)
- Load `ev_regional_powerbi.csv` for map visual using `lat` / `lon`
- KPI card: `bev_per_1000_pop` for adoption rate by state

---

## Methodology

### Data Sources

| Dataset | Source | Coverage |
|---------|--------|----------|
| New BEV registrations per year | KBA Jahresbilanz + EAFO | Germany, 2010–2024 |
| Cumulative BEV stock | KBA + Statista | Germany, 2010–2024 |
| Public charging points | BDEW / Statista | Germany, 2010–2024 |
| Regional BEV stock | KBA | 16 Bundeslaender, 2024 |
| EU country comparison | EAFO | 10 countries, 2023 |

### Forecasting Models

**Model 1 — Polynomial Linear Regression (degree 2)**
Fits a quadratic trend curve to historical data. Simple and interpretable. Best for capturing the overall growth trajectory.

**Model 2 — ARIMA (1,1,1)**
Classical time-series model. Accounts for autocorrelation in the series and handles the non-stationarity of EV growth data via differencing.

**Model 3 — Prophet**
Facebook's forecasting library, designed for time series with strong trends and potential structural breaks. Best performer on this dataset (lowest test MAE).

### Train / Test Split
- Training: 2010–2021
- Test (held-out): 2022–2024
- Forecast: 2025–2029

### Model Performance (Test Set, 2022–2024)

| Model | Test MAE |
|-------|----------|
| Linear Regression | 118,321 |
| ARIMA (1,1,1) | 106,331 |
| **Prophet** | **99,855** |

---

## Forecast Results (New BEV Registrations)

| Year | Linear Regression | ARIMA | Prophet |
|------|-------------------|-------|---------|
| 2025 | 699,347 | 714,831 | 712,076 |
| 2026 | 831,421 | 768,088 | 811,377 |
| 2027 | 974,513 | 811,744 | 910,677 |
| 2028 | 1,128,625 | 847,530 | 1,009,977 |
| 2029 | 1,293,755 | 876,864 | 1,109,549 |

---

## Relevance to EV Industry

This analysis demonstrates skills directly applicable to roles at EV infrastructure operators:

- **Demand forecasting** — projecting future EV volume to size network investments
- **Model comparison** — evaluating multiple time-series approaches with held-out test sets
- **Regional analysis** — identifying which Bundeslaender have highest adoption rates (Bayern, Hamburg, Baden-Wuerttemberg)
- **Infrastructure gap framing** — connecting demand growth to charging point requirements
- **BI dashboard readiness** — all outputs structured for direct Power BI consumption

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python (Pandas, NumPy) | Data wrangling |
| Scikit-learn | Polynomial regression |
| Statsmodels | ARIMA |
| Prophet | Bayesian structural time series |
| Matplotlib / Seaborn | Visualisation |
| Power BI | Business intelligence dashboard |

---

## Author

**Nisrina Afnan Walyadin**
MSc Mathematics, Technical University of Munich
[LinkedIn](https://linkedin.com/in/nisrina-walyadin-5b7345178) · nisrinawalyadin@gmail.com