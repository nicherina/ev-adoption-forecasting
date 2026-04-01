"""
ev_forecasting.py
=================
Time-series forecasting of EV adoption in Germany.

Compares 3 forecasting methods:
    1. Linear Regression  — simple trend extrapolation
    2. ARIMA              — classical time series model
    3. Prophet            — Facebook's forecasting library (handles seasonality well)

Also includes:
    - EDA of historical registration data
    - Regional heatmap analysis
    - EU comparison benchmarking
    - Export for Power BI

Run after fetch_ev_registrations.py

Author: Nisrina Afwan Walyadin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import os

warnings.filterwarnings("ignore")

# Try importing optional libraries
try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_ARIMA = True
except ImportError:
    HAS_ARIMA = False
    print("⚠️  statsmodels not installed. ARIMA skipped.")
    print("   Install with: pip install statsmodels\n")

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False
    print("⚠️  Prophet not installed. Prophet model skipped.")
    print("   Install with: pip install prophet\n")

# -- Config --------------------------------------------------------------------
DATA_DIR   = "data"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FORECAST_YEARS = 5          # How many years ahead to forecast
TRAIN_CUTOFF   = 2021       # Use data up to this year for training
COLORS = {
    "actual":      "#2A9D8F",
    "linear":      "#E63946",
    "arima":       "#F4A261",
    "prophet":     "#457B9D",
    "confidence":  "#DDDDDD",
}

# -- 1. Load Data --------------------------------------------------------------
print("=" * 60)
print("EV ADOPTION FORECASTING — GERMANY")
print("=" * 60)

df = pd.read_csv(f"{DATA_DIR}/germany_ev_timeseries.csv")
df_regional = pd.read_csv(f"{DATA_DIR}/germany_ev_regional.csv")
df_eu = pd.read_csv(f"{DATA_DIR}/eu_ev_comparison_2023.csv")

print(f"\nLoaded {len(df)} years of data ({df['year'].min()}–{df['year'].max()})")
print(df[["year", "new_bev_registrations", "total_ev_stock", "yoy_growth_pct"]].to_string(index=False))


# -- 2. EDA --------------------------------------------------------------------
print("\n-- EDA ------------------------------------------------------")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("EV Adoption in Germany — Historical Analysis (2010–2024)",
             fontsize=16, fontweight="bold")

# Plot 1: New BEV registrations over time
axes[0, 0].fill_between(df["year"], df["new_bev_registrations"],
                         alpha=0.3, color=COLORS["actual"])
axes[0, 0].plot(df["year"], df["new_bev_registrations"],
                color=COLORS["actual"], linewidth=2.5, marker="o")
axes[0, 0].set_title("New BEV Registrations per Year")
axes[0, 0].set_ylabel("Number of New EVs")
axes[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

# Plot 2: Total EV stock
axes[0, 1].fill_between(df["year"], df["total_ev_stock"],
                         alpha=0.3, color="#457B9D")
axes[0, 1].plot(df["year"], df["total_ev_stock"],
                color="#457B9D", linewidth=2.5, marker="o")
axes[0, 1].set_title("Total BEV Stock on Road")
axes[0, 1].set_ylabel("Total EVs")
axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1000:.0f}k"))

# Plot 3: YoY growth
colors_bar = ["#E63946" if x < 0 else "#2A9D8F" for x in df["yoy_growth_pct"].fillna(0)]
axes[0, 2].bar(df["year"], df["yoy_growth_pct"].fillna(0), color=colors_bar, edgecolor="white")
axes[0, 2].axhline(0, color="black", linewidth=0.8)
axes[0, 2].set_title("Year-on-Year Growth Rate (%)")
axes[0, 2].set_ylabel("Growth (%)")

# Plot 4: BEV vs PHEV
axes[1, 0].stackplot(df["year"],
                      df["new_bev_registrations"],
                      df["new_phev_registrations"],
                      labels=["BEV (Battery)", "PHEV (Plug-in Hybrid)"],
                      colors=[COLORS["actual"], "#F4A261"], alpha=0.8)
axes[1, 0].set_title("BEV vs PHEV New Registrations")
axes[1, 0].set_ylabel("Registrations")
axes[1, 0].legend(loc="upper left", fontsize=9)
axes[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

# Plot 5: Charging infrastructure vs EV stock
ax2 = axes[1, 1].twinx()
axes[1, 1].plot(df["year"], df["total_ev_stock"], color=COLORS["actual"],
                linewidth=2, label="EV Stock")
ax2.plot(df["year"], df["charging_stations"], color="#E63946",
         linewidth=2, linestyle="--", label="Charging Points")
axes[1, 1].set_title("EV Stock vs Charging Infrastructure")
axes[1, 1].set_ylabel("Total EVs", color=COLORS["actual"])
ax2.set_ylabel("Charging Points", color="#E63946")
axes[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

# Plot 6: EU comparison
df_eu_sorted = df_eu.sort_values("new_bev_2023", ascending=True)
bars = axes[1, 2].barh(df_eu_sorted["country"], df_eu_sorted["new_bev_2023"],
                        color=["#E63946" if c == "Germany" else "#2A9D8F"
                               for c in df_eu_sorted["country"]])
axes[1, 2].set_title("New BEV Registrations 2023 — EU Comparison")
axes[1, 2].set_xlabel("New Registrations")
axes[1, 2].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_ev_eda_overview.png", dpi=150, bbox_inches="tight")
print(f"Saved -> {OUTPUT_DIR}/01_ev_eda_overview.png")
plt.close()


# -- 3. Prepare Forecast Data --------------------------------------------------
train = df[df["year"] <= TRAIN_CUTOFF].copy()
test  = df[df["year"] >  TRAIN_CUTOFF].copy()

future_years = list(range(df["year"].max() + 1,
                           df["year"].max() + FORECAST_YEARS + 1))
all_future   = pd.DataFrame({"year": future_years})

X_train = train["year"].values.reshape(-1, 1)
y_train = train["new_bev_registrations"].values
X_test  = test["year"].values.reshape(-1, 1)
X_future = np.array(future_years).reshape(-1, 1)
X_all_future = np.array(
    list(test["year"]) + future_years
).reshape(-1, 1)

results = {}  # Store all forecasts here


# -- 4. Model 1: Polynomial Linear Regression ----------------------------------
print("\n-- Model 1: Polynomial Linear Regression --------------------")

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_all_poly   = poly.transform(X_all_future)

lr = LinearRegression()
lr.fit(X_train_poly, y_train)

lr_forecast = lr.predict(X_all_poly)
lr_forecast = np.maximum(lr_forecast, 0)  # No negative registrations

results["Linear Regression"] = {
    "years":    list(test["year"]) + future_years,
    "forecast": lr_forecast,
    "color":    COLORS["linear"],
}

# Test set MAE
if len(test) > 0:
    X_test_poly = poly.transform(X_test)
    lr_test_pred = lr.predict(X_test_poly)
    lr_mae = mean_absolute_error(test["new_bev_registrations"], lr_test_pred)
    print(f"  Test MAE: {lr_mae:,.0f} registrations")
    results["Linear Regression"]["mae"] = lr_mae

print(f"  Forecast {future_years[0]}–{future_years[-1]}:")
for y, v in zip(future_years, lr_forecast[-FORECAST_YEARS:]):
    print(f"    {y}: {v:,.0f}")


# -- 5. Model 2: ARIMA ---------------------------------------------------------
if HAS_ARIMA:
    print("\n-- Model 2: ARIMA -------------------------------------------")
    try:
        arima_model = ARIMA(y_train, order=(1, 1, 1))
        arima_fit   = arima_model.fit()

        n_forecast = len(test) + FORECAST_YEARS
        arima_pred = arima_fit.forecast(steps=n_forecast)
        arima_pred = np.maximum(arima_pred, 0)

        arima_years = list(test["year"]) + future_years

        results["ARIMA (1,1,1)"] = {
            "years":    arima_years,
            "forecast": arima_pred,
            "color":    COLORS["arima"],
        }

        if len(test) > 0:
            arima_mae = mean_absolute_error(
                test["new_bev_registrations"], arima_pred[:len(test)]
            )
            print(f"  Test MAE: {arima_mae:,.0f} registrations")
            results["ARIMA (1,1,1)"]["mae"] = arima_mae

        print(f"  Forecast {future_years[0]}–{future_years[-1]}:")
        for y, v in zip(future_years, arima_pred[-FORECAST_YEARS:]):
            print(f"    {y}: {v:,.0f}")

    except Exception as e:
        print(f"  ARIMA failed: {e}")


# -- 6. Model 3: Prophet -------------------------------------------------------
if HAS_PROPHET:
    print("\n-- Model 3: Prophet -----------------------------------------")
    try:
        prophet_df = pd.DataFrame({
            "ds": pd.to_datetime(train["year"].astype(str) + "-01-01"),
            "y":  train["new_bev_registrations"].values,
        })

        m = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.5,
        )
        m.fit(prophet_df)

        n_periods = len(test) + FORECAST_YEARS
        future_df = m.make_future_dataframe(periods=n_periods, freq="YE")
        forecast_df = m.predict(future_df)

        prophet_years = [d.year for d in forecast_df["ds"].tail(n_periods)]
        prophet_pred  = forecast_df["yhat"].tail(n_periods).values
        prophet_lower = forecast_df["yhat_lower"].tail(n_periods).values
        prophet_upper = forecast_df["yhat_upper"].tail(n_periods).values
        prophet_pred  = np.maximum(prophet_pred, 0)

        results["Prophet"] = {
            "years":    prophet_years,
            "forecast": prophet_pred,
            "lower":    prophet_lower,
            "upper":    prophet_upper,
            "color":    COLORS["prophet"],
        }

        if len(test) > 0:
            prophet_mae = mean_absolute_error(
                test["new_bev_registrations"], prophet_pred[:len(test)]
            )
            print(f"  Test MAE: {prophet_mae:,.0f} registrations")
            results["Prophet"]["mae"] = prophet_mae

        print(f"  Forecast {future_years[0]}–{future_years[-1]}:")
        for y, v in zip(future_years, prophet_pred[-FORECAST_YEARS:]):
            print(f"    {y}: {v:,.0f}")

    except Exception as e:
        print(f"  Prophet failed: {e}")


# -- 7. Forecast Comparison Plot -----------------------------------------------
print("\n-- Forecast Comparison Plot ----------------------------------")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("EV Adoption Forecast — Germany (New BEV Registrations)",
             fontsize=15, fontweight="bold")

for ax in axes:
    # Actual data
    ax.fill_between(df["year"], df["new_bev_registrations"],
                    alpha=0.15, color=COLORS["actual"])
    ax.plot(df["year"], df["new_bev_registrations"],
            color=COLORS["actual"], linewidth=2.5, marker="o",
            label="Actual", zorder=5)

    # Train/test split line
    ax.axvline(TRAIN_CUTOFF, color="gray", linestyle=":", alpha=0.7)
    ax.text(TRAIN_CUTOFF + 0.1, ax.get_ylim()[1] * 0.95,
            "← Train | Test ->", fontsize=8, color="gray")

    # Forecasts
    for model_name, res in results.items():
        years    = res["years"]
        forecast = res["forecast"]
        color    = res["color"]

        ax.plot(years, forecast, color=color, linewidth=2,
                linestyle="--", marker="s", markersize=5, label=model_name)

        # Confidence interval (Prophet only)
        if "lower" in res and "upper" in res:
            ax.fill_between(years, res["lower"], res["upper"],
                            alpha=0.15, color=color)

    # Forecast zone shading
    ax.axvspan(df["year"].max(), future_years[-1],
               alpha=0.05, color="gray", label="Forecast Zone")

    ax.set_xlabel("Year")
    ax.set_ylabel("New BEV Registrations")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

axes[0].set_title("Full View (2010–2029)")
axes[1].set_xlim(2019, future_years[-1])
axes[1].set_title("Zoomed: Recent + Forecast (2019–2029)")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_forecast_comparison.png", dpi=150, bbox_inches="tight")
print(f"Saved -> {OUTPUT_DIR}/02_forecast_comparison.png")
plt.close()


# -- 8. Model Comparison Table -------------------------------------------------
print("\n-- Model Performance Summary --------------------------------")

comparison_rows = []
for model_name, res in results.items():
    future_vals = res["forecast"][-FORECAST_YEARS:]
    row = {
        "Model":        model_name,
        "Test MAE":     f"{res.get('mae', 0):,.0f}" if "mae" in res else "N/A",
        f"{future_years[0]} Forecast": f"{future_vals[0]:,.0f}",
        f"{future_years[-1]} Forecast": f"{future_vals[-1]:,.0f}",
    }
    comparison_rows.append(row)

df_comparison = pd.DataFrame(comparison_rows)
print(df_comparison.to_string(index=False))

# Bar chart of model comparison
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(future_years))
width = 0.25

for i, (model_name, res) in enumerate(results.items()):
    future_vals = res["forecast"][-FORECAST_YEARS:]
    offset = (i - len(results) / 2 + 0.5) * width
    bars = ax.bar(x + offset, future_vals, width,
                  label=model_name, color=res["color"], alpha=0.85)

ax.set_title("Forecast Comparison by Year — All Models", fontsize=13, fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Predicted New BEV Registrations")
ax.set_xticks(x)
ax.set_xticklabels(future_years)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_forecast_bar_comparison.png", dpi=150, bbox_inches="tight")
print(f"Saved -> {OUTPUT_DIR}/03_forecast_bar_comparison.png")
plt.close()


# -- 9. Regional Analysis ------------------------------------------------------
print("\n-- Regional Analysis ----------------------------------------")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("EV Adoption by German State (Bundesland) — 2024",
             fontsize=14, fontweight="bold")

df_reg_sorted = df_regional.sort_values("bev_stock_2024", ascending=True)

# Absolute stock
axes[0].barh(df_reg_sorted["state"], df_reg_sorted["bev_stock_2024"],
             color="#2A9D8F", edgecolor="white")
axes[0].set_title("Total BEV Stock 2024")
axes[0].set_xlabel("Number of EVs")
axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

# Per 1000 population (fairer comparison)
df_reg_per = df_regional.sort_values("bev_per_1000_pop", ascending=True)
colors_per = ["#E63946" if v < df_regional["bev_per_1000_pop"].median()
              else "#2A9D8F" for v in df_reg_per["bev_per_1000_pop"]]
axes[1].barh(df_reg_per["state"], df_reg_per["bev_per_1000_pop"],
             color=colors_per, edgecolor="white")
axes[1].axvline(df_regional["bev_per_1000_pop"].median(),
                color="gray", linestyle="--", alpha=0.7, label="Median")
axes[1].set_title("BEV per 1,000 Population (adjusted)")
axes[1].set_xlabel("EVs per 1,000 residents")
axes[1].legend()

red_patch   = mpatches.Patch(color="#E63946", label="Below median (underserved)")
green_patch = mpatches.Patch(color="#2A9D8F", label="Above median")
axes[1].legend(handles=[red_patch, green_patch], fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_regional_analysis.png", dpi=150, bbox_inches="tight")
print(f"Saved -> {OUTPUT_DIR}/04_regional_analysis.png")
plt.close()


# -- 10. Export for Power BI ---------------------------------------------------
print("\n-- Exporting for Power BI -----------------------------------")

# Combine actual + all forecasts into one long-format table
rows = []

# Actual historical
for _, row in df.iterrows():
    rows.append({
        "year":   row["year"],
        "value":  row["new_bev_registrations"],
        "type":   "Actual",
        "model":  "Actual",
        "is_forecast": False,
    })

# Forecasts
for model_name, res in results.items():
    for y, v in zip(res["years"], res["forecast"]):
        rows.append({
            "year":   y,
            "value":  max(v, 0),
            "type":   "Forecast",
            "model":  model_name,
            "is_forecast": True,
        })

df_export = pd.DataFrame(rows)
df_export.to_csv(f"{OUTPUT_DIR}/ev_forecast_powerbi.csv", index=False)
print(f"Forecast export -> {OUTPUT_DIR}/ev_forecast_powerbi.csv")

df_regional.to_csv(f"{OUTPUT_DIR}/ev_regional_powerbi.csv", index=False)
print(f"Regional export -> {OUTPUT_DIR}/ev_regional_powerbi.csv")

df.to_csv(f"{OUTPUT_DIR}/ev_timeseries_powerbi.csv", index=False)
print(f"Time series export -> {OUTPUT_DIR}/ev_timeseries_powerbi.csv")


# -- Summary -------------------------------------------------------------------
print(f"""
{'='*60}
FORECASTING COMPLETE
{'='*60}
Charts saved to outputs/:
  01_ev_eda_overview.png       -> Historical EDA
  02_forecast_comparison.png   -> All models vs actual
  03_forecast_bar_comparison.png -> Model comparison bar chart
  04_regional_analysis.png     -> By Bundesland

Power BI files:
  ev_forecast_powerbi.csv      -> Main forecast table
  ev_regional_powerbi.csv      -> Regional map data
  ev_timeseries_powerbi.csv    -> Full historical series

Power BI Dashboard tips:
  1. Load ev_forecast_powerbi.csv
  2. Line chart: year (X) vs value (Y), legend = model
  3. Slicer: model (to toggle between forecasts)
  4. Load ev_regional_powerbi.csv for map visual (lat/lon)
  5. KPI card: bev_per_1000_pop to show adoption rate
{'='*60}
""")
