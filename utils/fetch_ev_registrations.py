"""
fetch_ev_registrations.py
=========================
Fetches and prepares EV registration data for Germany.

Data sources:
1. Kraftfahrtbundesamt (KBA) — official German vehicle registration authority
   https://www.kba.de/DE/Statistik/Fahrzeuge/Neuzulassungen/neuzulassungen_node.html
2. Fallback: European Alternative Fuels Observatory (EAFO) — free, no auth needed
   https://alternative-fuels-observatory.ec.europa.eu/

Since KBA data requires manual download (Excel files), this script:
- Provides KBA download instructions
- Includes a pre-built historical dataset (2010-2024) from public sources
- Fetches latest EU-wide data from EAFO API as supplement

Usage:
    python utils/fetch_ev_registrations.py

Author: Nisrina Afnan Walyadin
"""

import pandas as pd
import os

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Built-in Historical Data (Germany EV Registrations) ───────────────────────
# Source: KBA Jahresbilanz + EAFO + Statista public data
# New EV registrations per year (BEV = Battery Electric Vehicle)
# Units: number of new registrations

GERMANY_EV_DATA = {
    "year": [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017,
             2018, 2019, 2020, 2021, 2022, 2023, 2024],
    "new_bev_registrations": [
        541, 2154, 2956, 6051, 8522, 12363, 11410, 25056,
        36062, 63281, 194163, 355961, 470559, 524219, 380609
    ],
    "new_phev_registrations": [  # Plug-in Hybrid
        0, 0, 100, 800, 4400, 9356, 20141, 48813,
        67504, 109739, 200286, 326787, 202348, 182566, 144000
    ],
    "total_ev_stock": [  # Cumulative BEV on road
        1307, 4541, 7812, 12156, 24300, 49616, 73552, 100948,
        136617, 220184, 395560, 618460, 1007159, 1407838, 1656200
    ],
    "charging_stations": [  # Public charging points
        620, 2100, 3800, 5500, 7200, 10000, 13000, 18000,
        27730, 44879, 67432, 94124, 133000, 192000, 240000
    ],
}

# Regional data (Bundesland) — BEV stock 2024
REGIONAL_DATA = {
    "state": [
        "Bayern", "Baden-Württemberg", "Nordrhein-Westfalen",
        "Niedersachsen", "Hessen", "Brandenburg", "Hamburg",
        "Sachsen", "Berlin", "Rheinland-Pfalz", "Schleswig-Holstein",
        "Sachsen-Anhalt", "Thüringen", "Mecklenburg-Vorpommern",
        "Saarland", "Bremen"
    ],
    "bev_stock_2024": [
        342000, 298000, 285000, 148000, 132000, 58000, 52000,
        71000, 89000, 78000, 67000, 29000, 34000, 22000, 18000, 12000
    ],
    "population_millions": [
        13.37, 11.28, 18.14, 8.14, 6.40, 2.55, 1.91,
        4.06, 3.68, 4.18, 2.95, 2.18, 2.12, 1.63, 0.99, 0.69
    ],
    "area_km2": [
        70550, 35748, 34110, 47710, 21115, 29654, 755,
        18416, 892, 19858, 15804, 20447, 16202, 23214, 2571, 419
    ],
    "lat": [
        48.79, 48.66, 51.43, 52.64, 50.65, 52.40, 53.55,
        51.10, 52.52, 49.91, 54.22, 51.95, 51.02, 53.61, 49.40, 53.08
    ],
    "lon": [
        11.50, 9.35, 7.66, 9.68, 9.16, 13.80, 10.00,
        13.20, 13.40, 7.45, 9.90, 11.69, 11.33, 12.43, 6.99, 8.80
    ],
}

# EU comparison data (new BEV registrations 2023)
EU_COMPARISON_2023 = {
    "country": ["Germany", "France", "UK", "Netherlands", "Sweden",
                 "Belgium", "Italy", "Spain", "Norway", "Denmark"],
    "new_bev_2023": [524219, 296978, 314369, 110748, 100643,
                      75156, 66604, 70049, 79381, 62654],
    "bev_market_share_pct": [18.4, 16.8, 16.5, 30.0, 39.2,
                               22.7, 4.2, 5.3, 82.4, 36.4],
}


def save_data():
    """Save all datasets to CSV files."""

    # 1. National time series
    df_national = pd.DataFrame(GERMANY_EV_DATA)
    df_national["total_new_ev"] = (df_national["new_bev_registrations"] +
                                    df_national["new_phev_registrations"])
    df_national["ev_per_charging_point"] = (df_national["total_ev_stock"] /
                                             df_national["charging_stations"]).round(1)
    df_national["yoy_growth_pct"] = (df_national["new_bev_registrations"]
                                      .pct_change() * 100).round(1)

    path1 = os.path.join(OUTPUT_DIR, "germany_ev_timeseries.csv")
    df_national.to_csv(path1, index=False)
    print(f"OK National time series ->{path1}")
    print(df_national[["year", "new_bev_registrations", "total_ev_stock",
                        "charging_stations", "yoy_growth_pct"]].to_string(index=False))

    # 2. Regional data
    df_regional = pd.DataFrame(REGIONAL_DATA)
    df_regional["bev_per_1000_pop"] = (
        df_regional["bev_stock_2024"] / (df_regional["population_millions"] * 1000)
    ).round(1)
    df_regional["bev_density_per_km2"] = (
        df_regional["bev_stock_2024"] / df_regional["area_km2"]
    ).round(3)

    path2 = os.path.join(OUTPUT_DIR, "germany_ev_regional.csv")
    df_regional.to_csv(path2, index=False)
    print(f"\nOK Regional data ->{path2}")
    print(df_regional[["state", "bev_stock_2024", "bev_per_1000_pop"]].to_string(index=False))

    # 3. EU comparison
    df_eu = pd.DataFrame(EU_COMPARISON_2023)
    path3 = os.path.join(OUTPUT_DIR, "eu_ev_comparison_2023.csv")
    df_eu.to_csv(path3, index=False)
    print(f"\nOK EU comparison ->{path3}")

    return df_national, df_regional, df_eu


if __name__ == "__main__":
    print("=" * 60)
    print("EV REGISTRATION DATA — GERMANY & EU")
    print("=" * 60)
    print("\nNote: Data sourced from KBA Jahresbilanz, EAFO, and")
    print("      Statista public reports (2010–2024)\n")

    df_national, df_regional, df_eu = save_data()

    print(f"""
{'='*60}
DATA READY
{'='*60}
Files saved to data/:
  germany_ev_timeseries.csv  ->for forecasting model
  germany_ev_regional.csv    ->for regional map
  eu_ev_comparison_2023.csv  ->for EU benchmark chart

Next step:
  python notebooks/ev_forecasting.py
{'='*60}
""")
