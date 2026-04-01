"""
fetch_data.py
=============
Fetches EV charging station data from the Open Charge Map API (free, no auth needed for basic use).
Saves raw data to data/raw_ocm_data.json and a cleaned CSV to data/ev_stations.csv

Usage:
    python utils/fetch_data.py

API docs: https://openchargemap.org/site/develop/api
"""

import requests
import json
import pandas as pd
import os

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "data"
RAW_FILE   = os.path.join(OUTPUT_DIR, "raw_ocm_data.json")
CLEAN_FILE = os.path.join(OUTPUT_DIR, "ev_stations.csv")

# Countries to fetch (ISO codes). Start with DE + neighbouring countries.
COUNTRIES = ["DE", "AT", "NL", "FR", "CH", "PL", "CZ"]

API_URL = "https://api.openchargemap.io/v3/poi/"

# Optional: get a free API key at https://openchargemap.org/site/develop/api
# and paste it below for higher rate limits. Leave as None for anonymous access.
API_KEY = "ae8cdc60-feef-4eaf-8ce4-2ffddbc06179"


# ── Fetch ─────────────────────────────────────────────────────────────────────
def fetch_stations(country_code: str, max_results: int = 5000) -> list:
    """Fetch EV stations for a single country from Open Charge Map."""
    params = {
        "output":         "json",
        "countrycode":    country_code,
        "maxresults":     max_results,
        "compact":        True,
        "verbose":        False,
        "includecomments": False,
    }
    if API_KEY:
        params["key"] = API_KEY

    print(f"  Fetching {country_code}...", end=" ")
    response = requests.get(API_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    print(f"{len(data)} stations found.")
    return data


def fetch_all(countries: list = COUNTRIES) -> list:
    """Fetch stations for all countries and return combined list."""
    all_stations = []
    for code in countries:
        try:
            stations = fetch_stations(code)
            for s in stations:
                s["_country_code"] = code  # inject ISO code since API doesn't return it in compact mode
            all_stations.extend(stations)
        except Exception as e:
            print(f"  WARNING: Failed to fetch {code}: {e}")
    return all_stations


# ── Parse ─────────────────────────────────────────────────────────────────────
def parse_stations(raw: list) -> pd.DataFrame:
    """
    Extract the most useful fields from raw OCM JSON into a flat DataFrame.
    Key fields for our analysis:
        - location (lat, lon, city, country)
        - operator info
        - number of charging points
        - connection types / power levels
        - status (operational / not)
        - date added
    """
    records = []
    for s in raw:
        try:
            addr = s.get("AddressInfo", {})
            conns = s.get("Connections", []) or []

            # Power levels across all connections at this station
            power_levels = [
                c.get("PowerKW") for c in conns if c.get("PowerKW") is not None
            ]
            max_power_kw  = max(power_levels) if power_levels else None
            num_points    = sum(
                c.get("Quantity") or 1 for c in conns
            )

            # Charger speed category
            if max_power_kw is None:
                speed_cat = "Unknown"
            elif max_power_kw >= 100:
                speed_cat = "Ultra-Fast (>=100 kW)"    # IONITY territory
            elif max_power_kw >= 50:
                speed_cat = "Fast (50-99 kW)"
            elif max_power_kw >= 22:
                speed_cat = "Medium (22-49 kW)"
            else:
                speed_cat = "Slow (<22 kW)"

            records.append({
                "id":               s.get("ID"),
                "uuid":             s.get("UUID"),
                "title":            addr.get("Title"),
                "lat":              addr.get("Latitude"),
                "lon":              addr.get("Longitude"),
                "country_code":     s.get("_country_code") or addr.get("CountryCode"),
                "city":             addr.get("Town"),
                "state":            addr.get("StateOrProvince"),
                "postcode":         addr.get("Postcode"),
                "operator":         (s.get("OperatorInfo") or {}).get("Title"),
                "num_connections":  len(conns),
                "num_points":       num_points,
                "max_power_kw":     max_power_kw,
                "speed_category":   speed_cat,
                "is_operational":   (s.get("StatusType") or {}).get("IsOperational"),
                "date_created":     s.get("DateCreated"),
                "date_last_verified": s.get("DateLastVerified"),
                "usage_cost":       s.get("UsageCost"),
                "is_free":          "free" in str(s.get("UsageCost", "")).lower(),
            })
        except Exception:
            continue  # skip malformed records

    df = pd.DataFrame(records)

    # Clean up dates
    for col in ["date_created", "date_last_verified"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df["year_created"] = df["date_created"].dt.year

    return df


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Fetching EV Charging Station Data (Open Charge Map) ===")
    raw = fetch_all(COUNTRIES)

    # Save raw JSON
    with open(RAW_FILE, "w") as f:
        json.dump(raw, f)
    print(f"\nRaw data saved -> {RAW_FILE}  ({len(raw)} stations total)")

    # Parse and save clean CSV
    df = parse_stations(raw)
    df.to_csv(CLEAN_FILE, index=False)
    print(f"Clean CSV saved -> {CLEAN_FILE}  ({len(df)} rows, {df.shape[1]} columns)")
    print("\nSample:")
    sample = df[["country_code", "city", "max_power_kw", "speed_category", "num_points"]].head(10)
    print(sample.to_string().encode("ascii", errors="replace").decode("ascii"))
