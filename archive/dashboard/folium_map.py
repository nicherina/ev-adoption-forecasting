"""
folium_map.py
=============
Generates an interactive HTML map of EV charging stations using Folium.
Opens in any browser — include this in your GitHub Pages portfolio.

Run AFTER ev_charging_analysis.py has been executed.

Usage:
    python dashboard/folium_map.py
"""

import pandas as pd
import folium
from folium.plugins import MarkerCluster, HeatMap
import os

DATA_FILE  = "outputs/germany_stations_clustered.csv"
OUTPUT_MAP = "outputs/ev_charging_map.html"

# -- Load ----------------------------------------------------------------------
df = pd.read_csv(DATA_FILE)
df = df.dropna(subset=["lat", "lon"])

print(f"Loaded {len(df):,} stations for mapping.")

# -- Colour by speed category --------------------------------------------------
SPEED_COLORS = {
    "Ultra-Fast (>=100 kW)": "red",
    "Fast (50-99 kW)":       "orange",
    "Medium (22-49 kW)":     "green",
    "Slow (<22 kW)":         "blue",
    "Unknown":               "gray",
}

SPEED_ICONS = {
    "Ultra-Fast (>=100 kW)": "bolt",
    "Fast (50-99 kW)":       "flash",
    "Medium (22-49 kW)":     "plug",
    "Slow (<22 kW)":         "plug",
    "Unknown":               "question-sign",
}

# -- Build Map -----------------------------------------------------------------
m = folium.Map(
    location=[51.1657, 10.4515],  # Centre of Germany
    zoom_start=6,
    tiles="CartoDB positron"
)

# Layer 1: Heatmap (all stations)
heat_data = df[["lat", "lon"]].values.tolist()
HeatMap(heat_data, radius=8, blur=12, max_zoom=10,
        name="Density Heatmap").add_to(m)

# Layer 2: Clustered markers (all stations)
cluster_group = MarkerCluster(name="All Stations (clustered)").add_to(m)

for _, row in df.iterrows():
    speed   = row.get("speed_category", "Unknown")
    color   = SPEED_COLORS.get(speed, "gray")
    is_desert = row.get("is_desert", False)

    popup_html = f"""
    <div style="font-family:Arial; min-width:200px">
        <b style="font-size:14px">{row.get('title', 'EV Station')}</b><br>
        <hr style="margin:4px 0">
        📍 {row.get('city', '—')}, {row.get('state', '—')}<br>
        ⚡ {speed}<br>
        🔌 {int(row.get('num_points', 1))} charging point(s)<br>
        💡 Max power: {row.get('max_power_kw', '?')} kW<br>
        🏢 {row.get('operator', 'Unknown operator')}<br>
        {'🔴 <b>Charging Desert Zone</b>' if is_desert else ''}
    </div>
    """

    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=5 if speed == "Ultra-Fast (>=100 kW)" else 3,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=folium.Popup(popup_html, max_width=250),
        tooltip=f"{row.get('title', 'Station')} — {speed}",
    ).add_to(cluster_group)

# Layer 3: Ultra-fast only (IONITY territory)
ultra_group = folium.FeatureGroup(name="Ultra-Fast Only (>=100 kW)").add_to(m)
ultra = df[df["speed_category"] == "Ultra-Fast (>=100 kW)"]

for _, row in ultra.iterrows():
    folium.Marker(
        location=[row["lat"], row["lon"]],
        icon=folium.Icon(color="red", icon="bolt", prefix="fa"),
        tooltip=f"⚡ {row.get('title','Station')} — {row.get('max_power_kw','?')} kW",
    ).add_to(ultra_group)

# Layer 4: Charging deserts
desert_group = folium.FeatureGroup(name="Charging Desert Zones").add_to(m)
deserts = df[df["is_desert"] == True]

for _, row in deserts.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=8,
        color="#8B0000",
        fill=True,
        fill_color="#FF4444",
        fill_opacity=0.4,
        tooltip=f"⚠️ Desert Zone: {row.get('city','?')}",
    ).add_to(desert_group)

# Legend
legend_html = """
<div style="position:fixed; bottom:30px; left:30px; z-index:1000;
     background:white; padding:12px 16px; border-radius:8px;
     border:2px solid #ccc; font-family:Arial; font-size:12px;
     box-shadow: 2px 2px 6px rgba(0,0,0,0.3)">
    <b style="font-size:14px">⚡ EV Charging — Germany</b><br>
    <hr style="margin:6px 0">
    🔴 Ultra-Fast (>=100 kW)<br>
    🟠 Fast (50–99 kW)<br>
    🟢 Medium (22–49 kW)<br>
    🔵 Slow (&lt;22 kW)<br>
    ⚠️ Charging Desert Zone<br>
    <hr style="margin:6px 0">
    <small>Source: Open Charge Map</small>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# Layer control
folium.LayerControl(collapsed=False).add_to(m)

# -- Save ----------------------------------------------------------------------
m.save(OUTPUT_MAP)
print(f"\nOK Interactive map saved -> {OUTPUT_MAP}")
print("   Open this file in any browser to explore the map.")
print("   Include it in your GitHub Pages portfolio for maximum impact.")
