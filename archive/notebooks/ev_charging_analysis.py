"""
ev_charging_analysis.py
=======================
Main analysis script for the EV Charging Demand Portfolio Project.

Steps:
    1. Load and explore the data (EDA)
    2. K-means clustering to find charging infrastructure patterns
    3. Identify "charging deserts" (underserved areas)
    4. Export results for Power BI dashboard

Run after fetch_data.py has been executed.

Author: Nisrina Afnan Walyadin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
import os

warnings.filterwarnings("ignore")

# -- Config --------------------------------------------------------------------
DATA_FILE   = "data/ev_stations.csv"
OUTPUT_DIR  = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FOCUS_COUNTRY = "DE"   # Change to any country code for focused analysis
N_CLUSTERS    = 6      # Number of clusters (tuned via elbow method below)

COLORS = {
    "Ultra-Fast (>=100 kW)": "#E63946",
    "Fast (50-99 kW)":       "#F4A261",
    "Medium (22-49 kW)":     "#2A9D8F",
    "Slow (<22 kW)":         "#457B9D",
    "Unknown":               "#CCCCCC",
}

# -- 1. Load Data --------------------------------------------------------------
print("=" * 60)
print("EV CHARGING INFRASTRUCTURE ANALYSIS")
print("=" * 60)

df = pd.read_csv(DATA_FILE, parse_dates=["date_created", "date_last_verified"])
print(f"\nLoaded {len(df):,} stations across {df['country_code'].nunique()} countries.")
print(f"Columns: {list(df.columns)}\n")


# -- 2. Exploratory Data Analysis (EDA) ---------------------------------------
print("-- EDA ------------------------------------------------------")

# 2a. Station count by country
country_counts = df.groupby("country_code").size().sort_values(ascending=False)
print("\nStations per country:")
print(country_counts)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("EV Charging Infrastructure — Exploratory Analysis", fontsize=16, fontweight="bold")

# Plot 1: Stations by country
country_counts.plot(kind="bar", ax=axes[0, 0], color="#2A9D8F", edgecolor="white")
axes[0, 0].set_title("Charging Stations by Country")
axes[0, 0].set_xlabel("")
axes[0, 0].set_ylabel("Number of Stations")
axes[0, 0].tick_params(axis="x", rotation=0)

# Plot 2: Speed category distribution
speed_dist = df["speed_category"].value_counts()
colors_pie  = [COLORS.get(k, "#999") for k in speed_dist.index]
axes[0, 1].pie(speed_dist, labels=speed_dist.index, colors=colors_pie,
               autopct="%1.1f%%", startangle=90)
axes[0, 1].set_title("Charger Speed Distribution (All Countries)")

# Plot 3: Growth over time
yearly = df.groupby("year_created").size().reset_index(name="count")
yearly = yearly[yearly["year_created"].between(2010, 2025)]
axes[1, 0].fill_between(yearly["year_created"], yearly["count"], alpha=0.4, color="#E63946")
axes[1, 0].plot(yearly["year_created"], yearly["count"], color="#E63946", linewidth=2)
axes[1, 0].set_title("EV Charging Station Growth (2010–2025)")
axes[1, 0].set_xlabel("Year")
axes[1, 0].set_ylabel("New Stations Added")

# Plot 4: Power distribution (Germany only)
de = df[df["country_code"] == FOCUS_COUNTRY].copy()
de["max_power_kw"].dropna().clip(upper=350).hist(
    ax=axes[1, 1], bins=40, color="#457B9D", edgecolor="white"
)
axes[1, 1].axvline(100, color="#E63946", linestyle="--", label="100 kW (Ultra-Fast)")
axes[1, 1].axvline(50,  color="#F4A261", linestyle="--", label="50 kW (Fast)")
axes[1, 1].set_title(f"Power Output Distribution — {FOCUS_COUNTRY}")
axes[1, 1].set_xlabel("Max Power (kW)")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_eda_overview.png", dpi=150, bbox_inches="tight")
print(f"Saved -> {OUTPUT_DIR}/01_eda_overview.png")
plt.close()


# -- 3. Germany Deep-Dive ------------------------------------------------------
print(f"\n-- Germany Deep-Dive ----------------------------------------")

de = df[df["country_code"] == FOCUS_COUNTRY].dropna(subset=["lat", "lon"]).copy()
print(f"Germany stations with coordinates: {len(de):,}")

# Top operators in Germany
top_ops = de["operator"].value_counts().head(10)
print("\nTop 10 operators in Germany:")
print(top_ops)

# IONITY-specific: ultra-fast (>=100kW) share
ionity_share = de[de["speed_category"] == "Ultra-Fast (>=100 kW)"]
print(f"\nUltra-Fast (>=100 kW) stations in Germany: {len(ionity_share):,} "
      f"({len(ionity_share)/len(de)*100:.1f}%)")


# -- 4. K-Means Clustering -----------------------------------------------------
print(f"\n-- K-Means Clustering ---------------------------------------")
print("Goal: identify geographic clusters of charging infrastructure\n")

# Features for clustering
features = ["lat", "lon"]
X = de[features].dropna()

# Elbow method to validate N_CLUSTERS
inertias   = []
sil_scores = []
k_range    = range(2, 12)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X, labels))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("K-Means: Choosing Optimal Number of Clusters", fontsize=14, fontweight="bold")

axes[0].plot(k_range, inertias, "o-", color="#2A9D8F")
axes[0].axvline(N_CLUSTERS, color="#E63946", linestyle="--", label=f"k={N_CLUSTERS} (chosen)")
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia")
axes[0].legend()

axes[1].plot(k_range, sil_scores, "o-", color="#457B9D")
axes[1].axvline(N_CLUSTERS, color="#E63946", linestyle="--", label=f"k={N_CLUSTERS} (chosen)")
axes[1].set_title("Silhouette Score (higher = better)")
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_elbow_silhouette.png", dpi=150, bbox_inches="tight")
print(f"Saved -> {OUTPUT_DIR}/02_elbow_silhouette.png")
plt.close()

# Final clustering
km_final = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
de = de.copy()
de.loc[X.index, "cluster"] = km_final.fit_predict(X)
de["cluster"] = de["cluster"].astype("Int64")

# Cluster summary
cluster_summary = de.groupby("cluster").agg(
    num_stations      = ("id", "count"),
    avg_power_kw      = ("max_power_kw", "mean"),
    pct_ultra_fast    = ("speed_category", lambda x: (x == "Ultra-Fast (>=100 kW)").mean() * 100),
    avg_lat           = ("lat", "mean"),
    avg_lon           = ("lon", "mean"),
).round(2)

print("\nCluster Summary:")
print(cluster_summary)
cluster_summary.to_csv(f"{OUTPUT_DIR}/cluster_summary.csv")


# -- 5. Geographic Cluster Map -------------------------------------------------
print(f"\n-- Geographic Cluster Visualisation ------------------------")

cluster_colors = plt.cm.Set1(np.linspace(0, 0.9, N_CLUSTERS))

fig, axes = plt.subplots(1, 2, figsize=(18, 10))
fig.suptitle("EV Charging Infrastructure Clusters — Germany", fontsize=16, fontweight="bold")

# Left: all stations coloured by cluster
for i, (cluster_id, group) in enumerate(de.groupby("cluster")):
    axes[0].scatter(
        group["lon"], group["lat"],
        c=[cluster_colors[int(cluster_id)]],
        s=8, alpha=0.5, label=f"Cluster {int(cluster_id)+1} ({len(group):,})"
    )

# Centroids
centroids = km_final.cluster_centers_
axes[0].scatter(centroids[:, 1], centroids[:, 0],
                marker="*", s=300, c="black", zorder=5, label="Centroid")
axes[0].set_title("Stations by Cluster")
axes[0].set_xlabel("Longitude")
axes[0].set_ylabel("Latitude")
axes[0].legend(loc="lower left", fontsize=8)
axes[0].set_aspect("equal")

# Right: ultra-fast only (IONITY territory)
ultra = de[de["speed_category"] == "Ultra-Fast (>=100 kW)"]
non_ultra = de[de["speed_category"] != "Ultra-Fast (>=100 kW)"]

axes[1].scatter(non_ultra["lon"], non_ultra["lat"],
                c="#DDDDDD", s=5, alpha=0.3, label="Standard/Slow")
axes[1].scatter(ultra["lon"], ultra["lat"],
                c="#E63946", s=20, alpha=0.7, label="Ultra-Fast >=100kW")
axes[1].set_title("Ultra-Fast Charging Stations (>=100 kW)")
axes[1].set_xlabel("Longitude")
axes[1].set_ylabel("Latitude")
axes[1].legend()
axes[1].set_aspect("equal")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_cluster_map.png", dpi=150, bbox_inches="tight")
print(f"Saved -> {OUTPUT_DIR}/03_cluster_map.png")
plt.close()


# -- 6. Charging Desert Detection ---------------------------------------------
print(f"\n-- Charging Desert Detection --------------------------------")
print("Identifying clusters with LOW ultra-fast penetration = underserved areas\n")

threshold = 5.0  # less than 5% ultra-fast = potential desert
deserts = cluster_summary[cluster_summary["pct_ultra_fast"] < threshold].copy()
print(f"Clusters with <{threshold}% ultra-fast coverage (potential charging deserts):")
print(deserts[["num_stations", "avg_power_kw", "pct_ultra_fast", "avg_lat", "avg_lon"]])

# Visualise deserts
fig, ax = plt.subplots(figsize=(10, 12))
ax.scatter(de["lon"], de["lat"], c="#DDDDDD", s=5, alpha=0.2, label="All stations")

for cluster_id, row in deserts.iterrows():
    desert_stations = de[de["cluster"] == cluster_id]
    ax.scatter(desert_stations["lon"], desert_stations["lat"],
               c="#E63946", s=15, alpha=0.6)
    ax.annotate(
        f"⚡ Desert Zone\n({row['num_stations']:,} stations,\n{row['pct_ultra_fast']:.1f}% ultra-fast)",
        xy=(row["avg_lon"], row["avg_lat"]),
        fontsize=9, color="#E63946", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#E63946", alpha=0.8)
    )

ax.set_title("Charging Desert Detection — Germany\n(Red = clusters with <5% ultra-fast coverage)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_charging_deserts.png", dpi=150, bbox_inches="tight")
print(f"\nSaved -> {OUTPUT_DIR}/04_charging_deserts.png")
plt.close()


# -- 7. Export for Power BI ----------------------------------------------------
print(f"\n-- Exporting for Power BI -----------------------------------")

# Main export: all Germany stations with cluster labels
export_cols = [
    "id", "title", "lat", "lon", "city", "state", "postcode",
    "operator", "num_connections", "num_points",
    "max_power_kw", "speed_category", "is_operational",
    "year_created", "cluster", "is_free"
]
export_df = de[[c for c in export_cols if c in de.columns]].copy()
export_df["is_desert"] = export_df["cluster"].isin(deserts.index)

export_df.to_csv(f"{OUTPUT_DIR}/germany_stations_clustered.csv", index=False)
print(f"Main export -> {OUTPUT_DIR}/germany_stations_clustered.csv ({len(export_df):,} rows)")

cluster_summary.to_csv(f"{OUTPUT_DIR}/cluster_summary.csv")
print(f"Cluster summary -> {OUTPUT_DIR}/cluster_summary.csv")

# -- Summary -------------------------------------------------------------------
print(f"""
{'='*60}
ANALYSIS COMPLETE
{'='*60}
Outputs saved to: {OUTPUT_DIR}/
  01_eda_overview.png        -> Overview charts
  02_elbow_silhouette.png    -> Cluster selection validation
  03_cluster_map.png         -> Geographic cluster visualisation
  04_charging_deserts.png    -> Underserved area detection
  germany_stations_clustered.csv  -> Load into Power BI
  cluster_summary.csv        -> Cluster-level KPIs for Power BI

Next steps:
  1. Load germany_stations_clustered.csv into Power BI
  2. Create map visual using lat/lon fields
  3. Add slicers: speed_category, cluster, is_desert, year_created
  4. Build KPI cards: total stations, % ultra-fast, desert zones count
  5. Add folium_map.html for interactive web map (run dashboard/folium_map.py)
{'='*60}
""")
