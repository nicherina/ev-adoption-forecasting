"""
app.py — EV Adoption Forecasting Dashboard
==========================================
Streamlit app for the EV Adoption Forecasting portfolio project.
Run with: streamlit run app.py

Author: Nisrina Afnan Walyadin
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="EV Adoption Forecast · Germany",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

_black_font = go.layout.Template()
_black_font.layout.font = dict(color="black")
_black_font.layout.xaxis = dict(tickfont=dict(color="black"), title_font=dict(color="black"))
_black_font.layout.yaxis = dict(tickfont=dict(color="black"), title_font=dict(color="black"))
pio.templates["black_font"] = _black_font
pio.templates.default = "plotly_white+black_font"

# ── Constants ─────────────────────────────────────────────────────────────────
COLORS = {
    "actual":  "#2A9D8F",
    "linear":  "#E63946",
    "arima":   "#F4A261",
    "prophet": "#457B9D",
}
MODEL_COLORS = {
    "Actual":             "#2A9D8F",
    "Linear Regression":  "#E63946",
    "ARIMA (1,1,1)":      "#F4A261",
    "Prophet":            "#457B9D",
}
LAYOUT_DEFAULTS = dict(
    font_family="Inter, Arial, sans-serif",
    font_size=12,
    font_color="black",
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df_ts = pd.read_csv("data/germany_ev_timeseries.csv")
    df_region = pd.read_csv("data/germany_ev_regional.csv")
    df_eu = pd.read_csv("data/eu_ev_comparison_2023.csv")
    df_forecast = pd.read_csv("outputs/ev_forecast_powerbi.csv")

    df_forecast["year"] = df_forecast["year"].astype(int)
    df_forecast["value"] = df_forecast["value"].astype(float)

    return df_ts, df_region, df_eu, df_forecast


df_ts, df_region, df_eu, df_forecast = load_data()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ EV Adoption")
    st.caption("Germany · 2025–2029 Forecast")
    st.divider()

    page = st.radio(
        "Navigate",
        ["Overview", "Historical EDA", "Forecast Comparison", "Regional Analysis", "Model Performance"],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("Data: KBA · EAFO · BDEW")
    st.caption("Author: Nisrina Afnan Walyadin")


# ── Page: Overview ────────────────────────────────────────────────────────────
def page_overview():
    st.title("EV Adoption Forecasting — Germany")
    st.markdown("**How many people will buy an EV in the coming years — and what does that mean for charging infrastructure demand?**")
    st.divider()

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Model", "Prophet", "MAE 99,855")
    col2.metric("Forecast 2025", "712,076", "new BEV registrations")
    col3.metric("Forecast 2029", "1,109,549", "new BEV registrations")
    col4.metric("2024 Actual", "380,609", "-27.4% vs 2023", delta_color="inverse")

    st.divider()

    # Overview chart: actuals + Prophet forecast
    actuals = df_ts[["year", "new_bev_registrations"]].copy()
    prophet_fc = df_forecast[
        (df_forecast["model"] == "Prophet") & (df_forecast["is_forecast"])
    ][["year", "value"]].copy()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=actuals["year"], y=actuals["new_bev_registrations"],
        fill="tozeroy", fillcolor="rgba(42,157,143,0.15)",
        line=dict(color=COLORS["actual"], width=2.5),
        mode="lines+markers", name="Actual",
        hovertemplate="%{y:,.0f} registrations<extra>Actual</extra>",
    ))

    fig.add_trace(go.Scatter(
        x=prophet_fc["year"], y=prophet_fc["value"],
        line=dict(color=COLORS["prophet"], width=2, dash="dash"),
        mode="lines+markers", marker_symbol="square", name="Prophet Forecast",
        hovertemplate="%{y:,.0f} registrations<extra>Prophet</extra>",
    ))

    fig.add_vline(x=2024.5, line_dash="dot", line_color="gray",
                  annotation_text="Forecast starts", annotation_position="top right")

    fig.add_annotation(
        x=2024, y=380609,
        text="2024: -27.4% dip<br>(subsidy withdrawal)",
        showarrow=True, arrowhead=2, ax=40, ay=-60,
        font=dict(size=11, color=COLORS["linear"]),
        bgcolor="white", bordercolor=COLORS["linear"], borderwidth=1,
    )

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="New BEV Registrations in Germany — Actual & Prophet Forecast",
        xaxis_title="Year",
        yaxis_title="New Registrations",
        yaxis=dict(tickformat=",.0f"),
        height=420,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Note on 2024 dip:** Germany ended its EV purchase subsidy (Umweltbonus) in December 2023. "
        "New BEV registrations fell from 524,219 in 2023 to 380,609 in 2024 (-27.4%). "
        "All models trained on pre-2022 data overestimated 2024, which is expected given this policy shock."
    )


# ── Page: Historical EDA ──────────────────────────────────────────────────────
def page_eda():
    st.title("Historical EDA")
    st.caption("Germany BEV registration data — 2010 to 2024")
    st.divider()

    # Row 1
    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure(go.Scatter(
            x=df_ts["year"], y=df_ts["new_bev_registrations"],
            fill="tozeroy", fillcolor="rgba(42,157,143,0.15)",
            line=dict(color=COLORS["actual"], width=2.5),
            mode="lines+markers",
            hovertemplate="%{y:,.0f}<extra>New BEV Registrations</extra>",
        ))
        fig.update_layout(**LAYOUT_DEFAULTS, title="New BEV Registrations per Year",
                          yaxis=dict(tickformat=",.0f"), height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure(go.Scatter(
            x=df_ts["year"], y=df_ts["total_ev_stock"],
            fill="tozeroy", fillcolor="rgba(69,123,157,0.15)",
            line=dict(color=COLORS["prophet"], width=2.5),
            mode="lines+markers",
            hovertemplate="%{y:,.0f}<extra>Total BEV Stock</extra>",
        ))
        fig.update_layout(**LAYOUT_DEFAULTS, title="Total BEV Stock on Road",
                          yaxis=dict(tickformat=",.0f"), height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Row 2
    col3, col4 = st.columns(2)

    with col3:
        yoy = df_ts["yoy_growth_pct"].fillna(0)
        bar_colors = [COLORS["linear"] if v < 0 else COLORS["actual"] for v in yoy]
        fig = go.Figure(go.Bar(
            x=df_ts["year"], y=yoy,
            marker_color=bar_colors,
            hovertemplate="%{y:.1f}%<extra>YoY Growth</extra>",
        ))
        fig.add_hline(y=0, line_color="black", line_width=0.8)
        fig.update_layout(**LAYOUT_DEFAULTS, title="Year-on-Year Growth Rate (%)",
                          yaxis_title="Growth (%)", height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_ts["year"], y=df_ts["new_bev_registrations"],
            stackgroup="one", name="BEV (Battery)",
            fillcolor="rgba(42,157,143,0.8)",
            line=dict(color=COLORS["actual"]),
            hovertemplate="%{y:,.0f}<extra>BEV</extra>",
        ))
        fig.add_trace(go.Scatter(
            x=df_ts["year"], y=df_ts["new_phev_registrations"],
            stackgroup="one", name="PHEV (Plug-in Hybrid)",
            fillcolor="rgba(244,162,97,0.8)",
            line=dict(color=COLORS["arima"]),
            hovertemplate="%{y:,.0f}<extra>PHEV</extra>",
        ))
        fig.update_layout(**LAYOUT_DEFAULTS, title="BEV vs PHEV New Registrations",
                          yaxis=dict(tickformat=",.0f"), height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Row 3
    col5, col6 = st.columns(2)

    with col5:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=df_ts["year"], y=df_ts["total_ev_stock"],
            name="EV Stock", line=dict(color=COLORS["actual"], width=2),
            hovertemplate="%{y:,.0f}<extra>EV Stock</extra>",
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=df_ts["year"], y=df_ts["charging_stations"],
            name="Charging Points", line=dict(color=COLORS["linear"], width=2, dash="dash"),
            hovertemplate="%{y:,.0f}<extra>Charging Points</extra>",
        ), secondary_y=True)
        fig.update_yaxes(title_text="Total EVs", tickformat=",.0f", secondary_y=False)
        fig.update_yaxes(title_text="Charging Points", tickformat=",.0f", secondary_y=True)
        fig.update_layout(**LAYOUT_DEFAULTS, title="EV Stock vs Charging Infrastructure",
                          height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        df_eu_sorted = df_eu.sort_values("new_bev_2023", ascending=True)
        bar_colors = [COLORS["linear"] if c == "Germany" else COLORS["actual"]
                      for c in df_eu_sorted["country"]]
        fig = go.Figure(go.Bar(
            x=df_eu_sorted["new_bev_2023"], y=df_eu_sorted["country"],
            orientation="h", marker_color=bar_colors,
            customdata=df_eu_sorted["bev_market_share_pct"],
            hovertemplate="%{x:,.0f} registrations<br>Market share: %{customdata:.1f}%<extra>%{y}</extra>",
        ))
        fig.update_layout(**LAYOUT_DEFAULTS, title="New BEV Registrations 2023 — EU",
                          xaxis=dict(tickformat=",.0f"), height=300)
        st.plotly_chart(fig, use_container_width=True)


# ── Page: Forecast Comparison ─────────────────────────────────────────────────
def page_forecast():
    st.title("Forecast Comparison")
    st.caption("Three models forecasting new BEV registrations through 2029")
    st.divider()

    col_ctrl1, col_ctrl2 = st.columns([2, 1])
    with col_ctrl1:
        selected_models = st.multiselect(
            "Show models",
            ["Linear Regression", "ARIMA (1,1,1)", "Prophet"],
            default=["Linear Regression", "ARIMA (1,1,1)", "Prophet"],
        )
    with col_ctrl2:
        view = st.radio("View", ["Full (2010–2029)", "Zoomed (2019–2029)"], horizontal=True)

    # Main forecast chart
    fig = go.Figure()

    # Actuals
    actuals = df_ts[["year", "new_bev_registrations"]]
    fig.add_trace(go.Scatter(
        x=actuals["year"], y=actuals["new_bev_registrations"],
        fill="tozeroy", fillcolor="rgba(42,157,143,0.1)",
        line=dict(color=COLORS["actual"], width=2.5),
        mode="lines+markers", name="Actual",
        hovertemplate="%{y:,.0f}<extra>Actual</extra>",
    ))

    # Model forecasts
    model_key_map = {
        "Linear Regression": "linear",
        "ARIMA (1,1,1)":     "arima",
        "Prophet":           "prophet",
    }
    for model in selected_models:
        fc = df_forecast[
            (df_forecast["model"] == model) & (df_forecast["is_forecast"])
        ].sort_values("year")
        color = COLORS[model_key_map[model]]
        fig.add_trace(go.Scatter(
            x=fc["year"], y=fc["value"],
            line=dict(color=color, width=2, dash="dash"),
            mode="lines+markers", marker_symbol="square", marker_size=6,
            name=model,
            hovertemplate="%{y:,.0f}<extra>" + model + "</extra>",
        ))

    fig.add_vline(x=2021.5, line_dash="dot", line_color="gray",
                  annotation_text="Train cutoff", annotation_position="top left")
    fig.add_vrect(x0=2024.5, x1=2029.5, fillcolor="gray", opacity=0.04,
                  annotation_text="Forecast Zone", annotation_position="top left")

    if view == "Zoomed (2019–2029)":
        fig.update_xaxes(range=[2019, 2029])

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="New BEV Registrations — Actual vs Forecasts",
        xaxis_title="Year", yaxis_title="New Registrations",
        yaxis=dict(tickformat=",.0f"),
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Model performance + bar comparison side by side
    col_perf, col_bar = st.columns([1, 2])

    with col_perf:
        st.subheader("Model Performance")
        st.caption("Test set: 2022–2024")

        df_perf = pd.DataFrame({
            "Model": ["Linear Regression", "ARIMA (1,1,1)", "Prophet"],
            "Test MAE": [118321, 106331, 99855],
        })
        st.dataframe(
            df_perf.style
                .highlight_min(subset=["Test MAE"], color="#d4f5ef")
                .format({"Test MAE": "{:,.0f}"}),
            hide_index=True,
            use_container_width=True,
        )
        st.success("Prophet: lowest MAE — 99,855 registrations")

    with col_bar:
        st.subheader("Forecast by Year (2025–2029)")
        fc_future = df_forecast[
            (df_forecast["is_forecast"]) & (df_forecast["year"] >= 2025)
        ].copy()

        fig2 = go.Figure()
        for model in ["Linear Regression", "ARIMA (1,1,1)", "Prophet"]:
            subset = fc_future[fc_future["model"] == model]
            color = COLORS[model_key_map[model]]
            fig2.add_trace(go.Bar(
                x=subset["year"], y=subset["value"],
                name=model, marker_color=color,
                hovertemplate="%{y:,.0f}<extra>" + model + "</extra>",
            ))

        fig2.update_layout(
            **LAYOUT_DEFAULTS,
            barmode="group",
            xaxis=dict(tickmode="linear", dtick=1),
            yaxis=dict(tickformat=",.0f"),
            height=300,
        )
        st.plotly_chart(fig2, use_container_width=True)


# ── Page: Regional Analysis ───────────────────────────────────────────────────
def page_regional():
    st.title("Regional Analysis")
    st.caption("BEV adoption across Germany's 16 Bundeslaender (2024)")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        df_sorted = df_region.sort_values("bev_stock_2024", ascending=True)
        fig = go.Figure(go.Bar(
            x=df_sorted["bev_stock_2024"], y=df_sorted["state"],
            orientation="h", marker_color=COLORS["actual"],
            customdata=df_sorted[["bev_per_1000_pop", "population_millions"]].values,
            hovertemplate=(
                "%{x:,.0f} EVs<br>"
                "Per 1,000 pop: %{customdata[0]:.1f}<br>"
                "Population: %{customdata[1]:.2f}M"
                "<extra>%{y}</extra>"
            ),
        ))
        fig.update_layout(**LAYOUT_DEFAULTS, title="Total BEV Stock 2024",
                          xaxis=dict(tickformat=",.0f"), height=500)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        median_val = df_region["bev_per_1000_pop"].median()
        df_sorted2 = df_region.sort_values("bev_per_1000_pop", ascending=True)
        bar_colors = [
            COLORS["linear"] if v < median_val else COLORS["actual"]
            for v in df_sorted2["bev_per_1000_pop"]
        ]
        fig = go.Figure(go.Bar(
            x=df_sorted2["bev_per_1000_pop"], y=df_sorted2["state"],
            orientation="h", marker_color=bar_colors,
            hovertemplate="%{x:.1f} EVs per 1,000 residents<extra>%{y}</extra>",
        ))
        fig.add_vline(x=median_val, line_dash="dash", line_color="gray",
                      annotation_text=f"Median {median_val:.1f}", annotation_position="top right")
        fig.update_layout(**LAYOUT_DEFAULTS, title="BEV per 1,000 Population (adjusted)",
                          xaxis_title="EVs per 1,000 residents", height=500)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Scatter: density vs per-capita
    st.subheader("Urban vs Rural Split")
    fig3 = go.Figure(go.Scatter(
        x=df_region["bev_per_1000_pop"],
        y=df_region["bev_density_per_km2"],
        mode="markers+text",
        text=df_region["state"],
        textposition="top center",
        marker=dict(
            size=df_region["bev_stock_2024"] / 15000,
            color=COLORS["prophet"],
            opacity=0.75,
            line=dict(color="white", width=1),
        ),
        customdata=df_region[["bev_stock_2024", "population_millions"]].values,
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Per 1,000 pop: %{x:.1f}<br>"
            "Density (per km²): %{y:.3f}<br>"
            "Total stock: %{customdata[0]:,.0f}<br>"
            "Population: %{customdata[1]:.2f}M"
            "<extra></extra>"
        ),
    ))
    fig3.update_layout(
        **LAYOUT_DEFAULTS,
        title="BEV per Capita vs BEV Density — bubble size = total stock",
        xaxis_title="BEV per 1,000 population",
        yaxis_title="BEV density (per km²)",
        height=420,
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("City-states (Berlin, Hamburg, Bremen) dominate density but comparable per-capita to Flächenlaender.")


# ── Page: Model Performance ───────────────────────────────────────────────────
def page_model_perf():
    st.title("Model Performance")
    st.caption("How each forecasting model performs on held-out test data (2022–2024)")
    st.divider()

    with st.expander("Methodology", expanded=True):
        st.markdown("""
| | Detail |
|---|---|
| **Training period** | 2010–2021 |
| **Test period** | 2022–2024 |
| **Forecast horizon** | 2025–2029 |
| **Linear Regression** | Polynomial degree=2 |
| **ARIMA** | order=(1,1,1) |
| **Prophet** | changepoint_prior_scale=0.5, no seasonality |
        """)

    st.divider()

    # Test period comparison chart
    st.subheader("Actual vs Forecasts — Test Period (2022–2024)")

    actual_test = df_ts[df_ts["year"].isin([2022, 2023, 2024])][["year", "new_bev_registrations"]]
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=actual_test["year"], y=actual_test["new_bev_registrations"],
        line=dict(color=COLORS["actual"], width=3),
        mode="lines+markers", marker_size=10, name="Actual",
        hovertemplate="%{y:,.0f}<extra>Actual</extra>",
    ))

    model_key_map = {
        "Linear Regression": "linear",
        "ARIMA (1,1,1)":     "arima",
        "Prophet":           "prophet",
    }
    for model, key in model_key_map.items():
        fc = df_forecast[
            (df_forecast["model"] == model) &
            (df_forecast["is_forecast"]) &
            (df_forecast["year"].isin([2022, 2023, 2024]))
        ].sort_values("year")
        fig.add_trace(go.Scatter(
            x=fc["year"], y=fc["value"],
            line=dict(color=COLORS[key], width=2, dash="dash"),
            mode="lines+markers", marker_symbol="square", marker_size=8,
            name=model,
            hovertemplate="%{y:,.0f}<extra>" + model + "</extra>",
        ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        xaxis=dict(tickmode="linear", dtick=1),
        yaxis=dict(tickformat=",.0f"),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Absolute Error by Year")

        actual_vals = {2022: 470559, 2023: 524219, 2024: 380609}
        test_years = [2022, 2023, 2024]

        fig2 = go.Figure()
        for model, key in model_key_map.items():
            fc = df_forecast[
                (df_forecast["model"] == model) &
                (df_forecast["is_forecast"]) &
                (df_forecast["year"].isin(test_years))
            ].sort_values("year")
            errors = [abs(actual_vals[y] - v)
                      for y, v in zip(fc["year"], fc["value"])]
            fig2.add_trace(go.Bar(
                x=test_years, y=errors,
                name=model, marker_color=COLORS[key],
                hovertemplate="%{y:,.0f} error<extra>" + model + "</extra>",
            ))

        fig2.update_layout(
            **LAYOUT_DEFAULTS,
            barmode="group",
            xaxis=dict(tickmode="linear", dtick=1),
            yaxis=dict(tickformat=",.0f", title="Absolute Error"),
            height=320,
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("MAE Summary")

        mae_data = {
            "Model":    ["Linear Regression", "ARIMA (1,1,1)", "Prophet"],
            "Test MAE": [118321, 106331, 99855],
            "Color":    [COLORS["linear"], COLORS["arima"], COLORS["prophet"]],
        }
        df_mae = pd.DataFrame(mae_data).sort_values("Test MAE", ascending=True)

        fig3 = go.Figure(go.Bar(
            x=df_mae["Test MAE"], y=df_mae["Model"],
            orientation="h", marker_color=df_mae["Color"],
            hovertemplate="%{x:,.0f} registrations MAE<extra>%{y}</extra>",
        ))
        fig3.add_annotation(
            x=99855, y="Prophet",
            text="  Best performer",
            showarrow=False, xanchor="left",
            font=dict(size=11, color=COLORS["prophet"]),
        )
        fig3.update_layout(
            **LAYOUT_DEFAULTS,
            xaxis=dict(tickformat=",.0f", title="Mean Absolute Error"),
            height=320,
        )
        st.plotly_chart(fig3, use_container_width=True)


# ── Router ────────────────────────────────────────────────────────────────────
if page == "Overview":
    page_overview()
elif page == "Historical EDA":
    page_eda()
elif page == "Forecast Comparison":
    page_forecast()
elif page == "Regional Analysis":
    page_regional()
elif page == "Model Performance":
    page_model_perf()