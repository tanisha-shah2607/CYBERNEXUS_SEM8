"""
Analytical functions. Now reads from SQLite instead of CSV.
All public functions unchanged - rest of the app needs no edits.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from db import session_scope
from repositories import SalesRepository


def load_full_df() -> pd.DataFrame:
    with session_scope() as s:
        return SalesRepository(s).fetch_df()


df: pd.DataFrame = load_full_df()


def refresh_df() -> pd.DataFrame:
    global df
    df = load_full_df()
    return df


def sales_trend(data=None):
    d = data if data is not None else df
    trend = d.groupby("date", as_index=False)["sales_amount"].sum()
    return trend, px.line(trend, x="date", y="sales_amount", title="Sales Trend")


def top_products(data=None):
    d = data if data is not None else df
    return d.groupby("product", as_index=False)["sales_amount"].sum().sort_values("sales_amount", ascending=False)


def top_employees(data=None):
    d = data if data is not None else df
    return d.groupby("employee", as_index=False)["sales_amount"].sum().sort_values("sales_amount", ascending=False)


def sales_by_region(data=None):
    d = data if data is not None else df
    region = d.groupby("region", as_index=False)["sales_amount"].sum()
    return region, px.bar(region, x="region", y="sales_amount", title="Sales by Region")


def threat_breakdown(data=None):
    d = data if data is not None else df
    threats = d.groupby("threat_type", as_index=False)["attack_count"].sum().sort_values("attack_count", ascending=False)
    fig = px.bar(threats, x="threat_type", y="attack_count", title="Attack Volume by Threat Type", color="threat_type")
    return threats, fig


def threat_sales_correlation(data=None):
    d = data if data is not None else df
    agg = (d.groupby("region", as_index=False)
           .agg(sales_amount=("sales_amount", "sum"), attack_count=("attack_count", "sum"))
           .sort_values("sales_amount", ascending=False))
    fig = go.Figure()
    fig.add_bar(x=agg["region"], y=agg["sales_amount"], name="Sales ($)", yaxis="y1")
    fig.add_scatter(x=agg["region"], y=agg["attack_count"], name="Attack Count",
                    yaxis="y2", mode="lines+markers", line=dict(width=3))
    fig.update_layout(
        title="Threat-Sales Correlation by Region",
        yaxis=dict(title="Sales ($)"),
        yaxis2=dict(title="Attack Count", overlaying="y", side="right"),
        legend=dict(orientation="h", y=-0.2),
    )
    return agg, fig


THREAT_SEVERITY = {
    "Ransomware": 5.0, "Data Breach": 4.5, "Zero-Day": 4.5,
    "DDoS": 3.0, "Insider Threat": 3.5, "Phishing": 2.0,
    "Malware": 2.5, "SQL Injection": 3.5, "Brute Force": 1.5,
}


def risk_score(data=None) -> pd.DataFrame:
    d = (data if data is not None else df).copy()
    d["severity"] = d["threat_type"].map(lambda t: THREAT_SEVERITY.get(t, 1.0))
    d["risk"] = d["attack_count"] * d["severity"]
    return d


def risk_heatmap(data=None, freq: str = "W"):
    d = risk_score(data)
    if d.empty:
        return d, px.imshow([[0]], title="Risk Heatmap (no data)")
    d["bucket"] = d["date"].dt.to_period(freq).dt.start_time
    pivot = (d.groupby(["region", "bucket"], as_index=False)["risk"]
             .sum()
             .pivot(index="region", columns="bucket", values="risk")
             .fillna(0).sort_index())
    pivot.columns = [c.strftime("%Y-%m-%d") for c in pivot.columns]
    label = {"D": "day", "W": "week", "M": "month"}.get(freq, freq)
    fig = px.imshow(pivot, aspect="auto", color_continuous_scale="OrRd",
                    labels=dict(x="Period", y="Region", color="Risk"),
                    title=f"Threat Risk Heatmap (region × {label})")
    fig.update_xaxes(tickangle=-45)
    fig.update_layout(margin=dict(l=80, r=20, t=60, b=80))
    return pivot, fig


def top_risk_hotspots(data=None, n: int = 5) -> pd.DataFrame:
    d = risk_score(data)
    if d.empty:
        return d
    return (d.groupby(["region", "threat_type"], as_index=False)["risk"]
            .sum().sort_values("risk", ascending=False).head(n))


def detect_anomalies(data=None, contamination: float = 0.05):
    d = (data if data is not None else df).copy()
    if len(d) < 20:
        threshold = d["sales_amount"].mean() + 2 * d["sales_amount"].std()
        return d[d["sales_amount"] > threshold]
    features = d[["sales_amount", "licenses_sold", "attack_count"]].fillna(0)
    model = IsolationForest(contamination=contamination, random_state=42)
    d["anomaly_score"] = model.fit_predict(features)
    return d[d["anomaly_score"] == -1].drop(columns=["anomaly_score"]).sort_values("sales_amount", ascending=False)


def forecast_sales(data=None, days: int = 30, horizon: str = None, anchor_to_today: bool = True):
    if horizon:
        h = horizon.lower().strip().replace(" ", "_")
        if h == "next_year":
            today = pd.Timestamp.today().normalize()
            days = max(1, (pd.Timestamp(year=today.year + 1, month=12, day=31) - today).days)
        else:
            days = {"week": 7, "month": 30, "quarter": 90, "year": 365}.get(h, days)
    days = int(max(1, days))

    d = data if data is not None else df
    daily = d.groupby("date", as_index=False)["sales_amount"].sum().sort_values("date")
    daily["day_index"] = (daily["date"] - daily["date"].min()).dt.days

    # Train on ALL history but only show recent history in the chart so the
    # forecast window is visually prominent. Rule: show 2x the forecast period
    # of history, capped at 180 days. This way a 30-day forecast sits next to
    # 60 days of history (roughly 33% history / 67% forecast on screen).
    history_window = min(max(days * 2, 60), 180)

    model = LinearRegression().fit(daily[["day_index"]].values, daily["sales_amount"].values)

    last_date = daily["date"].max()
    today = pd.Timestamp.today().normalize()
    forecast_start = (
        today + pd.Timedelta(days=1)
        if anchor_to_today and last_date < today
        else last_date + pd.Timedelta(days=1)
    )
    gap = (forecast_start - last_date).days
    last_idx = int(daily["day_index"].max())
    future_idx = np.arange(last_idx + gap, last_idx + gap + days).reshape(-1, 1)
    future_dates = pd.date_range(start=forecast_start, periods=days)
    pred = model.predict(future_idx)
    forecast_df = pd.DataFrame({"date": future_dates, "predicted_sales": pred.round(2)})

    # Trim historical display to the recent window so forecast is prominent.
    cutoff = forecast_start - pd.Timedelta(days=history_window)
    hist_display = daily[daily["date"] >= cutoff][["date", "sales_amount"]].copy()
    hist_display = hist_display.rename(columns={"sales_amount": "value"})
    hist_display["series"] = "Historical (recent)"

    fcast = forecast_df.rename(columns={"predicted_sales": "value"}).copy()
    fcast["series"] = "Forecast"

    combined = pd.concat([hist_display, fcast], ignore_index=True)

    fig = px.line(
        combined, x="date", y="value", color="series",
        title=f"Sales Forecast — next {days} days ({forecast_start.date()} → {future_dates[-1].date()})",
        labels={"value": "Sales ($)", "date": "Date", "series": ""},
        color_discrete_map={"Historical (recent)": "#636EFA", "Forecast": "#FF6B6B"},
    )

    # Make the forecast line thicker and add markers so it stands out clearly.
    for trace in fig.data:
        if trace.name == "Forecast":
            trace.line.width = 3
            trace.mode = "lines+markers"
            trace.marker.size = 5
        else:
            trace.line.width = 1
            trace.opacity = 0.6

    # Add a vertical dashed line at the forecast boundary.
    fig.add_vline(
        x=forecast_start.timestamp() * 1000,
        line_dash="dash",
        line_color="gray",
        annotation_text="Forecast starts",
        annotation_position="top left",
    )

    fig.update_layout(
        legend=dict(orientation="h", y=-0.2),
        hovermode="x unified",
    )

    return forecast_df, fig


def summary_stats(data=None) -> dict:
    d = data if data is not None else df
    if len(d) == 0:
        return {"empty": True}
    return {
        "rows": int(len(d)),
        "date_range": [str(d["date"].min().date()), str(d["date"].max().date())],
        "total_sales": float(d["sales_amount"].sum()),
        "total_attacks": int(d["attack_count"].sum()),
        "regions": sorted(d["region"].unique().tolist()),
        "products": sorted(d["product"].unique().tolist()),
        "top_product": d.groupby("product")["sales_amount"].sum().idxmax(),
        "top_employee": d.groupby("employee")["sales_amount"].sum().idxmax(),
        "top_threat": d.groupby("threat_type")["attack_count"].sum().idxmax(),
        "today": str(pd.Timestamp.today().date()),
    }