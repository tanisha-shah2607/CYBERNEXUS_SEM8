"""LLM agent - routes queries to analysis functions. Falls back to keywords without API key."""
import json
import os
import pandas as pd
from tools import (
    detect_anomalies, forecast_sales, sales_by_region, sales_trend,
    summary_stats, threat_breakdown, threat_sales_correlation, top_employees, top_products,
)

TOOL_REGISTRY = {
    "sales_trend": (sales_trend, True),
    "top_products": (top_products, False),
    "top_employees": (top_employees, False),
    "sales_by_region": (sales_by_region, True),
    "threat_breakdown": (threat_breakdown, True),
    "threat_sales_correlation": (threat_sales_correlation, True),
    "detect_anomalies": (detect_anomalies, False),
    "forecast_sales": (forecast_sales, True),
}

TOOL_SCHEMA = [
    {"name": "sales_trend", "description": "Daily sales over time as a line chart.", "input_schema": {"type": "object", "properties": {}}},
    {"name": "top_products", "description": "Products ranked by total sales.", "input_schema": {"type": "object", "properties": {}}},
    {"name": "top_employees", "description": "Employees ranked by total sales.", "input_schema": {"type": "object", "properties": {}}},
    {"name": "sales_by_region", "description": "Sales by region as a bar chart.", "input_schema": {"type": "object", "properties": {}}},
    {"name": "threat_breakdown", "description": "Attack volume by threat type.", "input_schema": {"type": "object", "properties": {}}},
    {"name": "threat_sales_correlation", "description": "Correlation between sales and attacks by region.", "input_schema": {"type": "object", "properties": {}}},
    {"name": "detect_anomalies", "description": "Find unusual rows using IsolationForest.", "input_schema": {"type": "object", "properties": {}}},
    {
        "name": "forecast_sales",
        "description": (
            "Forecast future sales and return a line chart showing history + forecast. "
            "Use for ANY question about future sales, predictions, next year, next quarter, etc. "
            "Forecast always starts from today's real date."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "description": "Days to forecast. 7=week, 30=month, 90=quarter, 365=year."},
                "horizon": {"type": "string", "enum": ["week", "month", "quarter", "year", "next_year"],
                            "description": "Use next_year for 'forecast next year'."},
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are a sales and cybersecurity BI analyst. Pick the single best tool to answer the user's question. "
    "For any future sales question use forecast_sales. Briefly explain what was returned."
)


def _keyword_router(query: str):
    q = query.lower()

    # ── Priority 1: anomaly / outlier (explicit statistical intent) ──────────
    if any(w in q for w in ["anomal", "outlier", "unusual", "suspicious"]):
        return detect_anomalies(), None

    # ── Priority 2: cyber / threat topics ────────────────────────────────────
    # These must come BEFORE forecast so "predict top threats" routes here,
    # not to forecast_sales.
    is_cyber = any(w in q for w in [
        "threat", "attack", "cyber", "malware", "ransomware", "phishing",
        "ddos", "breach", "intrusion", "vulnerability", "incident",
    ])
    if is_cyber:
        if any(w in q for w in ["region", "sale", "correlat", "vs", "versus", "compare"]):
            return threat_sales_correlation()
        # "predict threats", "top threats next month", "forecast attacks" etc.
        # → show the threat breakdown chart; we can't truly forecast threat types
        # without a separate model, so return the best available cyber insight.
        return threat_breakdown()

    # ── Priority 3: sales forecast (only when explicitly about SALES future) ─
    is_forecast = any(w in q for w in [
        "forecast", "predict", "projection", "future", "upcoming",
        "next year", "next month", "next quarter", "next week",
    ])
    is_sales_context = any(w in q for w in [
        "sale", "revenue", "income", "earn", "performance", "growth",
    ])
    if is_forecast and (is_sales_context or not is_cyber):
        # Only forecast if the question is about sales, or generic future query
        if "next year" in q or ("year" in q and "next" in q):
            return forecast_sales(horizon="next_year")
        if "quarter" in q:
            return forecast_sales(horizon="quarter")
        if "month" in q:
            return forecast_sales(horizon="month")
        if "week" in q:
            return forecast_sales(horizon="week")
        return forecast_sales()

    # ── Priority 4: employee / staff ─────────────────────────────────────────
    if any(w in q for w in ["employee", "staff", "rep", "person", "who", "top performer"]):
        return top_employees(), None

    # ── Priority 5: product ──────────────────────────────────────────────────
    if any(w in q for w in ["product", "item", "sku", "license"]):
        return top_products(), None

    # ── Priority 6: region / geography ───────────────────────────────────────
    if any(w in q for w in ["region", "area", "location", "geography", "where"]):
        return sales_by_region()

    # ── Default: sales trend ─────────────────────────────────────────────────
    return sales_trend()


def _call_tool(name: str, args: dict):
    if name not in TOOL_REGISTRY:
        return f"Unknown tool: {name}", None
    fn, _ = TOOL_REGISTRY[name]
    result = fn(**args) if args else fn()
    if isinstance(result, tuple):
        return result
    return result, None


def _llm_agent(query: str):
    from anthropic import Anthropic
    client = Anthropic()
    context = f"Dataset summary: {json.dumps(summary_stats(), default=str)}\nToday: {pd.Timestamp.today().date()}\nQuestion: {query}"
    response = client.messages.create(
        model="claude-sonnet-4-5", max_tokens=1024,
        system=SYSTEM_PROMPT, tools=TOOL_SCHEMA,
        messages=[{"role": "user", "content": context}],
    )
    tool_use = next((b for b in response.content if getattr(b, "type", None) == "tool_use"), None)
    if tool_use is None:
        text = "\n".join(b.text for b in response.content if getattr(b, "type", None) == "text").strip()
        return text or "(No response)", None
    data, chart = _call_tool(tool_use.name, dict(tool_use.input or {}))
    try:
        preview = data.head(20).to_dict(orient="records") if hasattr(data, "to_dict") else str(data)[:2000]
        followup = client.messages.create(
            model="claude-sonnet-4-5", max_tokens=400, system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": context},
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use.id,
                                               "content": json.dumps(preview, default=str)}]},
            ],
        )
        explanation = "\n".join(b.text for b in followup.content if getattr(b, "type", None) == "text").strip()
    except Exception:
        explanation = ""
    return data, chart, explanation


def run_agent(query: str):
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            return _llm_agent(query)
        except Exception as e:
            print(f"[agent] LLM failed, falling back: {e}")
    return _keyword_router(query)