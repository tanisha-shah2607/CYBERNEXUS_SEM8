"""
Cyber BI Dashboard - Main Streamlit App
Auth-gated. Tabs: Dashboard | Saved | Scheduled Jobs | Reports | Admin
"""
import json
import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from agent import run_agent
from app_config import settings
from db import JobKind, Role, session_scope
from repositories import AuditLogRepository, SalesRepository
from services import AuthenticatedUser, DashboardService, JobService, ReportService, Scheduler
from tools import (
    refresh_df, risk_heatmap, summary_stats,
    threat_breakdown, threat_sales_correlation, top_risk_hotspots,
)
from ui import force_password_change, get_current_user, login_form, render_sidebar_user_panel

st.set_page_config(page_title="Cyber BI Dashboard", layout="wide")


# Boot background scheduler once per process
@st.cache_resource
def _boot_scheduler():
    s = Scheduler()
    if settings.scheduler_enabled:
        s.start()
    return s

_boot_scheduler()


# ── AUTH GATE ────────────────────────────────────────────────────────────────
user = get_current_user()
if user is None:
    login_form()
    st.stop()

if user.must_change_password:
    force_password_change(user)
    st.stop()


# ── SERVICES ─────────────────────────────────────────────────────────────────
dashboard_svc = DashboardService()
job_svc = JobService()
report_svc = ReportService()


# ── DATA HELPERS ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def load_filter_options():
    with session_scope() as s:
        sales = SalesRepository(s)
        lo, hi = sales.date_bounds()
        return {
            "regions": sales.distinct_regions(),
            "products": sales.distinct_products(),
            "date_min": lo.date() if lo else None,
            "date_max": hi.date() if hi else None,
        }


def query_df(start_date, end_date, regions, products) -> pd.DataFrame:
    with session_scope() as s:
        return SalesRepository(s).fetch_df(
            start_date=start_date, end_date=end_date,
            regions=regions, products=products,
        )


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
opts = load_filter_options()
if opts["date_min"] is None:
    st.error("No data in database. Run: python migrate.py init")
    st.stop()

st.sidebar.title("Filters")
date_range = st.sidebar.date_input(
    "Date range", value=(opts["date_min"], opts["date_max"]),
    min_value=opts["date_min"], max_value=opts["date_max"],
)
start_date, end_date = date_range if (isinstance(date_range, tuple) and len(date_range) == 2) else (opts["date_min"], opts["date_max"])
region_filter = st.sidebar.multiselect("Region", opts["regions"], default=opts["regions"])
product_filter = st.sidebar.multiselect("Product", opts["products"], default=opts["products"])
render_sidebar_user_panel(user)


# ── TABS ──────────────────────────────────────────────────────────────────────
tab_labels = ["Dashboard", "Saved", "Scheduled Jobs", "Reports"]
if user.role == Role.ADMIN:
    tab_labels.append("Admin")
tabs = st.tabs(tab_labels)


# ═══════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════
with tabs[0]:
    filtered_df = query_df(start_date, end_date, region_filter, product_filter)
    st.title("Cyber BI Dashboard")
    st.caption("AI-powered insights for cybersecurity sales operations.")

    if filtered_df.empty:
        st.warning("No data matches the current filters.")
        st.stop()

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Sales", f"${filtered_df['sales_amount'].sum():,.0f}")
    c2.metric("Employees", filtered_df["employee"].nunique())
    c3.metric("Products", filtered_df["product"].nunique())
    c4.metric("Threat Events", f"{filtered_df['attack_count'].sum():,}")
    st.markdown("---")

    # Row 1
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        trend = filtered_df.groupby("date", as_index=False)["sales_amount"].sum()
        st.plotly_chart(px.line(trend, x="date", y="sales_amount", title="Sales Trend"),
                        use_container_width=True, key="t_trend")
    with r1c2:
        reg = filtered_df.groupby("region", as_index=False)["sales_amount"].sum()
        st.plotly_chart(px.bar(reg, x="region", y="sales_amount", title="Sales by Region"),
                        use_container_width=True, key="t_region")

    # Row 2
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        prod = filtered_df.groupby("product", as_index=False)["sales_amount"].sum()
        st.plotly_chart(px.pie(prod, names="product", values="sales_amount", title="Product Share"),
                        use_container_width=True, key="t_product")
    with r2c2:
        emp = filtered_df.groupby("employee", as_index=False)["sales_amount"].sum()
        st.plotly_chart(px.bar(emp, x="employee", y="sales_amount", title="Employee Performance"),
                        use_container_width=True, key="t_emp")

    # Row 3 — cyber
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        _, threat_fig = threat_breakdown(filtered_df)
        st.plotly_chart(threat_fig, use_container_width=True, key="t_threat")
    with r3c2:
        _, corr_fig = threat_sales_correlation(filtered_df)
        st.plotly_chart(corr_fig, use_container_width=True, key="t_corr")

    # Row 4 — risk heatmap
    st.subheader("Threat Risk Heatmap")
    st.caption("Composite risk = attack count × threat severity.")
    hm1, hm2 = st.columns([3, 1])
    with hm2:
        freq_label = st.radio("Bucket", ["Daily", "Weekly", "Monthly"], index=1, key="hm_freq")
        top_n = st.slider("Top hotspots", 3, 15, 5, key="hm_topn")
    with hm1:
        _, heatmap_fig = risk_heatmap(filtered_df, freq={"Daily": "D", "Weekly": "W", "Monthly": "M"}[freq_label])
        st.plotly_chart(heatmap_fig, use_container_width=True, key="t_heatmap")
    hotspots = top_risk_hotspots(filtered_df, n=top_n)
    if not hotspots.empty:
        st.markdown("**Top risk hotspots**")
        st.dataframe(hotspots.rename(columns={"region": "Region", "threat_type": "Threat", "risk": "Risk score"}),
                     use_container_width=True, hide_index=True)

    st.markdown("---")

    # Save dashboard (analysts + admins)
    if user.role in (Role.ADMIN, Role.ANALYST):
        with st.expander("Save this view as a dashboard"):
            d_name = st.text_input("Dashboard name", key="save_name")
            d_desc = st.text_area("Description (optional)", key="save_desc")
            d_public = st.checkbox("Make public", key="save_public")
            if st.button("Save dashboard"):
                try:
                    did = dashboard_svc.save(
                        user=user, name=d_name, description=d_desc or None,
                        is_public=d_public,
                        filter_state={
                            "start_date": str(start_date), "end_date": str(end_date),
                            "regions": region_filter, "products": product_filter,
                        },
                    )
                    st.success(f"Saved as dashboard #{did}")
                except (ValueError, PermissionError) as e:
                    st.error(str(e))

    st.markdown("---")

    # AI Analyst
    st.subheader("Ask the AI Analyst")
    if os.getenv("ANTHROPIC_API_KEY"):
        st.caption("LLM agent active.")
    else:
        st.caption("Running in smart keyword mode. Set ANTHROPIC_API_KEY in your terminal for full LLM.")

    query = st.text_input("Ask about sales, threats, forecasts, employees...")
    if st.button("Analyze") and query:
        with st.spinner("Thinking..."):
            result = run_agent(query)
        st.write("### AI Insight")
        explanation, data, chart = "", None, None
        if isinstance(result, tuple):
            if len(result) == 3:
                data, chart, explanation = result
            elif len(result) == 2:
                data, chart = result
        else:
            data = result
        if explanation:
            st.write(explanation)
        if hasattr(data, "columns"):
            st.dataframe(data, use_container_width=True)
        elif data is not None:
            st.write(data)
        if chart is not None:
            st.plotly_chart(chart, use_container_width=True, key=f"ai_{datetime.now().timestamp()}")

        if user.role in (Role.ADMIN, Role.ANALYST) and data is not None:
            payload = (explanation + "\n\n" if explanation else "") + (
                data.to_string(index=False) if hasattr(data, "to_string") else str(data)
            )
            dl1, dl2 = st.columns(2)
            if dl1.button("Save as PDF"):
                rid, path = report_svc.save(user, title=query[:64], body_text=payload, file_format="pdf")
                with open(path, "rb") as f:
                    st.download_button("Download PDF", f.read(), file_name=path.name, mime="application/pdf")
            if dl2.button("Save as PPT"):
                rid, path = report_svc.save(user, title=query[:64], body_text=payload, file_format="pptx")
                with open(path, "rb") as f:
                    st.download_button("Download PPT", f.read(), file_name=path.name,
                                       mime="application/vnd.openxmlformats-officedocument.presentationml.presentation")


# ═══════════════════════════════════════════════════════════
# TAB 2 — SAVED DASHBOARDS
# ═══════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("Saved Dashboards")
    views = dashboard_svc.list_visible(user)
    if not views:
        st.info("No saved dashboards yet. Save one from the Dashboard tab.")
    for v in views:
        with st.container(border=True):
            st.markdown(f"**{v.name}** · by `{v.owner_email}` · {v.updated_at:%Y-%m-%d %H:%M}")
            if v.description:
                st.caption(v.description)
            st.code(json.dumps(v.filter_state, indent=2), language="json")
            col1, col2, col3 = st.columns(3)
            col1.caption("Public" if v.is_public else "Private")
            if v.can_edit:
                with col2.popover("Share"):
                    se = st.text_input("Share with email", key=f"se_{v.id}")
                    sed = st.checkbox("Can edit", key=f"sed_{v.id}")
                    if st.button("Send", key=f"sb_{v.id}"):
                        try:
                            dashboard_svc.share(user, v.id, se, can_edit=sed)
                            st.success(f"Shared with {se}")
                        except Exception as e:
                            st.error(str(e))
                if col3.button("Delete", key=f"del_{v.id}"):
                    try:
                        dashboard_svc.delete(user, v.id)
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))


# ═══════════════════════════════════════════════════════════
# TAB 3 — SCHEDULED JOBS
# ═══════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("Scheduled Jobs")
    st.caption(f"Scheduler polls every {settings.scheduler_poll_seconds}s.")

    if user.role in (Role.ADMIN, Role.ANALYST):
        with st.expander("Create a new job"):
            j_name = st.text_input("Job name", key="j_name")
            j_kind = st.selectbox("Kind", [k.value for k in JobKind], key="j_kind")
            j_interval = st.number_input("Run every (minutes)", min_value=1, value=60, key="j_interval")
            j_title = st.text_input("Report title", value="Daily summary", key="j_title")
            j_fmt = st.selectbox("Format", ["pdf", "pptx"], key="j_fmt")
            if st.button("Create job"):
                try:
                    kind = JobKind(j_kind)
                    config = {"owner_id": user.id, "title": j_title,
                              "body": f"Scheduled report. Stats: {summary_stats()}",
                              "format": j_fmt} if kind == JobKind.SCHEDULED_REPORT else {}
                    jid = job_svc.create(user=user, name=j_name, kind=kind,
                                         interval_minutes=int(j_interval), config=config)
                    st.success(f"Job #{jid} created.")
                except (ValueError, PermissionError) as e:
                    st.error(str(e))

    jobs = job_svc.list_mine(user)
    if not jobs:
        st.info("No scheduled jobs yet.")
    for j in jobs:
        with st.container(border=True):
            hc1, hc2, hc3 = st.columns([3, 1, 1])
            hc1.markdown(f"**{j.name}** · `{j.kind.value}` · every {j.interval_minutes} min")
            hc2.caption(f"next: {j.next_run_at:%H:%M:%S}")
            if hc3.button("Disable" if j.is_active else "Enable", key=f"tog_{j.id}"):
                job_svc.set_active(user, j.id, not j.is_active)
                st.rerun()
            for r in job_svc.recent_runs(j.id, limit=5):
                icon = {"succeeded": "✓", "failed": "✗", "running": "…", "pending": "·"}.get(r["status"], "·")
                st.write(f"`{icon}` {r['started_at']:%Y-%m-%d %H:%M} — {r['output'] or r['error'] or '(pending)'}")


# ═══════════════════════════════════════════════════════════
# TAB 4 — REPORTS
# ═══════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("My Reports")
    reports = report_svc.list_for_user(user, limit=50)
    if not reports:
        st.info("No reports yet. Generate one from the AI Analyst.")
    for r in reports:
        with st.container(border=True):
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"**{r['title']}** · `{r['file_format']}` · {r['created_at']:%Y-%m-%d %H:%M} · {r['size_bytes']//1024} KB")
            try:
                with open(r["file_path"], "rb") as f:
                    c2.download_button("Download", f.read(),
                                       file_name=os.path.basename(r["file_path"]),
                                       key=f"dl_{r['id']}")
            except FileNotFoundError:
                c2.caption("(missing)")


# ═══════════════════════════════════════════════════════════
# TAB 5 — ADMIN
# ═══════════════════════════════════════════════════════════
if user.role == Role.ADMIN:
    with tabs[4]:
        st.subheader("Admin Panel")

        st.markdown("### Users")
        from services import AuthService
        users_list = AuthService().list_users()
        st.dataframe(
            pd.DataFrame([{"id": u.id, "email": u.email, "role": u.role.value,
                           "must_change_pw": u.must_change_password} for u in users_list]),
            use_container_width=True, hide_index=True,
        )

        st.markdown("### Audit Log (last 100)")
        with session_scope() as s:
            entries = AuditLogRepository(s).recent(limit=100)
            rows = [{"when": a.created_at.strftime("%Y-%m-%d %H:%M:%S"), "user_id": a.user_id,
                     "action": a.action,
                     "target": f"{a.target_type or ''}#{a.target_id or ''}".strip("#"),
                     "details": json.dumps(a.details) if a.details else ""} for a in entries]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("### Database")
        if st.button("Refresh in-memory dataframe"):
            refresh_df()
            st.success("Refreshed.")