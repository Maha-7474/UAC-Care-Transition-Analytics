"""
Care Transition Efficiency & Placement Outcome Analytics
UAC Program — HHS / ORR Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UAC Care Transition Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Main background */
  .stApp { background-color: #0f1117; color: #e0e0e0; }
  [data-testid="stSidebar"] { background-color: #161b27; }
  
  /* KPI cards */
  .kpi-card {
    background: linear-gradient(135deg, #1a2035, #1e2a45);
    border: 1px solid #2a3a5c;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
  }
  .kpi-label { font-size: 11px; text-transform: uppercase; letter-spacing: 1.2px;
                color: #8a9bbf; margin-bottom: 4px; }
  .kpi-value { font-size: 28px; font-weight: 700; color: #e8edf5; line-height: 1; }
  .kpi-delta { font-size: 12px; margin-top: 5px; }
  .kpi-delta.up { color: #4ade80; }
  .kpi-delta.down { color: #f87171; }
  .kpi-delta.neutral { color: #94a3b8; }

  /* Alert badges */
  .alert-danger { background:#3b1212; border-left:4px solid #ef4444;
                  padding:10px 14px; border-radius:6px; margin:6px 0;
                  font-size:13px; color:#fca5a5; }
  .alert-warn   { background:#2d2008; border-left:4px solid #f59e0b;
                  padding:10px 14px; border-radius:6px; margin:6px 0;
                  font-size:13px; color:#fcd34d; }
  .alert-ok     { background:#0d2b1a; border-left:4px solid #22c55e;
                  padding:10px 14px; border-radius:6px; margin:6px 0;
                  font-size:13px; color:#86efac; }

  /* Section headers */
  .section-header {
    font-size: 16px; font-weight: 600; color: #93c5fd;
    text-transform: uppercase; letter-spacing: 0.8px;
    padding: 12px 0 6px; border-bottom: 1px solid #1e3a5f; margin-bottom: 16px;
  }
  .page-title { font-size:26px; font-weight:700; color:#e8edf5; }
  .page-subtitle { font-size:13px; color:#64748b; margin-top:2px; }
  
  /* Plotly overrides to dark */
  .js-plotly-plot .plotly .bg { fill: transparent !important; }
  
  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { gap: 8px; }
  .stTabs [data-baseweb="tab"] {
    background: #1a2035; border-radius: 8px 8px 0 0;
    color: #8a9bbf; font-size: 13px;
  }
  .stTabs [aria-selected="true"] { background: #1e3a5f !important; color: #93c5fd !important; }
</style>
""", unsafe_allow_html=True)

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,17,23,0.6)",
        font=dict(color="#c0cce0", size=11),
        xaxis=dict(gridcolor="#1e2d45", linecolor="#2a3a5c", zerolinecolor="#1e2d45"),
        yaxis=dict(gridcolor="#1e2d45", linecolor="#2a3a5c", zerolinecolor="#1e2d45"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8a9bbf")),
        margin=dict(t=40, b=40, l=50, r=20),
        colorway=["#3b82f6","#22d3ee","#a78bfa","#f59e0b","#34d399","#f87171"],
    )
)

COLORS = {
    "cbp_in":  "#f59e0b",
    "cbp_stk": "#fb923c",
    "transfer":"#3b82f6",
    "hhs_stk": "#a78bfa",
    "dischg":  "#22d3ee",
    "backlog":  "#f87171",
    "good":    "#4ade80",
}


# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("uac_data.csv", parse_dates=["Date"])
    df.columns = [c.strip() for c in df.columns]
    df = df.sort_values("Date").reset_index(drop=True)

    # Short aliases
    df["apprehended"]  = df["Children apprehended and placed in CBP custody"]
    df["cbp_stock"]    = df["Children in CBP custody"]
    df["transfers"]    = df["Children transferred out of CBP custody"]
    df["hhs_stock"]    = df["Children in HHS Care"]
    df["discharges"]   = df["Children discharged from HHS Care"]

    # Fiscal Year
    df["FY"] = df["Date"].apply(lambda d: d.year if d.month < 10 else d.year + 1)
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    df["Weekday"] = df["Date"].dt.day_name()
    df["IsWeekend"] = df["Date"].dt.weekday >= 5
    df["Quarter"] = "Q" + df["Date"].dt.quarter.astype(str)

    # ── KPI metrics ────────────────────────────────────────────────────────────
    # Transfer Efficiency Ratio: transfers ÷ CBP stock
    df["transfer_efficiency"] = np.where(
        df["cbp_stock"] > 0, df["transfers"] / df["cbp_stock"], np.nan
    )
    # Discharge Effectiveness: discharges ÷ HHS stock
    df["discharge_effectiveness"] = np.where(
        df["hhs_stock"] > 0, df["discharges"] / df["hhs_stock"], np.nan
    )
    # Pipeline Throughput: total exits ÷ total entries
    total_exits   = df["transfers"] + df["discharges"]
    total_entries = df["apprehended"] + df["transfers"]
    df["pipeline_throughput"] = np.where(
        total_entries > 0, total_exits / total_entries, np.nan
    )
    # Backlog Accumulation Rate: (apprehended - discharges) 7-day rolling
    df["net_daily_change"] = df["apprehended"] - df["discharges"]
    df["backlog_rate_7d"]  = df["net_daily_change"].rolling(7).mean()

    # Rolling averages
    for col in ["transfer_efficiency","discharge_effectiveness","pipeline_throughput"]:
        df[f"{col}_30d"] = df[col].rolling(30).mean()

    # Outcome stability: rolling std of discharge_effectiveness (30d)
    df["outcome_stability"] = 1 - df["discharge_effectiveness"].rolling(30).std().fillna(0) * 10
    df["outcome_stability"] = df["outcome_stability"].clip(0, 1)

    return df

df_full = load_data()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏥 UAC Analytics")
    st.markdown("---")

    min_d, max_d = df_full["Date"].min().date(), df_full["Date"].max().date()
    date_range = st.date_input(
        "📅 Date Range",
        value=(min_d, max_d),
        min_value=min_d,
        max_value=max_d,
    )

    st.markdown("---")
    st.markdown("**Metric Toggles**")
    show_30d = st.toggle("30-Day Rolling Average", value=True)
    show_raw = st.toggle("Raw Daily Values", value=False)

    st.markdown("---")
    st.markdown("**Threshold Alerts**")
    ter_thresh = st.slider("Transfer Efficiency Alert (< x)", 0.3, 1.5, 0.5, 0.05)
    de_thresh  = st.slider("Discharge Effectiveness Alert (< x)", 0.01, 0.10, 0.02, 0.005,
                           format="%.3f")

    st.markdown("---")
    st.markdown("**Fiscal Year Filter**")
    all_fy = sorted(df_full["FY"].unique())
    selected_fy = st.multiselect("Fiscal Years", all_fy, default=all_fy)

    st.markdown("---")
    st.caption("Data: HHS / ORR UAC Program  \nDashboard v1.0")

# ── Filter ─────────────────────────────────────────────────────────────────────
if len(date_range) == 2:
    s, e = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    s, e = df_full["Date"].min(), df_full["Date"].max()

df = df_full[(df_full["Date"] >= s) & (df_full["Date"] <= e) &
             (df_full["FY"].isin(selected_fy))].copy()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="page-title">Care Transition Efficiency & Placement Outcome Analytics</div>'
    '<div class="page-subtitle">U.S. HHS — Unaccompanied Children Program  •  CBP → HHS → Sponsor Pipeline</div>',
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

# ── KPI Row ────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

latest = df.iloc[-1] if len(df) > 0 else df_full.iloc[-1]
prev   = df.iloc[-8] if len(df) > 8 else df.iloc[0]

def kpi(col, label, val, prev_val, fmt=".2f", invert=False):
    pct = (val - prev_val) / (abs(prev_val) + 1e-9) * 100
    direction = "up" if pct > 0 else "down"
    if invert: direction = "down" if pct > 0 else "up"
    arrow = "▲" if pct > 0 else "▼"
    with col:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{format(val, fmt)}</div>
          <div class="kpi-delta {direction}">{arrow} {abs(pct):.1f}% vs 7d ago</div>
        </div>""", unsafe_allow_html=True)

kpi(k1, "Transfer Efficiency Ratio",
    latest["transfer_efficiency_30d"] or 0,
    prev["transfer_efficiency_30d"] or 0)
kpi(k2, "Discharge Effectiveness Index",
    (latest["discharge_effectiveness_30d"] or 0),
    (prev["discharge_effectiveness_30d"] or 0), fmt=".3f")
kpi(k3, "Pipeline Throughput Rate",
    latest["pipeline_throughput_30d"] or 0,
    prev["pipeline_throughput_30d"] or 0)
kpi(k4, "Backlog Accumulation (7d avg)",
    latest["backlog_rate_7d"] or 0,
    prev["backlog_rate_7d"] or 0, fmt=".0f", invert=True)
kpi(k5, "Outcome Stability Score",
    latest["outcome_stability"] or 0,
    prev["outcome_stability"] or 0)

st.markdown("<br>", unsafe_allow_html=True)

# ── Threshold Alerts ───────────────────────────────────────────────────────────
ter_now = latest["transfer_efficiency_30d"] or 0
de_now  = latest["discharge_effectiveness_30d"] or 0
alert_col1, alert_col2, alert_col3 = st.columns(3)

with alert_col1:
    if ter_now < ter_thresh:
        st.markdown(f'<div class="alert-danger">⚠️ Transfer Efficiency ({ter_now:.2f}) below threshold ({ter_thresh:.2f}) — CBP pipeline may be stalled</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-ok">✓ Transfer Efficiency ({ter_now:.2f}) within acceptable range</div>', unsafe_allow_html=True)

with alert_col2:
    if de_now < de_thresh:
        st.markdown(f'<div class="alert-danger">⚠️ Discharge Effectiveness ({de_now:.3f}) below threshold ({de_thresh:.3f}) — HHS placement bottleneck</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-ok">✓ Discharge Effectiveness ({de_now:.3f}) within acceptable range</div>', unsafe_allow_html=True)

with alert_col3:
    backlog_now = latest["backlog_rate_7d"] or 0
    if backlog_now > 200:
        st.markdown(f'<div class="alert-danger">⚠️ Backlog accumulating at {backlog_now:.0f}/day (7d avg) — system inflow exceeds exits</div>', unsafe_allow_html=True)
    elif backlog_now > 50:
        st.markdown(f'<div class="alert-warn">⚡ Moderate backlog ({backlog_now:.0f}/day) — monitor closely</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-ok">✓ Backlog rate ({backlog_now:.0f}/day) is stable</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Pipeline Flow",
    "⚡  Transition Efficiency",
    "🚧  Bottleneck Detection",
    "📈  Outcome Trends",
    "📋  Data Explorer",
])

# ────────────────────────────────────────────────────────────────────────────────
# TAB 1 — Pipeline Flow
# ────────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">Care Pipeline Flow Visualization</div>', unsafe_allow_html=True)

    # Sankey-style monthly aggregated pipeline
    monthly = df.groupby("Month").agg(
        apprehended=("apprehended","sum"),
        transfers=("transfers","sum"),
        hhs_stock_avg=("hhs_stock","mean"),
        discharges=("discharges","sum"),
    ).reset_index()

    # — Stock trend chart —
    fig_stock = go.Figure()
    fig_stock.update_layout(**PLOTLY_TEMPLATE["layout"],
                            title="Active Care Loads: CBP Stock vs HHS Stock (Daily)",
                            height=340)
    fig_stock.add_trace(go.Scatter(
        x=df["Date"], y=df["cbp_stock"],
        name="CBP Custody", fill="tozeroy",
        fillcolor="rgba(251,146,60,0.15)",
        line=dict(color=COLORS["cbp_stk"], width=1.5),
    ))
    fig_stock.add_trace(go.Scatter(
        x=df["Date"], y=df["hhs_stock"],
        name="HHS Care", fill="tozeroy",
        fillcolor="rgba(167,139,250,0.12)",
        line=dict(color=COLORS["hhs_stk"], width=1.5),
    ))
    st.plotly_chart(fig_stock, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        # Flow: Apprehensions vs Transfers vs Discharges monthly
        fig_flow = go.Figure()
        fig_flow.update_layout(**PLOTLY_TEMPLATE["layout"],
                               title="Monthly Flow: Apprehensions → Transfers → Discharges",
                               height=320, barmode="group")
        fig_flow.add_trace(go.Bar(x=monthly["Month"], y=monthly["apprehended"],
                                  name="Apprehended", marker_color=COLORS["cbp_in"]))
        fig_flow.add_trace(go.Bar(x=monthly["Month"], y=monthly["transfers"],
                                  name="Transferred to HHS", marker_color=COLORS["transfer"]))
        fig_flow.add_trace(go.Bar(x=monthly["Month"], y=monthly["discharges"],
                                  name="Discharged to Sponsor", marker_color=COLORS["dischg"]))
        fig_flow.update_xaxes(nticks=12, tickangle=45)
        st.plotly_chart(fig_flow, use_container_width=True)

    with col_b:
        # Funnel per fiscal year
        fy_agg = df.groupby("FY").agg(
            apprehended=("apprehended","sum"),
            transferred=("transfers","sum"),
            discharged=("discharges","sum"),
        ).reset_index()

        fig_funnel = go.Figure()
        fig_funnel.update_layout(**PLOTLY_TEMPLATE["layout"],
                                 title="Annual Pipeline Funnel by Fiscal Year",
                                 height=320)
        for _, row in fy_agg.iterrows():
            fig_funnel.add_trace(go.Bar(
                name=f"FY{int(row.FY)}",
                x=["Apprehended","Transferred","Discharged"],
                y=[row.apprehended, row.transferred, row.discharged],
            ))
        fig_funnel.update_layout(barmode="group")
        st.plotly_chart(fig_funnel, use_container_width=True)

    # Cumulative flow chart
    fig_cum = go.Figure()
    fig_cum.update_layout(**PLOTLY_TEMPLATE["layout"],
                          title="Cumulative Pipeline Totals",
                          height=300)
    fig_cum.add_trace(go.Scatter(
        x=df["Date"], y=df["apprehended"].cumsum(),
        name="Total Apprehended", line=dict(color=COLORS["cbp_in"])))
    fig_cum.add_trace(go.Scatter(
        x=df["Date"], y=df["transfers"].cumsum(),
        name="Total Transferred", line=dict(color=COLORS["transfer"])))
    fig_cum.add_trace(go.Scatter(
        x=df["Date"], y=df["discharges"].cumsum(),
        name="Total Discharged", line=dict(color=COLORS["dischg"])))
    st.plotly_chart(fig_cum, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────────
# TAB 2 — Transition Efficiency
# ────────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Transfer & Discharge Efficiency Panels</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fig_ter = go.Figure()
        fig_ter.update_layout(**PLOTLY_TEMPLATE["layout"],
                              title="Transfer Efficiency Ratio (Transfers ÷ CBP Stock)",
                              height=300)
        if show_raw:
            fig_ter.add_trace(go.Scatter(
                x=df["Date"], y=df["transfer_efficiency"],
                name="Daily", opacity=0.25,
                line=dict(color=COLORS["transfer"], width=0.8)))
        if show_30d:
            fig_ter.add_trace(go.Scatter(
                x=df["Date"], y=df["transfer_efficiency_30d"],
                name="30d Avg", line=dict(color=COLORS["transfer"], width=2)))
        fig_ter.add_hline(y=ter_thresh, line_dash="dot",
                         annotation_text=f"Alert Threshold ({ter_thresh})",
                         line_color="#ef4444", annotation_font_color="#ef4444")
        st.plotly_chart(fig_ter, use_container_width=True)

    with col2:
        fig_de = go.Figure()
        fig_de.update_layout(**PLOTLY_TEMPLATE["layout"],
                             title="Discharge Effectiveness Index (Discharges ÷ HHS Stock)",
                             height=300)
        if show_raw:
            fig_de.add_trace(go.Scatter(
                x=df["Date"], y=df["discharge_effectiveness"],
                name="Daily", opacity=0.25,
                line=dict(color=COLORS["dischg"], width=0.8)))
        if show_30d:
            fig_de.add_trace(go.Scatter(
                x=df["Date"], y=df["discharge_effectiveness_30d"],
                name="30d Avg", line=dict(color=COLORS["dischg"], width=2)))
        fig_de.add_hline(y=de_thresh, line_dash="dot",
                         annotation_text=f"Alert Threshold ({de_thresh:.3f})",
                         line_color="#ef4444", annotation_font_color="#ef4444")
        st.plotly_chart(fig_de, use_container_width=True)

    # Pipeline throughput
    fig_pt = go.Figure()
    fig_pt.update_layout(**PLOTLY_TEMPLATE["layout"],
                         title="Pipeline Throughput Rate (Total Exits ÷ Total Entries) — 30d Rolling",
                         height=280)
    if show_raw:
        fig_pt.add_trace(go.Scatter(
            x=df["Date"], y=df["pipeline_throughput"],
            name="Daily", opacity=0.2, line=dict(color="#a78bfa", width=0.8)))
    if show_30d:
        fig_pt.add_trace(go.Scatter(
            x=df["Date"], y=df["pipeline_throughput_30d"],
            name="30d Avg", line=dict(color="#a78bfa", width=2.2)))
    fig_pt.add_hline(y=1.0, line_dash="dash",
                     annotation_text="Breakeven (1.0)",
                     line_color="#22c55e", annotation_font_color="#22c55e")
    st.plotly_chart(fig_pt, use_container_width=True)

    # Weekday efficiency
    st.markdown('<div class="section-header">Weekday vs Weekend Efficiency</div>', unsafe_allow_html=True)

    wd_eff = df.groupby("Weekday").agg(
        ter=("transfer_efficiency","mean"),
        de=("discharge_effectiveness","mean"),
    ).reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])

    fig_wd = make_subplots(rows=1, cols=2,
                           subplot_titles=("Transfer Efficiency by Weekday",
                                           "Discharge Effectiveness by Weekday"))
    fig_wd.add_trace(go.Bar(x=wd_eff.index, y=wd_eff["ter"],
                            marker_color=[COLORS["cbp_stk"] if d in ["Saturday","Sunday"]
                                          else COLORS["transfer"] for d in wd_eff.index],
                            name="Transfer Eff"), row=1, col=1)
    fig_wd.add_trace(go.Bar(x=wd_eff.index, y=wd_eff["de"],
                            marker_color=[COLORS["cbp_stk"] if d in ["Saturday","Sunday"]
                                          else COLORS["dischg"] for d in wd_eff.index],
                            name="Discharge Eff"), row=1, col=2)
    fig_wd.update_layout(**PLOTLY_TEMPLATE["layout"], height=280, showlegend=False)
    st.plotly_chart(fig_wd, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────────
# TAB 3 — Bottleneck Detection
# ────────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Backlog & Bottleneck Detection</div>', unsafe_allow_html=True)

    # Backlog accumulation chart
    fig_bl = go.Figure()
    fig_bl.update_layout(**PLOTLY_TEMPLATE["layout"],
                         title="Net Daily Change: Apprehensions minus Discharges (Backlog Indicator)",
                         height=300)
    fig_bl.add_trace(go.Bar(
        x=df["Date"], y=df["net_daily_change"],
        marker_color=np.where(df["net_daily_change"] > 0, "#f87171", "#4ade80"),
        name="Net Change",
    ))
    fig_bl.add_trace(go.Scatter(
        x=df["Date"], y=df["backlog_rate_7d"],
        name="7d Moving Avg", line=dict(color="#f59e0b", width=2),
    ))
    fig_bl.add_hline(y=0, line_color="#64748b", line_width=1)
    st.plotly_chart(fig_bl, use_container_width=True)

    col_b1, col_b2 = st.columns(2)

    with col_b1:
        # HHS stock heatmap by month-FY
        hhs_pivot = df.groupby(["FY", df["Date"].dt.month])["hhs_stock"].mean().unstack()
        hhs_pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                              "Jul","Aug","Sep","Oct","Nov","Dec"][:len(hhs_pivot.columns)]

        fig_heat = px.imshow(
            hhs_pivot,
            color_continuous_scale="RdYlGn_r",
            title="HHS Stock Heatmap (Avg by FY × Month)",
            aspect="auto",
            labels=dict(color="Avg Children"),
        )
        fig_heat.update_layout(**PLOTLY_TEMPLATE["layout"], height=320,
                               coloraxis_colorbar=dict(tickfont=dict(color="#8a9bbf")))
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_b2:
        # Transfer lag: ratio of CBP stock to daily transfers
        df["transfer_lag_days"] = np.where(
            df["transfers"] > 0, df["cbp_stock"] / df["transfers"], np.nan
        )
        lag_monthly = df.groupby("Month")["transfer_lag_days"].median().reset_index()

        fig_lag = go.Figure()
        fig_lag.update_layout(**PLOTLY_TEMPLATE["layout"],
                              title="Estimated CBP Transfer Lag (Days)",
                              height=320)
        fig_lag.add_trace(go.Scatter(
            x=lag_monthly["Month"], y=lag_monthly["transfer_lag_days"],
            fill="tozeroy", fillcolor="rgba(248,113,113,0.15)",
            line=dict(color=COLORS["backlog"], width=2),
            name="Median Lag (days)",
        ))
        fig_lag.add_hline(y=3, line_dash="dash", line_color="#fcd34d",
                         annotation_text="72h Legal Requirement",
                         annotation_font_color="#fcd34d")
        fig_lag.update_xaxes(nticks=12, tickangle=45)
        st.plotly_chart(fig_lag, use_container_width=True)

    # Monthly inflow vs outflow
    m_io = df.groupby("Month").agg(
        inflow=("apprehended","sum"),
        outflow=("discharges","sum"),
    ).reset_index()
    m_io["gap"] = m_io["inflow"] - m_io["outflow"]

    fig_gap = go.Figure()
    fig_gap.update_layout(**PLOTLY_TEMPLATE["layout"],
                          title="Monthly Inflow vs Outflow Gap (Sustained Imbalance)",
                          height=280)
    fig_gap.add_trace(go.Bar(x=m_io["Month"], y=m_io["gap"],
                             marker_color=np.where(m_io["gap"] > 0, "#f87171", "#4ade80"),
                             name="Inflow − Outflow"))
    fig_gap.add_trace(go.Scatter(x=m_io["Month"],
                                  y=m_io["gap"].rolling(3, center=True).mean(),
                                  line=dict(color="#f59e0b", width=2.5),
                                  name="3-mo Rolling"))
    fig_gap.update_xaxes(nticks=16, tickangle=45)
    st.plotly_chart(fig_gap, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────────
# TAB 4 — Outcome Trends
# ────────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">Placement Outcome Trend Analysis</div>', unsafe_allow_html=True)

    col_o1, col_o2 = st.columns(2)

    with col_o1:
        # Monthly discharge performance
        monthly_d = df.groupby("Month").agg(
            discharges=("discharges","sum"),
            hhs_avg=("hhs_stock","mean"),
        ).reset_index()
        monthly_d["monthly_de"] = monthly_d["discharges"] / (monthly_d["hhs_avg"] * 30 + 1)

        fig_md = go.Figure()
        fig_md.update_layout(**PLOTLY_TEMPLATE["layout"],
                             title="Monthly Discharge Volume & Effectiveness",
                             height=310)
        fig_md.add_trace(go.Bar(x=monthly_d["Month"], y=monthly_d["discharges"],
                                name="Discharges", marker_color=COLORS["dischg"],
                                yaxis="y"))
        fig_md.add_trace(go.Scatter(x=monthly_d["Month"], y=monthly_d["monthly_de"],
                                    name="Effectiveness Rate", line=dict(color="#f59e0b", width=2),
                                    yaxis="y2"))
        fig_md.update_layout(
            yaxis=dict(title="Discharges", color=COLORS["dischg"]),
            yaxis2=dict(title="Rate", overlaying="y", side="right",
                        color="#f59e0b"),
        )
        fig_md.update_xaxes(nticks=12, tickangle=45)
        st.plotly_chart(fig_md, use_container_width=True)

    with col_o2:
        # Outcome Stability Score
        fig_os = go.Figure()
        fig_os.update_layout(**PLOTLY_TEMPLATE["layout"],
                             title="Outcome Stability Score (30d Rolling)",
                             height=310)
        fig_os.add_trace(go.Scatter(
            x=df["Date"], y=df["outcome_stability"],
            fill="tozeroy", fillcolor="rgba(34,211,238,0.10)",
            line=dict(color=COLORS["dischg"], width=2),
            name="Stability",
        ))
        fig_os.add_hline(y=0.6, line_dash="dot", line_color="#f59e0b",
                         annotation_text="Low-Stability Threshold",
                         annotation_font_color="#f59e0b")
        fig_os.update_yaxes(range=[0, 1.05])
        st.plotly_chart(fig_os, use_container_width=True)

    # FY-over-FY discharge comparison
    fy_disc = df.groupby("FY")["discharges"].agg(["sum","mean","std"]).reset_index()
    fy_disc.columns = ["FY","Total","Daily Avg","Std Dev"]

    fig_fy = go.Figure()
    fig_fy.update_layout(**PLOTLY_TEMPLATE["layout"],
                         title="Fiscal Year Discharge Performance (Total & Variability)",
                         height=300)
    fig_fy.add_trace(go.Bar(x=fy_disc["FY"].astype(str), y=fy_disc["Total"],
                            name="Total Discharged", marker_color=COLORS["dischg"],
                            error_y=dict(type="data", array=fy_disc["Std Dev"] * 365 / 12,
                                        color="#64748b")))
    st.plotly_chart(fig_fy, use_container_width=True)

    # Sudden drops detection
    df["dischg_30d"] = df["discharges"].rolling(30).mean()
    df["dischg_drop"] = df["dischg_30d"] < (df["dischg_30d"].shift(30) * 0.70)

    drops = df[df["dischg_drop"]]
    if len(drops) > 0:
        st.markdown("**🔴 Detected Significant Discharge Drops (>30% decline month-over-month)**")
        drop_summary = drops.groupby(drops["Date"].dt.to_period("M")).size().reset_index()
        drop_summary.columns = ["Period", "Alert Days"]
        st.dataframe(drop_summary.tail(12).set_index("Period"), use_container_width=True)
    else:
        st.success("✓ No significant sustained discharge drops detected in selected period.")

    # Seasonal pattern
    seasonal = df.groupby(df["Date"].dt.month).agg(
        avg_discharges=("discharges","mean"),
        avg_apprehended=("apprehended","mean"),
    ).reset_index()
    seasonal["Month_Name"] = pd.to_datetime(seasonal["Date"].astype(str), format="%m").dt.strftime("%b")

    fig_sea = go.Figure()
    fig_sea.update_layout(**PLOTLY_TEMPLATE["layout"],
                          title="Seasonal Pattern: Average Daily Apprehensions vs Discharges",
                          height=280)
    fig_sea.add_trace(go.Scatter(x=seasonal["Month_Name"], y=seasonal["avg_apprehended"],
                                 name="Apprehended", line=dict(color=COLORS["cbp_in"],width=2),
                                 mode="lines+markers"))
    fig_sea.add_trace(go.Scatter(x=seasonal["Month_Name"], y=seasonal["avg_discharges"],
                                 name="Discharged", line=dict(color=COLORS["dischg"],width=2),
                                 mode="lines+markers"))
    st.plotly_chart(fig_sea, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────────
# TAB 5 — Data Explorer
# ────────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">Data Explorer & Summary Statistics</div>', unsafe_allow_html=True)

    # Summary stats by FY
    summary = df.groupby("FY").agg(
        Total_Apprehended=("apprehended","sum"),
        Total_Transferred=("transfers","sum"),
        Total_Discharged=("discharges","sum"),
        Avg_CBP_Stock=("cbp_stock","mean"),
        Avg_HHS_Stock=("hhs_stock","mean"),
        Avg_Transfer_Efficiency=("transfer_efficiency","mean"),
        Avg_Discharge_Effectiveness=("discharge_effectiveness","mean"),
        Avg_Throughput=("pipeline_throughput","mean"),
    ).reset_index()

    for c in ["Avg_CBP_Stock","Avg_HHS_Stock"]:
        summary[c] = summary[c].round(0).astype(int)
    for c in ["Avg_Transfer_Efficiency","Avg_Discharge_Effectiveness","Avg_Throughput"]:
        summary[c] = summary[c].round(4)

    st.dataframe(
        summary.set_index("FY"),
        use_container_width=True,
        height=300,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Raw Data Sample (filtered)**")
    display_cols = [
        "Date","apprehended","cbp_stock","transfers",
        "hhs_stock","discharges",
        "transfer_efficiency","discharge_effectiveness","pipeline_throughput",
    ]
    st.dataframe(
        df[display_cols].tail(200).set_index("Date").style.format({
            "transfer_efficiency": "{:.3f}",
            "discharge_effectiveness": "{:.4f}",
            "pipeline_throughput": "{:.3f}",
        }),
        use_container_width=True, height=350,
    )

    csv = df[display_cols].to_csv(index=False)
    st.download_button("⬇️ Download Filtered Data as CSV", csv,
                       "uac_filtered_data.csv", "text/csv")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#374151;font-size:11px;">'
    'Care Transition Efficiency & Placement Outcome Analytics  •  '
    'U.S. Department of Health and Human Services — Office of Refugee Resettlement  •  '
    'UAC Program Data  •  Built with Streamlit & Plotly'
    '</div>',
    unsafe_allow_html=True,
)
