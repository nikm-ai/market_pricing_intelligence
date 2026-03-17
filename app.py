import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from generate_data import generate_marketplace_data
from model import fit_model

st.set_page_config(
    page_title="Rental Pricing Efficiency: A Gradient Boosting Approach",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  /* ── Hide sidebar entirely ── */
  [data-testid="stSidebar"] { display: none !important; }
  [data-testid="collapsedControl"] { display: none !important; }

  /* ── Base / typography ── */
  @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500;600&display=swap');

  .stApp { background: #111110; }
  .block-container {
    padding-top: 3rem; padding-bottom: 4rem;
    max-width: 1160px; background: #111110;
  }

  /* All default streamlit text */
  .stApp, .stApp p, .stApp div, .stApp span,
  .stMarkdown, .stMarkdown p {
    color: #e8e4dc !important;
  }

  /* ── Paper title block ── */
  .paper-title {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 30px; font-weight: 500; line-height: 1.25;
    color: #e8e4dc; margin-bottom: 0.35rem; letter-spacing: -0.01em;
  }
  .paper-byline {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; color: #6b6660;
    margin-bottom: 1.75rem; letter-spacing: 0.04em;
    text-transform: uppercase;
  }

  /* ── Abstract ── */
  .abstract-box {
    border-top: 0.5px solid rgba(232,228,220,0.12);
    border-bottom: 0.5px solid rgba(232,228,220,0.12);
    padding: 1rem 0; margin-bottom: 2.25rem;
  }
  .abstract-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px; font-weight: 500; letter-spacing: 0.16em;
    text-transform: uppercase; color: #6b6660; margin-bottom: 6px;
  }
  .abstract-text {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 15px; line-height: 1.8; color: #a09b90; max-width: 860px;
  }

  /* ── Filter bar ── */
  .filter-bar {
    background: #191917;
    border: 0.5px solid rgba(232,228,220,0.10);
    border-radius: 4px;
    padding: 1rem 1.25rem;
    margin-bottom: 2.5rem;
  }
  .filter-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px; font-weight: 500; letter-spacing: 0.16em;
    text-transform: uppercase; color: #6b6660;
    margin-bottom: 0.75rem; display: block;
  }
  .filter-divider {
    border: none;
    border-top: 0.5px solid rgba(232,228,220,0.08);
    margin: 1rem 0;
  }

  /* ── Section headers ── */
  .sec-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px; font-weight: 500; letter-spacing: 0.16em;
    text-transform: uppercase; color: #6b6660;
    margin: 2.5rem 0 1rem; padding-bottom: 6px;
    border-bottom: 0.5px solid rgba(232,228,220,0.10);
    display: block;
  }

  /* ── KPI cards ── */
  .kpi-card {
    border: 0.5px solid rgba(232,228,220,0.10);
    border-radius: 3px; padding: 1rem 1.1rem;
    background: #191917;
  }
  .kpi-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px; font-weight: 500; letter-spacing: 0.12em;
    text-transform: uppercase; color: #6b6660; margin-bottom: 6px;
  }
  .kpi-value {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 32px; font-weight: 400; line-height: 1; color: #e8e4dc;
  }
  .kpi-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; margin-top: 5px; color: #6b6660;
  }
  .pos { color: #6aacad !important; font-weight: 500; }
  .neg { color: #c97a6a !important; font-weight: 500; }
  .acc { color: #c8b97a !important; font-weight: 500; }

  /* ── Scenario control panel ── */
  .ctrl-panel {
    background: #191917;
    border: 0.5px solid rgba(232,228,220,0.10);
    border-radius: 4px;
    padding: 0.9rem 1.25rem;
    margin-bottom: 1.5rem;
  }

  /* ── Figure captions ── */
  .fig-caption {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 13.5px; line-height: 1.75; color: #6b6660;
    margin-top: 0.25rem; margin-bottom: 1.25rem;
    font-style: italic;
  }
  .fig-caption b { font-style: normal; font-weight: 600; color: #a09b90; }

  /* ── Segment cards ── */
  .seg-card {
    border: 0.5px solid rgba(232,228,220,0.10);
    border-radius: 3px; padding: 0.8rem 1rem; margin-bottom: 6px;
    background: #191917;
  }
  .seg-name  { font-family: 'IBM Plex Mono', monospace; font-size: 11px;
               font-weight: 500; color: #e8e4dc; letter-spacing: 0.04em; }
  .seg-count { font-family: 'IBM Plex Mono', monospace; font-size: 11px;
               color: #6b6660; }
  .seg-price { font-family: 'EB Garamond', Georgia, serif; font-size: 14px;
               color: #a09b90; margin-top: 4px; }
  .seg-lift  { font-family: 'IBM Plex Mono', monospace; font-size: 11px;
               color: #6b6660; margin-top: 3px; }

  /* ── Model KPI cards ── */
  .model-kpi {
    border: 0.5px solid rgba(232,228,220,0.10); border-radius: 3px;
    padding: 0.85rem 1rem; background: #191917; text-align: center;
  }
  .model-kpi-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px; font-weight: 500; letter-spacing: 0.12em;
    text-transform: uppercase; color: #6b6660; margin-bottom: 5px;
  }
  .model-kpi-value {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 28px; font-weight: 400; color: #e8e4dc;
  }
  .model-kpi-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; color: #6b6660; margin-top: 3px;
  }
  .model-kpi-good { color: #6aacad !important; }
  .model-kpi-mid  { color: #c8b97a !important; }

  /* ── Note sections ── */
  .note-head {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; font-weight: 500; letter-spacing: 0.1em;
    text-transform: uppercase; color: #6b6660;
    margin-bottom: 8px;
  }
  .note-body {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 14px; line-height: 1.75; color: #a09b90;
  }

  /* ── Footer ── */
  .paper-footer {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; color: #6b6660;
    margin-top: 3rem; padding-top: 1rem;
    border-top: 0.5px solid rgba(232,228,220,0.10);
    line-height: 1.7;
  }

  /* ── Streamlit widget overrides ── */
  /* Labels */
  .stSelectbox label, .stMultiSelect label,
  .stSlider label, .stNumberInput label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 10px !important; font-weight: 500 !important;
    letter-spacing: 0.10em !important;
    text-transform: uppercase !important;
    color: #6b6660 !important;
  }

  /* Input backgrounds */
  .stSelectbox > div > div,
  .stMultiSelect > div > div {
    background: #222220 !important;
    border: 0.5px solid rgba(232,228,220,0.15) !important;
    border-radius: 3px !important;
    color: #e8e4dc !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
  }

  /* Slider track */
  .stSlider [data-baseweb="slider"] [role="slider"] {
    background: #c8b97a !important;
    border-color: #c8b97a !important;
  }
  .stSlider [data-baseweb="slider"] div[data-testid="stTickBar"] {
    color: #6b6660 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 10px !important;
  }

  /* Multiselect tags */
  [data-baseweb="tag"] {
    background: #2c2c29 !important;
    border: 0.5px solid rgba(232,228,220,0.15) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    color: #a09b90 !important;
  }

  /* Tab buttons */
  [data-testid="stTabs"] button {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important; font-weight: 500 !important;
    letter-spacing: 0.08em !important; text-transform: uppercase !important;
    color: #6b6660 !important;
  }
  [data-testid="stTabs"] button[aria-selected="true"] {
    color: #c8b97a !important;
    border-bottom-color: #c8b97a !important;
  }

  /* Dataframe */
  .stDataFrame { border: 0.5px solid rgba(232,228,220,0.10) !important; }

  /* Metric */
  [data-testid="metric-container"] {
    background: #191917;
    border: 0.5px solid rgba(232,228,220,0.10);
    border-radius: 3px; padding: 0.75rem 1rem;
  }
  [data-testid="metric-container"] label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 9px !important; letter-spacing: 0.12em !important;
    text-transform: uppercase !important; color: #6b6660 !important;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'EB Garamond', Georgia, serif !important;
    font-size: 26px !important; color: #e8e4dc !important;
  }

  /* Expander */
  [data-testid="stExpander"] {
    background: #191917 !important;
    border: 0.5px solid rgba(232,228,220,0.10) !important;
    border-radius: 4px !important;
  }
  [data-testid="stExpander"] summary {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 10px !important; font-weight: 500 !important;
    letter-spacing: 0.12em !important; text-transform: uppercase !important;
    color: #6b6660 !important;
  }
  [data-testid="stExpander"] summary:hover { color: #c8b97a !important; }
</style>
""", unsafe_allow_html=True)

# ── Chart helpers ──────────────────────────────────────────────────────────
FONT   = dict(size=11, color="#a09b90", family="IBM Plex Mono, monospace")
LEGEND = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
              font=dict(size=10, color="#a09b90", family="IBM Plex Mono, monospace"),
              bgcolor="rgba(0,0,0,0)")
BASE   = dict(
    plot_bgcolor="#191917", paper_bgcolor="#191917", font=FONT,
    margin=dict(l=8, r=8, t=16, b=8), legend=LEGEND,
)

def ax(title, grid=True):
    return dict(
        title=dict(text=title, font=dict(size=10, color="#6b6660",
                   family="IBM Plex Mono, monospace")),
        tickfont=dict(size=10, color="#6b6660", family="IBM Plex Mono, monospace"),
        gridcolor="rgba(232,228,220,0.06)" if grid else "rgba(0,0,0,0)",
        linecolor="rgba(232,228,220,0.12)", linewidth=1, showline=True,
        showgrid=grid, zeroline=False, ticks="outside", ticklen=3,
    )

SEG_COLORS = {"Budget": "#6aacad", "Mid-Market": "#c8b97a", "Premium": "#a09b90"}
SEG_ORDER  = ["Budget", "Mid-Market", "Premium"]

# ── Data + Model ───────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return generate_marketplace_data(n=2000)

df_full = load_data()

with st.spinner("Fitting model..."):
    ma = fit_model(df_full)

# ── Title block ────────────────────────────────────────────────────────────
st.markdown("""
<div class="paper-title">Rental Pricing Efficiency in Multi-Market Residential Portfolios</div>
<div class="paper-byline">Gradient boosting approach &nbsp;·&nbsp; Mispricing detection &nbsp;·&nbsp; Revenue optimization</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="abstract-box">
  <div class="abstract-label">Abstract</div>
  <div class="abstract-text">
    Manual rent-setting in residential property management produces systematic pricing
    inefficiencies at scale: a portion of the portfolio is priced above market equilibrium,
    generating vacancy drag, while another portion is priced below, forgoing attainable revenue.
    This analysis applies a Gradient Boosting Regressor trained on observable listing features
    to predict market-clearing monthly rent across {len(df_full):,} synthetic listings in five
    U.S. markets. The model achieves a cross-validated R\u00b2 of {ma['r2_mean']:.3f}
    (MAE\u202f=\u202f${ma['mae_mean']:.0f}/month), outperforming a Ridge regression baseline
    (R\u00b2\u202f=\u202f{ma['baseline_r2_mean']:.3f}) by
    {(ma['r2_mean']-ma['baseline_r2_mean'])*100:.1f} percentage points.
    Deviations exceeding five percent from model estimates are flagged as pricing gaps.
  </div>
</div>
""", unsafe_allow_html=True)

# ── Inline filter bar ──────────────────────────────────────────────────────
with st.expander("Filter parameters", expanded=True):
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        cities = st.multiselect(
            "City", sorted(df_full["city"].unique()),
            default=sorted(df_full["city"].unique())
        )
    with fc2:
        prop_types = st.multiselect(
            "Property type", ["Studio", "1BR", "2BR", "3BR+"],
            default=["Studio", "1BR", "2BR", "3BR+"]
        )
    with fc3:
        segments = st.multiselect(
            "Market segment", ["Budget", "Mid-Market", "Premium"],
            default=["Budget", "Mid-Market", "Premium"]
        )

    fs1, fs2, fs3 = st.columns(3)
    with fs1:
        dem_range = st.slider("Demand score range", 0, 100, (0, 100))
    with fs2:
        elasticity = st.slider(
            "Demand elasticity", 0.5, 2.0, 1.0, 0.1,
            help="Price elasticity of demand. Higher values indicate greater occupancy sensitivity to price."
        )
    with fs3:
        occ_target = st.slider("Target occupancy (%)", 70, 98, 88)

# ── Filter ─────────────────────────────────────────────────────────────────
df = df_full[
    df_full["city"].isin(cities) &
    df_full["property_type"].isin(prop_types) &
    df_full["segment"].isin(segments) &
    df_full["demand_score"].between(dem_range[0], dem_range[1])
].copy()

if elasticity != 1.0 or occ_target != 88:
    adj = occ_target / 100 - 0.88
    df["recommended_price"]          = (df["recommended_price"] * (1 + adj * 0.5 / elasticity)).round(0).astype(int)
    df["annual_revenue_recommended"] = (df["recommended_price"] * (occ_target / 100) * 12).round(0).astype(int)
    df["annual_revenue_lift"]        = df["annual_revenue_recommended"] - df["annual_revenue_current"]
    df["price_gap_pct"]              = ((df["recommended_price"] - df["current_price"]) / df["current_price"] * 100).round(1)

if len(df) == 0:
    st.warning("No listings satisfy the current filter criteria.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════
tab_dash, tab_model = st.tabs(["Results and Analysis", "Model Performance"])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1: RESULTS
# ══════════════════════════════════════════════════════════════════════════
with tab_dash:

    # ── Portfolio summary ──────────────────────────────────────────────────
    st.markdown('<div class="sec-header">1 &nbsp;·&nbsp; Portfolio summary statistics</div>', unsafe_allow_html=True)

    total_lift  = df["annual_revenue_lift"].sum()
    med_cur     = df["current_price"].median()
    med_rec     = df["recommended_price"].median()
    med_demand  = df["demand_score"].median()
    pct_over    = (df["price_gap_pct"] < -5).mean() * 100
    pct_under   = (df["price_gap_pct"] > 5).mean() * 100
    avg_occ     = df["occupancy_rate"].mean() * 100
    n_over      = int((df["price_gap_pct"] < -5).sum())
    n_under     = int((df["price_gap_pct"] > 5).sum())
    price_delta = med_rec - med_cur
    delta_pct   = price_delta / med_cur * 100
    occ_gap     = avg_occ - occ_target
    lift_word   = "net gain" if total_lift >= 0 else "net drag"

    k1, k2, k3, k4 = st.columns(4)
    lift_cls = "pos" if total_lift >= 0 else "neg"
    delta_cls = "pos" if price_delta >= 0 else "neg"
    with k1:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">Est. annual revenue impact</div>
          <div class="kpi-value <{lift_cls}>">${total_lift/1e6:.2f}M</div>
          <div class="kpi-sub">full-adoption repricing scenario</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">Median model-recommended rent</div>
          <div class="kpi-value">${med_rec:,.0f}</div>
          <div class="kpi-sub">vs. <span class="{delta_cls}">${med_cur:,.0f} current ({delta_pct:+.1f}%)</span></div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">Listings with material pricing gap</div>
          <div class="kpi-value">{pct_over + pct_under:.0f}%</div>
          <div class="kpi-sub">
            <span class="neg">{pct_over:.0f}% above threshold</span> &nbsp;·&nbsp;
            <span class="pos">{pct_under:.0f}% below threshold</span>
          </div>
        </div>""", unsafe_allow_html=True)
    with k4:
        occ_dir = f"{abs(occ_gap):.1f} pts {'above' if occ_gap >= 0 else 'below'} target"
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">Mean portfolio occupancy</div>
          <div class="kpi-value">{avg_occ:.1f}%</div>
          <div class="kpi-sub">{occ_dir} &nbsp;·&nbsp; demand index {med_demand:.0f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div class="fig-caption">
      <b>Table 1.</b> Portfolio-level statistics for the filtered sample (N\u202f=\u202f{len(df):,} listings,
      {len(cities)} {'market' if len(cities)==1 else 'markets'}).
      A listing is classified as mispriced if its current rent deviates from the model estimate
      by more than five percent. Of {len(df):,} listings, {n_over:,} ({pct_over:.0f}%) are above
      the threshold and {n_under:,} ({pct_under:.0f}%) are below it. Under full adoption,
      systematic repricing generates an estimated {lift_word} of ${abs(total_lift)/1e6:.2f}M annually.
    </div>""", unsafe_allow_html=True)

    # ── Pricing distribution ───────────────────────────────────────────────
    st.markdown('<div class="sec-header">2 &nbsp;·&nbsp; Rent distribution by market segment</div>', unsafe_allow_html=True)

    seg_sum = df.groupby("segment", observed=True).agg(
        listings   = ("current_price",       "count"),
        med_cur    = ("current_price",        "median"),
        med_rec    = ("recommended_price",    "median"),
        total_lift = ("annual_revenue_lift",  "sum"),
        avg_occ    = ("occupancy_rate",       "mean"),
    ).reindex(SEG_ORDER)

    big_seg  = seg_sum["total_lift"].abs().idxmax() if not seg_sum.empty else "Mid-Market"
    big_lift = seg_sum.loc[big_seg, "total_lift"] if big_seg in seg_sum.index else 0

    col_box, col_seg = st.columns([3, 2])
    with col_box:
        fig_box = go.Figure()
        for seg in SEG_ORDER:
            sd = df[df["segment"] == seg]
            if len(sd) == 0: continue
            fig_box.add_trace(go.Box(
                y=sd["current_price"], name=f"{seg} (current)",
                marker_color=SEG_COLORS[seg], opacity=0.85,
                legendgroup=seg, boxmean=True,
            ))
            fig_box.add_trace(go.Box(
                y=sd["recommended_price"], name=f"{seg} (model estimate)",
                marker_color=SEG_COLORS[seg], opacity=0.32,
                legendgroup=seg, boxmean=True,
            ))
        fig_box.update_layout(**BASE, height=360,
            yaxis=dict(**ax("Monthly rent (USD)")),
            xaxis=dict(showgrid=False,
                       tickfont=dict(size=10, color="#6b6660", family="IBM Plex Mono, monospace"),
                       linecolor="rgba(232,228,220,0.12)", linewidth=1, showline=True),
        )
        st.plotly_chart(fig_box, use_container_width=True)

        st.markdown(f"""<div class="fig-caption">
          <b>Figure 1.</b> Box plots of current listed rents (solid) versus model-estimated
          market-clearing rents (translucent) by segment. The {big_seg} segment exhibits the
          largest aggregate pricing gap (estimated ${abs(big_lift)/1e3:.0f}K annual impact
          under full correction).
        </div>""", unsafe_allow_html=True)

    with col_seg:
        for seg in SEG_ORDER:
            if seg not in seg_sum.index or pd.isna(seg_sum.loc[seg, "med_cur"]):
                continue
            row = seg_sum.loc[seg]
            lift = row["total_lift"]
            dp   = (row["med_rec"] - row["med_cur"]) / row["med_cur"] * 100
            dc   = "pos" if lift >= 0 else "neg"
            ls   = f"${lift/1e3:+.0f}K" if abs(lift) < 1e6 else f"${lift/1e6:+.2f}M"
            st.markdown(f"""<div class="seg-card">
              <div style="display:flex;justify-content:space-between;align-items:baseline;">
                <span class="seg-name">{seg}</span>
                <span class="seg-count">N\u202f=\u202f{int(row['listings']):,}</span>
              </div>
              <div class="seg-price">
                ${row['med_cur']:,.0f} &rarr; <b>${row['med_rec']:,.0f}</b>
                <span class="{dc}"> ({dp:+.1f}%)</span>
              </div>
              <div class="seg-lift">
                Revenue impact: <span class="{dc}">{ls}</span> &nbsp;·&nbsp;
                Occupancy: {row['avg_occ']*100:.1f}%
              </div>
            </div>""", unsafe_allow_html=True)

    # ── Opportunity map ────────────────────────────────────────────────────
    st.markdown('<div class="sec-header">3 &nbsp;·&nbsp; Mispricing across the demand distribution</div>', unsafe_allow_html=True)

    hd_op = len(df[(df["demand_score"] > 60) & (df["price_gap_pct"] < -5)])
    ld_up = len(df[(df["demand_score"] < 40) & (df["price_gap_pct"] > 5)])

    col_sc, col_city = st.columns([3, 2])
    with col_sc:
        rng_j = np.random.default_rng(7)
        samp  = df.sample(min(400, len(df)), random_state=42).copy()
        samp["demand_j"] = samp["demand_score"] + rng_j.uniform(-0.6, 0.6, len(samp))
        samp["gap_j"]    = samp["price_gap_pct"] + rng_j.uniform(-0.2, 0.2, len(samp))

        fig_sc = go.Figure()
        fig_sc.add_shape(type="rect", x0=60, x1=100, y0=5,  y1=55,
                         fillcolor="rgba(106,172,173,0.06)", line_width=0)
        fig_sc.add_shape(type="rect", x0=60, x1=100, y0=-55, y1=-5,
                         fillcolor="rgba(200,185,122,0.06)", line_width=0)
        fig_sc.add_shape(type="rect", x0=0,  x1=40,  y0=5,  y1=55,
                         fillcolor="rgba(106,172,173,0.03)", line_width=0)
        fig_sc.add_shape(type="rect", x0=0,  x1=40,  y0=-55, y1=-5,
                         fillcolor="rgba(201,122,106,0.04)", line_width=0)

        fig_sc.add_hline(y=5,  line_dash="dash", line_color="#6aacad", line_width=1.2,
                         annotation_text="+5% threshold", annotation_position="top right",
                         annotation_font=dict(size=9, color="#6aacad",
                                              family="IBM Plex Mono, monospace"))
        fig_sc.add_hline(y=-5, line_dash="dash", line_color="#c97a6a", line_width=1.2,
                         annotation_text="-5% threshold", annotation_position="bottom right",
                         annotation_font=dict(size=9, color="#c97a6a",
                                              family="IBM Plex Mono, monospace"))
        fig_sc.add_hline(y=0, line_color="rgba(232,228,220,0.10)", line_width=0.75)
        fig_sc.add_vline(x=60, line_dash="dot", line_color="rgba(232,228,220,0.15)", line_width=0.75)

        for seg in SEG_ORDER:
            sd = samp[samp["segment"] == seg]
            fig_sc.add_trace(go.Scatter(
                x=sd["demand_j"], y=sd["gap_j"],
                mode="markers", name=seg,
                marker=dict(color=SEG_COLORS[seg], size=6, opacity=0.5,
                            line=dict(width=0.3, color="rgba(232,228,220,0.15)")),
                customdata=sd[["city","property_type","current_price",
                               "recommended_price","days_on_market"]].values,
                hovertemplate=(
                    "<b>%{customdata[0]}</b>, %{customdata[1]}<br>"
                    "Current rent: $%{customdata[2]:,}<br>"
                    "Model estimate: $%{customdata[3]:,}<br>"
                    "Days on market: %{customdata[4]}<extra></extra>"
                ),
            ))

        fig_sc.update_layout(**BASE, height=380,
            yaxis=dict(**ax("Price deviation from model estimate (%)"), range=[-55, 55]),
            xaxis=dict(**ax("Neighborhood demand index"), range=[0, 100]),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

        st.markdown(f"""<div class="fig-caption">
          <b>Figure 2.</b> Price deviation vs. demand index for 400 sampled listings.
          Dashed lines mark the five-percent thresholds. Currently {hd_op:,} listings
          occupy the high-demand overpriced quadrant and {ld_up:,} the low-demand
          underpriced quadrant.
        </div>""", unsafe_allow_html=True)

    with col_city:
        city_s = df.groupby("city").agg(
            pct_over  = ("price_gap_pct", lambda x: (x < -5).mean() * 100),
            pct_under = ("price_gap_pct", lambda x: (x > 5).mean()  * 100),
        ).sort_values("pct_over", ascending=True)

        worst = city_s["pct_over"].idxmax()  if not city_s.empty else ""
        best  = city_s["pct_under"].idxmax() if not city_s.empty else ""

        fig_city = go.Figure()
        fig_city.add_trace(go.Bar(y=city_s.index, x=city_s["pct_over"],
            name="Above threshold", orientation="h", marker_color="#c97a6a"))
        fig_city.add_trace(go.Bar(y=city_s.index, x=city_s["pct_under"],
            name="Below threshold", orientation="h", marker_color="#6aacad"))
        fig_city.update_layout(**BASE, height=300, barmode="group",
            xaxis=dict(**ax("Share of listings (%)")),
            yaxis=dict(showgrid=False,
                       tickfont=dict(size=10, color="#6b6660", family="IBM Plex Mono, monospace"),
                       linecolor="rgba(232,228,220,0.12)", linewidth=1, showline=True),
        )
        st.plotly_chart(fig_city, use_container_width=True)

        st.markdown(f"""<div class="fig-caption">
          <b>Figure 3.</b> Mispricing share by metropolitan market.
          {worst} exhibits the highest above-threshold rate;
          {best} the highest below-threshold rate.
        </div>""", unsafe_allow_html=True)

    # ── ROI explorer ───────────────────────────────────────────────────────
    st.markdown('<div class="sec-header">4 &nbsp;·&nbsp; Revenue impact under partial adoption</div>', unsafe_allow_html=True)

    ec1, ec2, ec3 = st.columns(3)
    with ec1: sel_city = st.selectbox("Market", ["All"] + sorted(df["city"].unique()))
    with ec2: sel_type = st.selectbox("Property type ", ["All","Studio","1BR","2BR","3BR+"])
    with ec3: adopt = st.slider("Adoption rate (%)", 10, 100, 60, 5,
                                 help="Proportion of eligible managers implementing the recommendation.")

    roi_df = df.copy()
    if sel_city != "All": roi_df = roi_df[roi_df["city"] == sel_city]
    if sel_type != "All": roi_df = roi_df[roi_df["property_type"] == sel_type]

    eligible  = roi_df[roi_df["annual_revenue_lift"] > 0]
    adopters  = eligible.sample(frac=adopt/100, random_state=99) if len(eligible) > 0 else pd.DataFrame()
    proj_lift = adopters["annual_revenue_lift"].sum() if len(adopters) > 0 else 0
    avg_lift  = adopters["annual_revenue_lift"].mean() if len(adopters) > 0 else 0
    n_adopt   = len(adopters)
    city_str  = sel_city if sel_city != "All" else "all markets"
    type_str  = sel_type if sel_type != "All" else "all property types"

    r1, r2, r3, r4 = st.columns(4)
    with r1: st.metric("Listings in scope",  f"{len(roi_df):,}")
    with r2: st.metric("Adopting listings",  f"{n_adopt:,}")
    with r3: st.metric("Projected revenue lift",
                       f"${proj_lift/1e3:.0f}K" if proj_lift < 1e6 else f"${proj_lift/1e6:.2f}M")
    with r4: st.metric("Mean lift per listing",
                       f"${avg_lift:,.0f}/yr" if avg_lift and not np.isnan(avg_lift) else "n/a")

    if len(roi_df) > 0:
        lift_seg = roi_df.groupby("segment", observed=True)["annual_revenue_lift"].sum().reindex(SEG_ORDER)
        valid    = lift_seg.dropna()

        bar_colors = [SEG_COLORS[s] if valid[s] >= 0 else "#c97a6a" for s in valid.index]
        bar_text   = [
            f"+${v/1e3:.0f}K" if v >= 0 and abs(v) < 1e6
            else f"+${v/1e6:.2f}M" if v >= 0
            else f"-${abs(v)/1e3:.0f}K" if abs(v) < 1e6
            else f"-${abs(v)/1e6:.2f}M"
            for v in valid.values
        ]

        fig_roi = go.Figure()
        fig_roi.add_trace(go.Bar(
            x=valid.index, y=valid.values / 1e3,
            marker_color=bar_colors, marker_line_width=0,
            text=bar_text, textposition="outside",
            textfont=dict(size=11, color="#a09b90", family="IBM Plex Mono, monospace"),
        ))
        fig_roi.add_hline(y=0, line_color="rgba(232,228,220,0.25)", line_width=1.0)

        y_max = max(abs(valid.values / 1e3).max() * 1.25, 50)
        fig_roi.update_layout(**BASE, height=280, showlegend=False,
            yaxis={**ax("Projected annual revenue lift (USD thousands)"),
                   "zeroline": False, "range": [-y_max, y_max]},
            xaxis=dict(showgrid=False,
                       tickfont=dict(size=11, color="#a09b90", family="IBM Plex Mono, monospace"),
                       linecolor="rgba(232,228,220,0.12)", linewidth=1, showline=True),
        )
        st.plotly_chart(fig_roi, use_container_width=True)

    lift_fmt = f"${proj_lift/1e6:.2f}M" if abs(proj_lift) >= 1e6 else f"${proj_lift/1e3:,.0f}K"
    avg_fmt  = f"${avg_lift:,.0f}"
    st.markdown(f"""<div class="fig-caption">
      <b>Figure 4.</b> Projected revenue lift by segment at {adopt}% adoption for {city_str}
      ({type_str}). Of {len(roi_df):,} listings, {n_adopt:,} would reprice, generating an
      estimated {lift_fmt} incremental annually (mean {avg_fmt} per listing).
    </div>""", unsafe_allow_html=True)

    # ── Listing-level detail ───────────────────────────────────────────────
    st.markdown('<div class="sec-header">5 &nbsp;·&nbsp; Listing-level pricing detail</div>', unsafe_allow_html=True)

    top_thresh = df["annual_revenue_lift"].quantile(0.9)

    sort_col = st.selectbox("Sort by", [
        "annual_revenue_lift","price_gap_pct","demand_score","days_on_market","current_price"
    ], format_func=lambda x: {
        "annual_revenue_lift": "Projected revenue lift (descending)",
        "price_gap_pct":       "Price deviation from model estimate (%)",
        "demand_score":        "Neighborhood demand index",
        "days_on_market":      "Days on market",
        "current_price":       "Current listed rent",
    }[x])

    disp = df.sort_values(sort_col, ascending=False).head(200)[[
        "city","property_type","segment","sqft",
        "current_price","recommended_price","price_gap_pct",
        "demand_score","occupancy_rate","days_on_market","annual_revenue_lift",
    ]].rename(columns={
        "city":               "Market",
        "property_type":      "Type",
        "segment":            "Segment",
        "sqft":               "Sq. ft.",
        "current_price":      "Current rent ($)",
        "recommended_price":  "Model estimate ($)",
        "price_gap_pct":      "Deviation (%)",
        "demand_score":       "Demand index",
        "occupancy_rate":     "Occupancy",
        "days_on_market":     "DOM",
        "annual_revenue_lift":"Annual lift ($)",
    })
    disp["Occupancy"] = (disp["Occupancy"] * 100).round(1).astype(str) + "%"

    st.dataframe(disp, use_container_width=True, height=340, hide_index=True,
        column_config={
            "Current rent ($)":   st.column_config.NumberColumn(format="$%d"),
            "Model estimate ($)": st.column_config.NumberColumn(format="$%d"),
            "Deviation (%)":      st.column_config.NumberColumn(format="%.1f%%"),
            "Annual lift ($)":    st.column_config.NumberColumn(format="$%d"),
            "Demand index":       st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
        },
    )

    st.markdown(f"""<div class="fig-caption">
      <b>Table 2.</b> Listing-level detail, top 200 records by selected criterion
      ({min(200, len(df))} of {len(df):,} listings shown). DOM = days on market.
      The 90th percentile of projected annual lift in the current sample is ${top_thresh:,.0f}.
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div class="paper-footer">
      Model: GradientBoostingRegressor (300 estimators, max depth 4, learning rate 0.06, subsample 0.80).
      Features: square footage, neighborhood demand index, distance to city center, market, property type.
      Evaluation: 5-fold CV. CV R\u00b2\u202f=\u202f{ma['r2_mean']:.3f} (SD {ma['r2_std']:.3f}).
      CV MAE\u202f=\u202f${ma['mae_mean']:.0f}/month (SD ${ma['mae_std']:.0f}).
      Ridge baseline CV R\u00b2\u202f=\u202f{ma['baseline_r2_mean']:.3f}.
      Data: synthetic, N\u202f=\u202f2,000 listings, five U.S. markets.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 2: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════
with tab_model:

    st.markdown('<div class="sec-header">6 &nbsp;·&nbsp; Model specification and evaluation</div>', unsafe_allow_html=True)

    st.markdown(f"""<div class="abstract-text" style="margin-bottom:1.5rem;color:#a09b90;">
      A Gradient Boosting Regressor estimated on five observable listing-level features.
      Comparable transaction price is excluded from the feature set: including an oracle
      comparable would inflate apparent performance while offering little operational insight,
      since the goal is to estimate market-clearing rent from characteristics observable at
      the time of listing. All metrics below reflect out-of-fold cross-validated performance.
      The GBM is compared against a regularized linear baseline (Ridge, alpha\u202f=\u202f10.0)
      to quantify the contribution of non-linear feature interactions.
    </div>""", unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f"""<div class="model-kpi">
          <div class="model-kpi-label">CV R&sup2;</div>
          <div class="model-kpi-value model-kpi-good">{ma['r2_mean']:.3f}</div>
          <div class="model-kpi-sub">SD {ma['r2_std']:.4f}</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="model-kpi">
          <div class="model-kpi-label">CV MAE</div>
          <div class="model-kpi-value">${ma['mae_mean']:.0f}</div>
          <div class="model-kpi-sub">SD ${ma['mae_std']:.0f}/mo</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="model-kpi">
          <div class="model-kpi-label">CV RMSE</div>
          <div class="model-kpi-value">${ma['rmse_mean']:.0f}</div>
          <div class="model-kpi-sub">SD ${ma['rmse_std']:.0f}/mo</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="model-kpi">
          <div class="model-kpi-label">Ridge baseline R&sup2;</div>
          <div class="model-kpi-value model-kpi-mid">{ma['baseline_r2_mean']:.3f}</div>
          <div class="model-kpi-sub">GBM +{(ma['r2_mean']-ma['baseline_r2_mean'])*100:.1f} pp</div>
        </div>""", unsafe_allow_html=True)
    with m5:
        mae_pct = ma['mae_mean'] / df_full['recommended_price'].mean() * 100
        st.markdown(f"""<div class="model-kpi">
          <div class="model-kpi-label">MAPE</div>
          <div class="model-kpi-value model-kpi-good">{mae_pct:.1f}%</div>
          <div class="model-kpi-sub">mean rent ${df_full['recommended_price'].mean():,.0f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Feature importance + CV stability ─────────────────────────────────
    st.markdown('<div class="sec-header">7 &nbsp;·&nbsp; Feature importance and cross-validation stability</div>', unsafe_allow_html=True)

    fi_col, cv_col = st.columns([3, 2])
    with fi_col:
        imp    = ma["imp_df"]
        colors = ["rgba(200,185,122,0.4)" if i < len(imp)-1 else "#c8b97a" for i in range(len(imp))]
        imp_labels = [f"{v*100:.1f}%" for v in imp["Importance"]]
        fig_imp = go.Figure(go.Bar(
            x=imp["Importance"], y=imp["Feature"], orientation="h",
            marker_color=colors,
            text=imp_labels, textposition="outside",
            textfont=dict(size=10, color="#a09b90", family="IBM Plex Mono, monospace"),
            cliponaxis=False,
        ))
        fig_imp.update_layout(
            **{**BASE, "margin": dict(l=8, r=80, t=16, b=40)},
            height=320,
            xaxis=dict(**ax("Relative importance", grid=False),
                       tickformat=".0%",
                       range=[0, imp["Importance"].max() * 1.35]),
            yaxis=dict(showgrid=False,
                       tickfont=dict(size=10, color="#a09b90", family="IBM Plex Mono, monospace"),
                       linecolor="rgba(232,228,220,0.12)", linewidth=1, showline=True),
            showlegend=False,
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown(f"""<div class="fig-caption">
          <b>Figure 5.</b> Relative feature importance from the fitted GBM (mean reduction in
          squared error per feature). Square footage accounts for {imp.iloc[-1]['Importance']*100:.1f}%
          of predictive variance; metropolitan market {imp.iloc[-2]['Importance']*100:.1f}%.
          The relatively low importance of the demand index ({imp.iloc[0]['Importance']*100:.1f}%)
          reflects the synthetic data generating process; in production data with higher-resolution
          demand signals this feature would carry greater weight.
        </div>""", unsafe_allow_html=True)

    with cv_col:
        cv_df = ma["cv_df"]
        fig_cv = go.Figure(go.Bar(
            x=cv_df["Fold"], y=cv_df["R\u00b2"],
            marker_color="#c8b97a",
            text=[f"{v:.4f}" for v in cv_df["R\u00b2"]],
            textposition="inside",
            textfont=dict(size=10, color="#111110", family="IBM Plex Mono, monospace"),
        ))
        fig_cv.add_hline(y=ma["r2_mean"], line_dash="dash",
                         line_color="rgba(200,185,122,0.5)", line_width=1.2)
        fig_cv.add_trace(go.Scatter(
            x=["Fold 1"], y=[ma["r2_mean"] + 0.0007],
            mode="text",
            text=[f"Mean\u202f=\u202f{ma['r2_mean']:.4f}"],
            textposition="top right",
            textfont=dict(size=10, color="#c8b97a", family="IBM Plex Mono, monospace"),
            showlegend=False,
        ))
        r2_vals = cv_df["R\u00b2"].values
        fig_cv.update_layout(**BASE, height=300,
            yaxis=dict(**ax("R\u00b2"), range=[min(r2_vals)-0.006, max(r2_vals)+0.010]),
            xaxis=dict(showgrid=False,
                       tickfont=dict(size=10, color="#6b6660", family="IBM Plex Mono, monospace"),
                       linecolor="rgba(232,228,220,0.12)", linewidth=1, showline=True),
            showlegend=False,
        )
        st.plotly_chart(fig_cv, use_container_width=True)

        st.markdown(f"""<div class="fig-caption">
          <b>Figure 6.</b> Cross-validated R\u00b2 per fold.
          Low variance (SD\u202f=\u202f{ma['r2_std']:.4f}) indicates stable generalization
          across data partitions.
        </div>""", unsafe_allow_html=True)

    # ── Predicted vs actual + residual distribution ────────────────────────
    st.markdown('<div class="sec-header">8 &nbsp;·&nbsp; Prediction accuracy and residual diagnostics</div>', unsafe_allow_html=True)

    pa_col, res_col = st.columns(2)
    with pa_col:
        df_pred   = ma["df_pred"]
        samp_pred = df_pred.sample(min(600, len(df_pred)), random_state=42)
        p_min = min(samp_pred["recommended_price"].min(), samp_pred["predicted_price"].min()) * 0.95
        p_max = max(samp_pred["recommended_price"].max(), samp_pred["predicted_price"].max()) * 1.05

        fig_pa = go.Figure()
        fig_pa.add_trace(go.Scatter(
            x=[p_min, p_max], y=[p_min, p_max],
            mode="lines", line=dict(color="rgba(232,228,220,0.15)", width=1.2, dash="dash"),
            name="45° line", showlegend=True,
        ))
        for seg in SEG_ORDER:
            sd = samp_pred[samp_pred["segment"] == seg]
            if len(sd) == 0: continue
            fig_pa.add_trace(go.Scatter(
                x=sd["recommended_price"], y=sd["predicted_price"],
                mode="markers", name=seg,
                marker=dict(color=SEG_COLORS[seg], size=5, opacity=0.45,
                            line=dict(width=0.3, color="rgba(232,228,220,0.1)")),
                hovertemplate="Actual: $%{x:,}<br>Predicted: $%{y:,}<extra></extra>",
            ))
        fig_pa.update_layout(**BASE, height=360,
            xaxis=dict(**ax("Actual rent (USD)")),
            yaxis=dict(**ax("Predicted rent (USD)")),
            annotations=[dict(
                x=0.05, y=0.95, xref="paper", yref="paper",
                text=f"R\u00b2 = {ma['insample_r2']:.4f}   MAE = ${ma['insample_mae']:.0f}",
                showarrow=False,
                font=dict(size=10, color="#a09b90", family="IBM Plex Mono, monospace"),
                bgcolor="rgba(25,25,23,0.9)",
                bordercolor="rgba(232,228,220,0.12)", borderwidth=1,
            )],
        )
        st.plotly_chart(fig_pa, use_container_width=True)

        st.markdown(f"""<div class="fig-caption">
          <b>Figure 7.</b> Predicted vs. actual rent (N\u202f=\u202f600 subsample).
          In-sample R\u00b2\u202f=\u202f{ma['insample_r2']:.4f}, MAE\u202f=\u202f${ma['insample_mae']:.0f}.
          In-sample metrics slightly exceed CV metrics as expected.
        </div>""", unsafe_allow_html=True)

    with res_col:
        residuals = ma["residuals"]
        fig_res = go.Figure()
        fig_res.add_trace(go.Histogram(
            x=residuals, nbinsx=40,
            marker_color="#c8b97a", opacity=0.65, name="Residuals",
        ))
        fig_res.add_vline(x=0, line_color="rgba(232,228,220,0.3)", line_width=1.2,
                          annotation_text="Zero",
                          annotation_position="top right",
                          annotation_font=dict(size=9, color="#6b6660",
                                               family="IBM Plex Mono, monospace"))
        fig_res.add_vline(x=residuals.mean(), line_dash="dash",
                          line_color="#c97a6a", line_width=1.2,
                          annotation_text=f"Mean = ${residuals.mean():.1f}",
                          annotation_position="top left",
                          annotation_font=dict(size=9, color="#c97a6a",
                                               family="IBM Plex Mono, monospace"))
        fig_res.update_layout(**BASE, height=360,
            xaxis=dict(**ax("Residual (actual minus predicted, USD)")),
            yaxis=dict(**ax("Frequency")),
            showlegend=False,
            annotations=[dict(
                x=0.97, y=0.95, xref="paper", yref="paper",
                text=f"Mean: ${residuals.mean():.1f}   SD: ${residuals.std():.0f}",
                showarrow=False,
                font=dict(size=10, color="#a09b90", family="IBM Plex Mono, monospace"),
                bgcolor="rgba(25,25,23,0.9)",
                bordercolor="rgba(232,228,220,0.12)", borderwidth=1,
                xanchor="right",
            )],
        )
        st.plotly_chart(fig_res, use_container_width=True)

        st.markdown(f"""<div class="fig-caption">
          <b>Figure 8.</b> Residual distribution. Approximately symmetric and centered near zero
          (mean\u202f=\u202f${residuals.mean():.1f}, SD\u202f=\u202f${residuals.std():.0f}),
          consistent with an unbiased estimator.
        </div>""", unsafe_allow_html=True)

    # ── MAE by segment + homoscedasticity ──────────────────────────────────
    st.markdown('<div class="sec-header">9 &nbsp;·&nbsp; Error decomposition by segment and price level</div>', unsafe_allow_html=True)

    seg_col, hetero_col = st.columns(2)
    with seg_col:
        seg_err = df_pred.groupby("segment", observed=True).agg(
            mae     = ("abs_error",  "mean"),
            rmse    = ("residual",   lambda x: np.sqrt((x**2).mean())),
            pct_err = ("pct_error",  "mean"),
            n       = ("abs_error",  "count"),
        ).reindex(SEG_ORDER).dropna()

        fig_seg = go.Figure(go.Bar(
            x=seg_err.index, y=seg_err["mae"],
            marker_color=[SEG_COLORS[s] for s in seg_err.index],
            text=[f"${v:.0f}" for v in seg_err["mae"]],
            textposition="outside",
            textfont=dict(size=10, color="#a09b90", family="IBM Plex Mono, monospace"),
        ))
        fig_seg.update_layout(**BASE, height=280, showlegend=False,
            yaxis=dict(**ax("Mean absolute error (USD)")),
            xaxis=dict(showgrid=False,
                       tickfont=dict(size=11, color="#a09b90", family="IBM Plex Mono, monospace"),
                       linecolor="rgba(232,228,220,0.12)", linewidth=1, showline=True),
        )
        st.plotly_chart(fig_seg, use_container_width=True)

        tbl = seg_err[["mae","rmse","pct_err","n"]].copy()
        tbl.columns = ["MAE ($)", "RMSE ($)", "MAPE (%)", "N"]
        tbl["MAE ($)"]  = tbl["MAE ($)"].round(0).astype(int)
        tbl["RMSE ($)"] = tbl["RMSE ($)"].round(0).astype(int)
        tbl["MAPE (%)"] = tbl["MAPE (%)"].round(2)
        st.dataframe(tbl, use_container_width=True)

        st.markdown("""<div class="fig-caption">
          <b>Figure 9 and Table 3.</b> MAE by segment. Absolute error scales with rent level;
          MAPE is more stable across segments and is the primary comparability metric.
        </div>""", unsafe_allow_html=True)

    with hetero_col:
        samp2   = df_pred.sample(min(500, len(df_pred)), random_state=55)
        fig_het = go.Figure()
        fig_het.add_trace(go.Scatter(
            x=samp2["predicted_price"], y=samp2["residual"],
            mode="markers",
            marker=dict(color="#c8b97a", size=5, opacity=0.35,
                        line=dict(width=0.3, color="rgba(232,228,220,0.1)")),
            hovertemplate="Predicted: $%{x:,}<br>Residual: $%{y:,}<extra></extra>",
        ))
        fig_het.add_hline(y=0, line_color="rgba(232,228,220,0.15)", line_width=1.0)
        fig_het.update_layout(**BASE, height=280,
            xaxis=dict(**ax("Predicted rent (USD)")),
            yaxis=dict(**ax("Residual (USD)")),
            showlegend=False,
        )
        st.plotly_chart(fig_het, use_container_width=True)

        st.markdown("""<div class="fig-caption">
          <b>Figure 10.</b> Residuals vs. predicted values. Absence of a funnel pattern
          indicates approximately constant error variance across the rent distribution.
        </div>""", unsafe_allow_html=True)

    # ── Notes ────────────────────────────────────────────────────────────
    st.markdown('<div class="sec-header">10 &nbsp;·&nbsp; Limitations and production considerations</div>', unsafe_allow_html=True)

    n1, n2, n3 = st.columns(3)
    with n1:
        st.markdown('<div class="note-head">Model strengths</div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="note-body">
          The gradient boosting specification captures non-linear interactions among city,
          property type, and demand index that are inaccessible to linear models, yielding
          a {(ma['r2_mean']-ma['baseline_r2_mean'])*100:.1f} pp improvement in CV R\u00b2
          over the Ridge baseline. Residuals are well-centered
          (mean\u202f=\u202f${abs(ma['residuals'].mean()):.0f}) and homoscedastic.
        </div>""", unsafe_allow_html=True)
    with n2:
        st.markdown('<div class="note-head">Known limitations</div>', unsafe_allow_html=True)
        st.markdown("""<div class="note-body">
          Estimated on synthetic data; out-of-sample performance on real transaction data
          depends on comparable quality and coverage. Seasonality, macroeconomic cycles,
          and unit-level amenities are not represented. The demand index is treated as a
          static cross-sectional input rather than a rolling market-specific signal.
        </div>""", unsafe_allow_html=True)
    with n3:
        st.markdown('<div class="note-head">Production implementation</div>', unsafe_allow_html=True)
        st.markdown("""<div class="note-body">
          A production deployment would retrain on a rolling transaction window with
          automated monitoring for feature distribution shift. Point estimates would be
          accompanied by prediction intervals. A staged rollout design would enable
          causal estimation of occupancy and revenue effects via randomized experiment.
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div class="paper-footer">
      Model: GradientBoostingRegressor (n_estimators\u202f=\u202f300, max_depth\u202f=\u202f4,
      learning_rate\u202f=\u202f0.06, subsample\u202f=\u202f0.80, min_samples_leaf\u202f=\u202f15).
      Baseline: Ridge (alpha\u202f=\u202f10.0) with StandardScaler.
      Evaluation: stratified 5-fold CV (random_state\u202f=\u202f42).
      CV R\u00b2\u202f=\u202f{ma['r2_mean']:.4f} (SD {ma['r2_std']:.4f}),
      CV MAE\u202f=\u202f${ma['mae_mean']:.0f}/month (SD ${ma['mae_std']:.0f}),
      CV RMSE\u202f=\u202f${ma['rmse_mean']:.0f}/month.
      Data: synthetic, N\u202f=\u202f2,000 listings, five U.S. markets.
    </div>""", unsafe_allow_html=True)
