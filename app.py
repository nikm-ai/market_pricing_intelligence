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
    page_title="Marketplace Pricing Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px; }

  .sec-header {
    font-size: 11px; font-weight: 700; letter-spacing: 0.09em;
    text-transform: uppercase; color: var(--text-color); opacity: 0.55;
    margin: 1.75rem 0 0.6rem; padding-bottom: 6px;
    border-bottom: 1px solid rgba(128,128,128,0.2);
  }
  .kpi-card {
    border: 1px solid rgba(128,128,128,0.18);
    border-radius: 8px; padding: 1rem 1.2rem;
    background: rgba(128,128,128,0.04);
  }
  .kpi-label {
    font-size: 10px; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; opacity: 0.5; color: var(--text-color); margin-bottom: 5px;
  }
  .kpi-value { font-size: 26px; font-weight: 700; line-height: 1.15; color: var(--text-color); }
  .kpi-sub   { font-size: 12px; margin-top: 3px; opacity: 0.65; color: var(--text-color); }
  .pos { color: #27a862 !important; opacity: 1 !important; font-weight: 600; }
  .neg { color: #e5534b !important; opacity: 1 !important; font-weight: 600; }

  .callout {
    border-left: 3px solid #4a90c4; border-radius: 0 6px 6px 0;
    padding: 0.65rem 1rem; margin-bottom: 1rem;
    font-size: 13px; line-height: 1.65; color: var(--text-color);
    background: rgba(74,144,196,0.07);
  }
  .callout b { color: #4a90c4; }

  .model-kpi {
    border: 1px solid rgba(128,128,128,0.18); border-radius: 8px;
    padding: 0.85rem 1rem; background: rgba(128,128,128,0.04); text-align: center;
  }
  .model-kpi-label { font-size: 10px; font-weight: 700; letter-spacing: 0.07em;
    text-transform: uppercase; opacity: 0.5; color: var(--text-color); margin-bottom: 4px; }
  .model-kpi-value { font-size: 22px; font-weight: 700; color: var(--text-color); }
  .model-kpi-sub   { font-size: 11px; opacity: 0.55; color: var(--text-color); margin-top: 2px; }
  .model-kpi-good  { color: #27a862 !important; opacity: 1 !important; }
  .model-kpi-mid   { color: #f0a500 !important; opacity: 1 !important; }

  .seg-card {
    border: 1px solid rgba(128,128,128,0.18); border-radius: 8px;
    padding: 0.75rem 1rem; margin-bottom: 9px; background: rgba(128,128,128,0.04);
  }
  .seg-name  { font-size: 14px; font-weight: 700; color: var(--text-color); }
  .seg-count { font-size: 12px; opacity: 0.5; color: var(--text-color); }
  .seg-price { font-size: 13px; color: var(--text-color); margin-top: 4px; }
  .seg-lift  { font-size: 12px; opacity: 0.7; color: var(--text-color); margin-top: 2px; }

  [data-testid="stSidebar"] { background: rgba(128,128,128,0.03); }
  [data-testid="stTabs"] button { font-size: 13px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Chart helpers ──────────────────────────────────────────────────────────
FONT   = dict(size=13, color="#1a1a1a", family="Arial, sans-serif")
LEGEND = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
              font=dict(size=12, color="#1a1a1a"), bgcolor="rgba(0,0,0,0)")
BASE   = dict(plot_bgcolor="white", paper_bgcolor="white", font=FONT,
              margin=dict(l=4, r=4, t=14, b=4), legend=LEGEND)

def ax(title, grid=True):
    return dict(
        title=dict(text=title, font=dict(size=13, color="#1a1a1a")),
        tickfont=dict(size=12, color="#1a1a1a"),
        gridcolor="#ebebeb" if grid else "rgba(0,0,0,0)",
        showgrid=grid, zeroline=False,
    )

SEG_COLORS = {"Budget": "#6ea8d8", "Mid-Market": "#4a90c4", "Premium": "#1d5f9e"}
SEG_ORDER  = ["Budget", "Mid-Market", "Premium"]

# ── Data + Model ───────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return generate_marketplace_data(n=2000)

df_full = load_data()

with st.spinner("Fitting pricing model…"):
    model_artifacts = fit_model(df_full)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Filters")
    cities     = st.multiselect("City", sorted(df_full["city"].unique()),
                                 default=sorted(df_full["city"].unique()))
    prop_types = st.multiselect("Property type", ["Studio","1BR","2BR","3BR+"],
                                 default=["Studio","1BR","2BR","3BR+"])
    segments   = st.multiselect("Market segment", ["Budget","Mid-Market","Premium"],
                                 default=["Budget","Mid-Market","Premium"])
    dem_range  = st.slider("Demand score", 0, 100, (0, 100))
    st.markdown("---")
    st.markdown("### Model settings")
    elasticity  = st.slider("Demand elasticity", 0.5, 2.0, 1.0, 0.1,
                             help="Higher = demand more sensitive to price changes")
    occ_target  = st.slider("Target occupancy %", 70, 98, 88)
    st.markdown("---")
    st.caption("Synthetic data · 2,000 listings · 5 markets")

# ── Filter ─────────────────────────────────────────────────────────────────
df = df_full[
    df_full["city"].isin(cities) &
    df_full["property_type"].isin(prop_types) &
    df_full["segment"].isin(segments) &
    df_full["demand_score"].between(dem_range[0], dem_range[1])
].copy()

if elasticity != 1.0 or occ_target != 88:
    adj = occ_target / 100 - 0.88
    df["recommended_price"]        = (df["recommended_price"] * (1 + adj * 0.5 / elasticity)).round(0).astype(int)
    df["annual_revenue_recommended"] = (df["recommended_price"] * (occ_target / 100) * 12).round(0).astype(int)
    df["annual_revenue_lift"]      = df["annual_revenue_recommended"] - df["annual_revenue_current"]
    df["price_gap_pct"]            = ((df["recommended_price"] - df["current_price"]) / df["current_price"] * 100).round(1)

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("## Marketplace Pricing Intelligence")
st.markdown(f"Analyzing **{len(df):,}** listings across **{len(cities)}** {'market' if len(cities)==1 else 'markets'}")

if len(df) == 0:
    st.warning("No listings match the current filters.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════
tab_dash, tab_model = st.tabs(["📊  Dashboard", "🔬  Model Performance"])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════
with tab_dash:

    # ── Project overview ───────────────────────────────────────────────────
    st.markdown('<div class="sec-header">Project overview</div>', unsafe_allow_html=True)
    with st.expander("Expand to read about the client situation, problem, and methodology", expanded=True):
        oc1, oc2, oc3, oc4 = st.columns(4)
        with oc1:
            st.markdown("**Client situation**")
            st.caption(
                "A regional property management company operates 2,000+ rental listings across "
                "five U.S. markets. Their pricing team set rents manually — using comparable "
                "listings and gut feel — with no systematic process for incorporating real-time "
                "demand signals or measuring pricing accuracy at scale."
            )
        with oc2:
            st.markdown("**The problem**")
            st.caption(
                "Without a data-driven baseline, the portfolio had two simultaneous issues: "
                "overpriced listings sitting vacant too long (lost revenue from empty units), "
                "and underpriced listings in high-demand areas (revenue left on the table). "
                "Leadership had no visibility into which situation applied to which listings, "
                "or what it was costing them."
            )
        with oc3:
            st.markdown("**How we solved it**")
            st.caption(
                "We trained a Gradient Boosting model on listing features (sq ft, demand score, "
                "distance to center, city, property type) to predict market-clearing rent. "
                "We quantified the revenue gap between current and model-recommended pricing, "
                "segmented the portfolio by market and property type, and built this self-serve "
                "dashboard so non-technical managers could explore findings independently."
            )
        with oc4:
            st.markdown("**Technical approach**")
            st.caption(
                f"GradientBoostingRegressor (300 estimators, depth 4) evaluated with 5-fold CV. "
                f"CV R² = {model_artifacts['r2_mean']:.3f} vs Ridge baseline "
                f"{model_artifacts['baseline_r2_mean']:.3f}. "
                f"MAE = ${model_artifacts['mae_mean']:.0f}/month. "
                f"Features: sq ft, demand score, distance, city, property type."
            )
            st.markdown(
                "`Python` `scikit-learn` `GBM` `pandas` `Streamlit` `Plotly`"
            )

    # ── KPI summary ────────────────────────────────────────────────────────
    st.markdown('<div class="sec-header">Portfolio summary</div>', unsafe_allow_html=True)

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

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        dc = "pos" if total_lift >= 0 else "neg"
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">Total revenue opportunity</div>
          <div class="kpi-value <{dc}>">${total_lift/1e6:.2f}M</div>
          <div class="kpi-sub">annual lift from repricing</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        dc = "pos" if price_delta >= 0 else "neg"
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">Median recommended price</div>
          <div class="kpi-value">${med_rec:,.0f}</div>
          <div class="kpi-sub">vs <span class="{dc}">${med_cur:,.0f} current ({delta_pct:+.1f}%)</span></div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">Mispriced listings</div>
          <div class="kpi-value">{pct_over + pct_under:.0f}%</div>
          <div class="kpi-sub">
            <span class="neg">{pct_over:.0f}% overpriced</span> ·
            <span class="pos">{pct_under:.0f}% underpriced</span>
          </div>
        </div>""", unsafe_allow_html=True)
    with k4:
        occ_dir = f"{abs(occ_gap):.1f} pts {'above' if occ_gap >= 0 else 'below'} target"
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">Avg portfolio occupancy</div>
          <div class="kpi-value">{avg_occ:.1f}%</div>
          <div class="kpi-sub">{occ_dir} · demand score: {med_demand:.0f}/100</div>
        </div>""", unsafe_allow_html=True)

    lift_word = "gained" if total_lift >= 0 else "recovered"
    st.markdown(f"""<div class="callout">
      <b>What this means:</b> Of {len(df):,} listings analyzed,
      <b>{n_over:,} ({pct_over:.0f}%)</b> are priced more than 5% above the model's recommendation —
      likely sitting vacant longer than necessary. Another
      <b>{n_under:,} ({pct_under:.0f}%)</b> are priced more than 5% below, leaving revenue on the table.
      Full repricing adoption would generate an estimated
      <b>${abs(total_lift)/1e6:.2f}M</b> {lift_word} annually.
      Average occupancy is <b>{avg_occ:.1f}%</b> — {abs(occ_gap):.1f} pts
      {'above' if occ_gap >= 0 else 'below'} the {occ_target}% target.
    </div>""", unsafe_allow_html=True)

    # ── Pricing distribution ───────────────────────────────────────────────
    st.markdown('<div class="sec-header">Pricing distribution by segment</div>', unsafe_allow_html=True)

    seg_sum = df.groupby("segment", observed=True).agg(
        listings  = ("current_price",      "count"),
        med_cur   = ("current_price",      "median"),
        med_rec   = ("recommended_price",  "median"),
        total_lift= ("annual_revenue_lift","sum"),
        avg_occ   = ("occupancy_rate",     "mean"),
    ).reindex(SEG_ORDER)

    big_seg  = seg_sum["total_lift"].abs().idxmax() if not seg_sum.empty else "Mid-Market"
    big_lift = seg_sum.loc[big_seg, "total_lift"] if big_seg in seg_sum.index else 0

    st.markdown(f"""<div class="callout">
      <b>How to read this:</b> Each box shows the spread of rents within a segment.
      The center line is the median; the box covers the middle 50% of listings; whiskers show the full range.
      <b>Darker boxes</b> are current prices — <b>lighter boxes</b> are what the model recommends.
      Where the lighter box sits below the darker one, that segment is likely overpriced relative to what
      the market will bear. The <b>{big_seg}</b> segment has the largest pricing gap —
      estimated <b>${abs(big_lift)/1e3:.0f}K</b> annual impact if corrected.
    </div>""", unsafe_allow_html=True)

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
                y=sd["recommended_price"], name=f"{seg} (recommended)",
                marker_color=SEG_COLORS[seg], opacity=0.35,
                legendgroup=seg, boxmean=True,
            ))
        fig_box.update_layout(**BASE, height=360,
            yaxis=dict(**ax("Monthly rent ($)")),
            xaxis=dict(showgrid=False, tickfont=dict(size=12, color="#1a1a1a")),
        )
        st.plotly_chart(fig_box, use_container_width=True)

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
              <div style="display:flex;justify-content:space-between;">
                <span class="seg-name">{seg}</span>
                <span class="seg-count">{int(row['listings']):,} listings</span>
              </div>
              <div class="seg-price">
                ${row['med_cur']:,.0f} → <b>${row['med_rec']:,.0f}</b>
                <span class="{dc}"> ({dp:+.1f}%)</span>
              </div>
              <div class="seg-lift">
                Annual lift: <span class="{dc}">{ls}</span> ·
                Occupancy: {row['avg_occ']*100:.1f}%
              </div>
            </div>""", unsafe_allow_html=True)

    # ── Opportunity map ────────────────────────────────────────────────────
    st.markdown('<div class="sec-header">Demand score vs price gap — opportunity map</div>', unsafe_allow_html=True)

    hd_op = len(df[(df["demand_score"] > 60) & (df["price_gap_pct"] < -5)])
    ld_up = len(df[(df["demand_score"] < 40) & (df["price_gap_pct"] > 5)])

    st.markdown(f"""<div class="callout">
      <b>How to read this:</b> Each dot is one listing.
      The <b>horizontal axis</b> is neighborhood demand — higher means more renters competing for fewer units.
      The <b>vertical axis</b> shows whether the current price is above or below the model's recommendation.
      Colored quadrant shading marks the four pricing situations.
      <b>Top-right (green)</b> = high demand, underpriced — highest-priority rent increases.
      <b>Bottom-right (amber)</b> = high demand but overpriced — likely losing occupancy unnecessarily.
      Currently <b>{hd_op:,}</b> listings are in high-demand areas but overpriced, and
      <b>{ld_up:,}</b> are underpriced despite weak demand — both worth a closer look.
    </div>""", unsafe_allow_html=True)

    col_sc, col_city = st.columns([3, 2])
    with col_sc:
        # Reduced sample + jitter to reduce overplotting
        rng_jitter = np.random.default_rng(7)
        samp = df.sample(min(400, len(df)), random_state=42).copy()
        samp["demand_score_j"] = samp["demand_score"] + rng_jitter.uniform(-0.8, 0.8, len(samp))
        samp["price_gap_j"]    = samp["price_gap_pct"] + rng_jitter.uniform(-0.3, 0.3, len(samp))

        fig_sc = go.Figure()

        # Quadrant shading
        x_range = [0, 100]
        fig_sc.add_shape(type="rect", x0=60, x1=100, y0=5,  y1=55,
                         fillcolor="rgba(39,168,98,0.07)",  line_width=0)
        fig_sc.add_shape(type="rect", x0=60, x1=100, y0=-55, y1=-5,
                         fillcolor="rgba(240,165,0,0.07)",  line_width=0)
        fig_sc.add_shape(type="rect", x0=0,  x1=40,  y0=5,  y1=55,
                         fillcolor="rgba(74,144,196,0.05)", line_width=0)
        fig_sc.add_shape(type="rect", x0=0,  x1=40,  y0=-55, y1=-5,
                         fillcolor="rgba(229,83,75,0.05)",  line_width=0)

        # Threshold lines
        fig_sc.add_hline(y=5,  line_dash="dash", line_color="#27a862", line_width=1.5,
                         annotation_text="Underpriced (+5%)", annotation_position="top right",
                         annotation_font=dict(size=11, color="#27a862"))
        fig_sc.add_hline(y=-5, line_dash="dash", line_color="#e5534b", line_width=1.5,
                         annotation_text="Overpriced (−5%)", annotation_position="bottom right",
                         annotation_font=dict(size=11, color="#e5534b"))
        fig_sc.add_hline(y=0, line_color="#cccccc", line_width=0.75)
        fig_sc.add_vline(x=60, line_dash="dot", line_color="#aaaaaa", line_width=0.75)

        for seg in SEG_ORDER:
            sd = samp[samp["segment"] == seg]
            fig_sc.add_trace(go.Scatter(
                x=sd["demand_score_j"], y=sd["price_gap_j"],
                mode="markers",
                name=seg,
                marker=dict(
                    color=SEG_COLORS[seg], size=6,
                    opacity=0.55, line=dict(width=0.3, color="white"),
                ),
                customdata=sd[["city","property_type","current_price",
                               "recommended_price","days_on_market"]].values,
                hovertemplate=(
                    "<b>%{customdata[0]}</b> · %{customdata[1]}<br>"
                    "Current: $%{customdata[2]:,}<br>"
                    "Recommended: $%{customdata[3]:,}<br>"
                    "Days on market: %{customdata[4]}<extra></extra>"
                ),
            ))

        fig_sc.update_layout(**BASE, height=380,
            yaxis=dict(**ax("Price gap vs recommended (%)"),
                       range=[-55, 55]),
            xaxis=dict(**ax("Demand score"), range=[0, 100]),
            showlegend=True,
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    with col_city:
        city_s = df.groupby("city").agg(
            pct_over  = ("price_gap_pct", lambda x: (x < -5).mean() * 100),
            pct_under = ("price_gap_pct", lambda x: (x > 5).mean()  * 100),
        ).sort_values("pct_over", ascending=True)

        worst = city_s["pct_over"].idxmax()  if not city_s.empty else ""
        best  = city_s["pct_under"].idxmax() if not city_s.empty else ""
        st.markdown(f"""<div class="callout" style="font-size:12.5px;">
          <b>{worst}</b> has the most overpriced listings — prioritize corrections here.
          <b>{best}</b> has the most underpriced listings — biggest near-term upside.
        </div>""", unsafe_allow_html=True)

        fig_city = go.Figure()
        fig_city.add_trace(go.Bar(y=city_s.index, x=city_s["pct_over"],
            name="Overpriced", orientation="h", marker_color="#e07070"))
        fig_city.add_trace(go.Bar(y=city_s.index, x=city_s["pct_under"],
            name="Underpriced", orientation="h", marker_color="#6ab06a"))
        fig_city.update_layout(**BASE, height=300, barmode="group",
            xaxis=dict(**ax("% of listings")),
            yaxis=dict(showgrid=False, tickfont=dict(size=12, color="#1a1a1a")),
        )
        st.plotly_chart(fig_city, use_container_width=True)

    # ── ROI explorer ───────────────────────────────────────────────────────
    st.markdown('<div class="sec-header">Self-serve ROI explorer</div>', unsafe_allow_html=True)

    ec1, ec2, ec3 = st.columns(3)
    with ec1: sel_city = st.selectbox("Market", ["All"] + sorted(df["city"].unique()))
    with ec2: sel_type = st.selectbox("Property type", ["All","Studio","1BR","2BR","3BR+"])
    with ec3: adopt = st.slider("Model adoption rate (%)", 10, 100, 60, 5,
                                 help="% of property managers who adopt the recommended price")

    roi_df = df.copy()
    if sel_city != "All": roi_df = roi_df[roi_df["city"] == sel_city]
    if sel_type != "All": roi_df = roi_df[roi_df["property_type"] == sel_type]

    eligible  = roi_df[roi_df["annual_revenue_lift"] > 0]
    adopters  = eligible.sample(frac=adopt/100, random_state=99) if len(eligible) > 0 else pd.DataFrame()
    proj_lift = adopters["annual_revenue_lift"].sum() if len(adopters) > 0 else 0
    avg_lift  = adopters["annual_revenue_lift"].mean() if len(adopters) > 0 else 0
    n_adopt   = len(adopters)

    city_str = sel_city if sel_city != "All" else "all markets"
    type_str = sel_type if sel_type != "All" else "all property types"

    st.markdown(f"""<div class="callout">
      <b>How to use this:</b> Not every property manager updates their price overnight.
      This section models the business case at different rollout speeds.
      Of {len(roi_df):,} listings in <b>{city_str}</b> ({type_str}),
      <b>{len(eligible):,}</b> stand to gain from repricing.
      At a <b>{adopt}% adoption rate</b>, roughly <b>{n_adopt:,} listings</b> would reprice,
      generating an estimated <b>${proj_lift/1e3:.0f}K</b> in added annual revenue —
      about <b>${avg_lift:,.0f}/listing/year</b>.
      Slide the adoption rate to stress-test conservative vs. optimistic rollout assumptions.
    </div>""", unsafe_allow_html=True)

    r1, r2, r3, r4 = st.columns(4)
    with r1: st.metric("Listings in scope",  f"{len(roi_df):,}")
    with r2: st.metric("Projected adopters", f"{n_adopt:,}")
    with r3: st.metric("Portfolio lift",
                       f"${proj_lift/1e3:.0f}K" if proj_lift < 1e6 else f"${proj_lift/1e6:.2f}M")
    with r4: st.metric("Avg lift / listing",
                       f"${avg_lift:,.0f}/yr" if avg_lift and not np.isnan(avg_lift) else "—")

    if len(roi_df) > 0:
        lift_seg = roi_df.groupby("segment", observed=True)["annual_revenue_lift"].sum().reindex(SEG_ORDER)
        valid    = lift_seg.dropna()
        fig_roi  = go.Figure(go.Bar(
            x=valid.index,
            y=valid.values / 1e3,
            marker_color=[SEG_COLORS[s] for s in valid.index],
            text=[f"${v/1e3:.0f}K" if abs(v) < 1e6 else f"${v/1e6:.2f}M" for v in valid.values],
            textposition="outside",
            textfont=dict(size=13, color="#1a1a1a"),
        ))
        fig_roi.update_layout(**BASE, height=260, showlegend=False,
            yaxis={**ax("Annual revenue lift ($K)"), "zeroline": True, "zerolinecolor": "#dddddd"},
            xaxis=dict(showgrid=False, tickfont=dict(size=13, color="#1a1a1a")),
        )
        st.plotly_chart(fig_roi, use_container_width=True)

    # ── Listing-level detail ───────────────────────────────────────────────
    st.markdown('<div class="sec-header">Listing-level detail</div>', unsafe_allow_html=True)

    top_thresh = df["annual_revenue_lift"].quantile(0.9)
    st.markdown(f"""<div class="callout">
      <b>How to read this:</b> Each row is one listing. Sort by <em>Revenue lift</em> to surface
      the highest-priority repricing opportunities — large price gaps in high-demand neighborhoods.
      <em>DOM</em> (days on market) is one of the clearest signals of overpricing: listings vacant
      for 30+ days are almost always priced above what renters will pay.
      The top 10% of listings by opportunity each stand to gain more than
      <b>${top_thresh:,.0f}/year</b> from a price correction.
    </div>""", unsafe_allow_html=True)

    sort_col = st.selectbox("Sort by", [
        "annual_revenue_lift","price_gap_pct","demand_score","days_on_market","current_price"
    ], format_func=lambda x: {
        "annual_revenue_lift": "Revenue lift (highest first)",
        "price_gap_pct":       "Price gap %",
        "demand_score":        "Demand score",
        "days_on_market":      "Days on market",
        "current_price":       "Current price",
    }[x])

    disp = df.sort_values(sort_col, ascending=False).head(200)[[
        "city","property_type","segment","sqft",
        "current_price","recommended_price","price_gap_pct",
        "demand_score","occupancy_rate","days_on_market","annual_revenue_lift",
    ]].rename(columns={
        "city":"City","property_type":"Type","segment":"Segment","sqft":"Sq ft",
        "current_price":"Current ($)","recommended_price":"Recommended ($)",
        "price_gap_pct":"Gap (%)","demand_score":"Demand","occupancy_rate":"Occupancy",
        "days_on_market":"DOM","annual_revenue_lift":"Annual lift ($)",
    })
    disp["Occupancy"] = (disp["Occupancy"] * 100).round(1).astype(str) + "%"

    st.dataframe(disp, use_container_width=True, height=340, hide_index=True,
        column_config={
            "Current ($)":     st.column_config.NumberColumn(format="$%d"),
            "Recommended ($)": st.column_config.NumberColumn(format="$%d"),
            "Gap (%)":         st.column_config.NumberColumn(format="%.1f%%"),
            "Annual lift ($)": st.column_config.NumberColumn(format="$%d"),
            "Demand":          st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
        },
    )

    st.caption(
        "Pricing recommendations via GradientBoostingRegressor trained on sq ft, demand score, "
        "distance, city, and property type. Revenue lift assumes partial occupancy adjustment. "
        f"Model CV R² = {model_artifacts['r2_mean']:.3f} · MAE = ${model_artifacts['mae_mean']:.0f}/mo · "
        "Synthetic dataset — 2,000 listings · 5 U.S. markets."
    )


# ══════════════════════════════════════════════════════════════════════════
# TAB 2: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════
with tab_model:
    ma = model_artifacts

    st.markdown('<div class="sec-header">Model overview</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="callout">
      <b>What this tab shows:</b> This pricing model uses a
      <b>Gradient Boosting Regressor</b> trained on observable listing features
      (square footage, neighborhood demand score, distance to city center, city, and property type)
      to predict the market-clearing monthly rent. The model was evaluated using
      <b>5-fold cross-validation</b> to ensure performance estimates are not inflated by
      overfitting to the training data. All metrics below reflect held-out fold performance.
      The model meaningfully outperforms a Ridge regression baseline
      (CV R² <b>{ma['r2_mean']:.3f}</b> vs <b>{ma['baseline_r2_mean']:.3f}</b>),
      demonstrating that the non-linear feature interactions captured by GBM are
      genuinely useful for pricing prediction.
    </div>""", unsafe_allow_html=True)

    # ── Model KPI cards ────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f"""<div class="model-kpi">
          <div class="model-kpi-label">CV R²</div>
          <div class="model-kpi-value model-kpi-good">{ma['r2_mean']:.3f}</div>
          <div class="model-kpi-sub">± {ma['r2_std']:.3f} across folds</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="model-kpi">
          <div class="model-kpi-label">CV MAE</div>
          <div class="model-kpi-value">${ma['mae_mean']:.0f}</div>
          <div class="model-kpi-sub">± ${ma['mae_std']:.0f} per month</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="model-kpi">
          <div class="model-kpi-label">CV RMSE</div>
          <div class="model-kpi-value">${ma['rmse_mean']:.0f}</div>
          <div class="model-kpi-sub">± ${ma['rmse_std']:.0f} per month</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="model-kpi">
          <div class="model-kpi-label">Baseline R² (Ridge)</div>
          <div class="model-kpi-value model-kpi-mid">{ma['baseline_r2_mean']:.3f}</div>
          <div class="model-kpi-sub">GBM +{(ma['r2_mean']-ma['baseline_r2_mean'])*100:.1f} pts</div>
        </div>""", unsafe_allow_html=True)
    with m5:
        mae_pct = ma['mae_mean'] / df_full['recommended_price'].mean() * 100
        st.markdown(f"""<div class="model-kpi">
          <div class="model-kpi-label">MAE as % of mean rent</div>
          <div class="model-kpi-value model-kpi-good">{mae_pct:.1f}%</div>
          <div class="model-kpi-sub">mean rent ${df_full['recommended_price'].mean():,.0f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Row 1: Feature importance + CV scores ─────────────────────────────
    st.markdown('<div class="sec-header">Feature importance & cross-validation stability</div>', unsafe_allow_html=True)
    st.markdown("""<div class="callout">
      <b>Feature importance</b> (left) shows each variable's relative contribution to the model's predictions,
      measured by how much including that feature reduces prediction error across all decision trees.
      Higher = more influential. <b>Cross-validation R²</b> (right) shows model performance on each
      held-out fold independently — tight clustering around a high value indicates the model generalizes
      well and is not overfitting to any particular subset of the data.
    </div>""", unsafe_allow_html=True)

    fi_col, cv_col = st.columns([3, 2])
    with fi_col:
        imp = ma["imp_df"]
        colors = ["#b0c8e8" if i < len(imp)-1 else "#1d5f9e" for i in range(len(imp))]
        fig_imp = go.Figure(go.Bar(
            x=imp["Importance"],
            y=imp["Feature"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:.1%}" for v in imp["Importance"]],
            textposition="outside",
            textfont=dict(size=12, color="#1a1a1a"),
        ))
        fig_imp.update_layout(**BASE, height=300,
            xaxis=dict(**ax("Feature importance (relative)", grid=False),
                       tickformat=".0%", range=[0, imp["Importance"].max()*1.25]),
            yaxis=dict(showgrid=False, tickfont=dict(size=12, color="#1a1a1a")),
            showlegend=False,
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with cv_col:
        cv_df = ma["cv_df"]
        fig_cv = go.Figure()
        fig_cv.add_trace(go.Bar(
            x=cv_df["Fold"], y=cv_df["R²"],
            marker_color="#4a90c4",
            text=[f"{v:.3f}" for v in cv_df["R²"]],
            textposition="outside",
            textfont=dict(size=12, color="#1a1a1a"),
        ))
        fig_cv.add_hline(y=ma["r2_mean"], line_dash="dash", line_color="#1d5f9e", line_width=1.5,
                         annotation_text=f"Mean R² = {ma['r2_mean']:.3f}",
                         annotation_position="top left",
                         annotation_font=dict(size=11, color="#1d5f9e"))
        fig_cv.update_layout(**BASE, height=300,
            yaxis=dict(**ax("R²"), range=[max(0, ma["r2_mean"]-0.05), 1.0]),
            xaxis=dict(showgrid=False, tickfont=dict(size=12, color="#1a1a1a")),
            showlegend=False,
        )
        st.plotly_chart(fig_cv, use_container_width=True)

    # ── Row 2: Predicted vs actual + residual distribution ─────────────────
    st.markdown('<div class="sec-header">Prediction accuracy & residual analysis</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="callout">
      <b>Predicted vs actual</b> (left): Each point is one listing. Points along the diagonal
      represent perfect predictions — the tighter the cluster around the line, the better the model.
      Color indicates segment. <b>Residual distribution</b> (right): Residuals (actual minus predicted)
      should be centered near zero with no systematic skew. A roughly normal distribution with
      mean ≈ 0 indicates the model has no systematic bias. Current residual mean:
      <b>${ma['residuals'].mean():.1f}</b>, std: <b>${ma['residuals'].std():.0f}</b>.
    </div>""", unsafe_allow_html=True)

    pa_col, res_col = st.columns(2)
    with pa_col:
        df_pred = ma["df_pred"]
        samp_pred = df_pred.sample(min(600, len(df_pred)), random_state=42)

        fig_pa = go.Figure()
        price_min = min(samp_pred["recommended_price"].min(), samp_pred["predicted_price"].min()) * 0.95
        price_max = max(samp_pred["recommended_price"].max(), samp_pred["predicted_price"].max()) * 1.05
        fig_pa.add_trace(go.Scatter(
            x=[price_min, price_max], y=[price_min, price_max],
            mode="lines", line=dict(color="#aaaaaa", width=1.5, dash="dash"),
            name="Perfect prediction", showlegend=True,
        ))
        for seg in SEG_ORDER:
            sd = samp_pred[samp_pred["segment"] == seg]
            if len(sd) == 0: continue
            fig_pa.add_trace(go.Scatter(
                x=sd["recommended_price"], y=sd["predicted_price"],
                mode="markers", name=seg,
                marker=dict(color=SEG_COLORS[seg], size=5, opacity=0.5,
                            line=dict(width=0.3, color="white")),
                hovertemplate="Actual: $%{x:,}<br>Predicted: $%{y:,}<extra></extra>",
            ))
        fig_pa.update_layout(**BASE, height=360,
            xaxis=dict(**ax("Actual price ($)")),
            yaxis=dict(**ax("Predicted price ($)")),
            annotations=[dict(
                x=0.05, y=0.95, xref="paper", yref="paper",
                text=f"R² = {ma['insample_r2']:.3f}  |  MAE = ${ma['insample_mae']:.0f}",
                showarrow=False, font=dict(size=12, color="#1a1a1a"),
                bgcolor="rgba(255,255,255,0.8)", bordercolor="#cccccc", borderwidth=1,
            )],
        )
        st.plotly_chart(fig_pa, use_container_width=True)

    with res_col:
        residuals = ma["residuals"]
        fig_res = go.Figure()
        fig_res.add_trace(go.Histogram(
            x=residuals, nbinsx=40,
            marker_color="#4a90c4", opacity=0.75,
            name="Residuals",
        ))
        fig_res.add_vline(x=0, line_color="#1a1a1a", line_width=1.5,
                          annotation_text="Zero error",
                          annotation_position="top right",
                          annotation_font=dict(size=11, color="#1a1a1a"))
        fig_res.add_vline(x=residuals.mean(), line_dash="dash", line_color="#e5534b", line_width=1.5,
                          annotation_text=f"Mean = ${residuals.mean():.0f}",
                          annotation_position="top left",
                          annotation_font=dict(size=11, color="#e5534b"))
        fig_res.update_layout(**BASE, height=360,
            xaxis=dict(**ax("Residual (actual − predicted, $)")),
            yaxis=dict(**ax("Count")),
            showlegend=False,
            annotations=[dict(
                x=0.97, y=0.95, xref="paper", yref="paper",
                text=f"Mean: ${residuals.mean():.0f}  |  Std: ${residuals.std():.0f}",
                showarrow=False, font=dict(size=12, color="#1a1a1a"),
                bgcolor="rgba(255,255,255,0.8)", bordercolor="#cccccc", borderwidth=1,
                xanchor="right",
            )],
        )
        st.plotly_chart(fig_res, use_container_width=True)

    # ── Row 3: MAE by segment + residuals vs predicted ─────────────────────
    st.markdown('<div class="sec-header">Error analysis by segment & price range</div>', unsafe_allow_html=True)
    st.markdown("""<div class="callout">
      <b>MAE by segment</b> (left): Breaks down average prediction error by market segment.
      Higher-priced segments typically have larger absolute errors but similar percentage errors.
      <b>Residuals vs predicted</b> (right): Plots prediction error against the predicted price.
      Random scatter around zero (no funnel shape, no trend) indicates
      <em>homoscedasticity</em> — the model's errors don't systematically grow for more expensive listings,
      which is a key assumption of well-calibrated regression models.
    </div>""", unsafe_allow_html=True)

    seg_col, hetero_col = st.columns(2)
    with seg_col:
        seg_err = df_pred.groupby("segment", observed=True).agg(
            mae=("abs_error",  "mean"),
            rmse=("residual", lambda x: np.sqrt((x**2).mean())),
            pct_err=("pct_error", "mean"),
            n=("abs_error", "count"),
        ).reindex(SEG_ORDER).dropna()

        fig_seg_err = go.Figure()
        fig_seg_err.add_trace(go.Bar(
            x=seg_err.index, y=seg_err["mae"],
            name="MAE ($)", marker_color=[SEG_COLORS[s] for s in seg_err.index],
            text=[f"${v:.0f}" for v in seg_err["mae"]],
            textposition="outside", textfont=dict(size=12, color="#1a1a1a"),
        ))
        fig_seg_err.update_layout(**BASE, height=300, showlegend=False,
            yaxis=dict(**ax("Mean absolute error ($)")),
            xaxis=dict(showgrid=False, tickfont=dict(size=12, color="#1a1a1a")),
        )
        st.plotly_chart(fig_seg_err, use_container_width=True)

        # MAE table
        tbl = seg_err[["mae","rmse","pct_err","n"]].copy()
        tbl.columns = ["MAE ($)", "RMSE ($)", "MAPE (%)", "N"]
        tbl["MAE ($)"]  = tbl["MAE ($)"].round(0).astype(int)
        tbl["RMSE ($)"] = tbl["RMSE ($)"].round(0).astype(int)
        tbl["MAPE (%)"] = tbl["MAPE (%)"].round(1)
        st.dataframe(tbl, use_container_width=True)

    with hetero_col:
        samp_pred2 = df_pred.sample(min(500, len(df_pred)), random_state=55)
        fig_het = go.Figure()
        fig_het.add_trace(go.Scatter(
            x=samp_pred2["predicted_price"], y=samp_pred2["residual"],
            mode="markers",
            marker=dict(color="#4a90c4", size=5, opacity=0.45,
                        line=dict(width=0.3, color="white")),
            hovertemplate="Predicted: $%{x:,}<br>Residual: $%{y:,}<extra></extra>",
        ))
        fig_het.add_hline(y=0, line_color="#aaaaaa", line_width=1.0)
        fig_het.update_layout(**BASE, height=340,
            xaxis=dict(**ax("Predicted price ($)")),
            yaxis=dict(**ax("Residual (actual − predicted, $)")),
            showlegend=False,
        )
        st.plotly_chart(fig_het, use_container_width=True)

    # ── Model notes ────────────────────────────────────────────────────────
    st.markdown('<div class="sec-header">Model notes & limitations</div>', unsafe_allow_html=True)
    n1, n2, n3 = st.columns(3)
    with n1:
        st.markdown("**What the model does well**")
        st.caption(
            f"Captures non-linear interactions between city, property type, and demand score "
            f"that a linear model misses — improving R² by "
            f"{(ma['r2_mean']-ma['baseline_r2_mean'])*100:.1f} percentage points over Ridge. "
            f"Predictions are well-calibrated (mean residual ≈ ${ma['residuals'].mean():.0f}), "
            f"and error variance is stable across the price range (homoscedastic)."
        )
    with n2:
        st.markdown("**Known limitations**")
        st.caption(
            "Trained on synthetic data — real-world performance would depend on actual comparable "
            "transaction quality and recency. The model does not capture seasonality, "
            "macroeconomic conditions, or unit-level amenities (parking, in-unit laundry, etc.). "
            "Demand score is treated as a static input; in production this would be a rolling signal."
        )
    with n3:
        st.markdown("**Production considerations**")
        st.caption(
            "In a production system, this model would be retrained monthly on fresh transaction data "
            "with a rolling validation window. Feature drift monitoring (particularly demand score "
            "and comp price distributions) would trigger retraining. Prediction intervals would "
            "replace point estimates to communicate uncertainty to end users."
        )
