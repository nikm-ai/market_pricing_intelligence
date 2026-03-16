import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from generate_data import generate_marketplace_data

st.set_page_config(
    page_title="Marketplace Pricing Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px; }

    /* Metric cards */
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        border: 1px solid #dee2e6;
    }
    .metric-label {
        font-size: 11px; color: #495057; font-weight: 600;
        margin-bottom: 4px; letter-spacing: 0.06em; text-transform: uppercase;
    }
    .metric-value { font-size: 24px; font-weight: 700; color: #212529; line-height: 1.2; }
    .metric-delta { font-size: 12px; margin-top: 3px; color: #495057; }
    .delta-pos { color: #198754; font-weight: 600; }
    .delta-neg { color: #dc3545; font-weight: 600; }

    /* Section headers */
    .section-header {
        font-size: 13px; font-weight: 700; color: #212529;
        letter-spacing: 0.07em; text-transform: uppercase;
        margin: 1.75rem 0 0.5rem; padding-bottom: 7px;
        border-bottom: 2px solid #dee2e6;
    }

    /* Explanation callouts */
    .explanation-box {
        background: #f0f4f8;
        border-left: 3px solid #4a90c4;
        border-radius: 0 6px 6px 0;
        padding: 0.7rem 1rem;
        margin-bottom: 1rem;
        font-size: 13px; color: #212529; line-height: 1.6;
    }
    .explanation-box strong { color: #1d5f9e; }

    /* Project overview card */
    .overview-card {
        background: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 1.5rem 1.75rem;
        margin-bottom: 1.5rem;
    }
    .overview-title {
        font-size: 15px; font-weight: 700; color: #212529;
        margin-bottom: 1rem; letter-spacing: 0.02em;
    }
    .overview-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 1rem;
    }
    .overview-block { }
    .overview-block-label {
        font-size: 10px; font-weight: 700; color: #6c757d;
        letter-spacing: 0.08em; text-transform: uppercase;
        margin-bottom: 5px;
    }
    .overview-block-text {
        font-size: 13px; color: #343a40; line-height: 1.55;
    }
    .tag-row { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 0.75rem; }
    .tag {
        display: inline-block; padding: 2px 9px;
        background: #e9f0f8; color: #1d5f9e;
        border-radius: 10px; font-size: 11px; font-weight: 600;
    }

    /* Segment detail cards */
    .seg-card {
        background: #f8f9fa; border-radius: 8px;
        padding: 0.75rem 1rem; margin-bottom: 10px;
        border: 1px solid #dee2e6;
    }

    div[data-testid="stSidebar"] { background: #f8f9fa; }
    .stSelectbox label, .stMultiSelect label, .stSlider label {
        font-size: 13px; font-weight: 600; color: #212529;
    }
</style>
""", unsafe_allow_html=True)

# ── Chart style helpers ────────────────────────────────────────────────────
FONT = dict(size=13, color="#212529", family="Arial, sans-serif")

def axis(title_text, show_grid=True):
    return dict(
        title=dict(text=title_text, font=dict(size=13, color="#212529")),
        tickfont=dict(size=12, color="#212529"),
        gridcolor="#e9ecef" if show_grid else "rgba(0,0,0,0)",
        showgrid=show_grid,
    )

LAYOUT_BASE = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=FONT,
    margin=dict(l=4, r=4, t=12, b=4),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02,
        xanchor="left", x=0,
        font=dict(size=12, color="#212529"),
        bgcolor="rgba(0,0,0,0)",
    ),
)

# ── Data ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return generate_marketplace_data(n=2000)

df_full = load_data()

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Filters")
    cities = st.multiselect(
        "City", options=sorted(df_full["city"].unique()),
        default=sorted(df_full["city"].unique()),
    )
    prop_types = st.multiselect(
        "Property type", options=["Studio", "1BR", "2BR", "3BR+"],
        default=["Studio", "1BR", "2BR", "3BR+"],
    )
    segments = st.multiselect(
        "Market segment", options=["Budget", "Mid-Market", "Premium"],
        default=["Budget", "Mid-Market", "Premium"],
    )
    demand_range = st.slider("Demand score range", 0, 100, (0, 100), step=1)

    st.markdown("---")
    st.markdown("### Model settings")
    price_sensitivity = st.slider(
        "Demand elasticity", 0.5, 2.0, 1.0, step=0.1,
        help="How strongly demand responds to price changes. Higher = more elastic market.",
    )
    occupancy_target = st.slider("Target occupancy %", 70, 98, 88, step=1)
    st.markdown("---")
    st.caption("Synthetic data · 2,000 listings · 5 markets")

# ── Filter & recompute ─────────────────────────────────────────────────────
df = df_full[
    df_full["city"].isin(cities) &
    df_full["property_type"].isin(prop_types) &
    df_full["segment"].isin(segments) &
    df_full["demand_score"].between(demand_range[0], demand_range[1])
].copy()

if price_sensitivity != 1.0 or occupancy_target != 88:
    occ_adjust = (occupancy_target / 100 - 0.88)
    df["recommended_price"] = (
        df["recommended_price"] * (1 + occ_adjust * 0.5 / price_sensitivity)
    ).round(0).astype(int)
    df["annual_revenue_recommended"] = (
        df["recommended_price"] * (occupancy_target / 100) * 12
    ).round(0).astype(int)
    df["annual_revenue_lift"] = df["annual_revenue_recommended"] - df["annual_revenue_current"]
    df["price_gap_pct"] = (
        (df["recommended_price"] - df["current_price"]) / df["current_price"] * 100
    ).round(1)

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("## Marketplace Pricing Intelligence")
st.markdown(
    f"Analyzing **{len(df):,}** listings across "
    f"**{len(cities)}** {'market' if len(cities) == 1 else 'markets'}"
)

if len(df) == 0:
    st.warning("No listings match the current filters.")
    st.stop()

# ── Project overview ───────────────────────────────────────────────────────
with st.expander("Project overview — click to expand", expanded=True):
    st.markdown("""
    <div class="overview-card">
      <div class="overview-title">About this project</div>
      <div class="overview-grid">

        <div class="overview-block">
          <div class="overview-block-label">Client situation</div>
          <div class="overview-block-text">
            A regional property management company operates 2,000+ rental listings across
            five U.S. markets. Their pricing team was setting rents manually, using
            a mix of comparable listings and gut feel — with no systematic process for
            incorporating real-time demand signals or measuring pricing accuracy at scale.
          </div>
        </div>

        <div class="overview-block">
          <div class="overview-block-label">The problem</div>
          <div class="overview-block-text">
            Without a data-driven pricing baseline, the portfolio had two simultaneous
            issues: overpriced listings sitting vacant too long (lost revenue from empty
            units), and underpriced listings in high-demand neighborhoods (revenue left
            on the table). Leadership had no visibility into which situation applied to
            which listings, or how much it was costing them.
          </div>
        </div>

        <div class="overview-block">
          <div class="overview-block-label">How we solved it</div>
          <div class="overview-block-text">
            We built a demand-weighted regression model on comparable transaction prices,
            neighborhood demand scores, and distance-to-center to generate a recommended
            price for each listing. We then quantified the revenue gap between current
            and recommended pricing, segmented the portfolio by market and property type,
            and built this self-serve dashboard so non-technical property managers could
            explore findings and model adoption scenarios on their own.
          </div>
        </div>

        <div class="overview-block">
          <div class="overview-block-label">Technical approach</div>
          <div class="overview-block-text">
            Synthetic dataset of 2,000 listings generated with realistic price
            distributions, demand scores, occupancy rates, and days-on-market.
            Pricing model uses demand-weighted regression on comps, demand score,
            and distance. Revenue lift estimated via price gap × partial occupancy
            adjustment. Built in Python with Streamlit and Plotly.
          </div>
          <div class="tag-row">
            <span class="tag">Python</span>
            <span class="tag">Streamlit</span>
            <span class="tag">Plotly</span>
            <span class="tag">pandas</span>
            <span class="tag">scikit-learn</span>
            <span class="tag">Regression</span>
            <span class="tag">Segmentation</span>
          </div>
        </div>

      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Compute stats ──────────────────────────────────────────────────────────
total_lift = df["annual_revenue_lift"].sum()
median_current = df["current_price"].median()
median_recommended = df["recommended_price"].median()
median_demand = df["demand_score"].median()
pct_overpriced = (df["price_gap_pct"] < -5).mean() * 100
pct_underpriced = (df["price_gap_pct"] > 5).mean() * 100
avg_occupancy = df["occupancy_rate"].mean() * 100
n_overpriced = int((df["price_gap_pct"] < -5).sum())
n_underpriced = int((df["price_gap_pct"] > 5).sum())
total_listings = len(df)

# ── KPI cards ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Portfolio summary</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    dc = "delta-pos" if total_lift >= 0 else "delta-neg"
    st.markdown(f"""<div class="metric-card">
      <div class="metric-label">Total revenue opportunity</div>
      <div class="metric-value">${total_lift/1e6:.2f}M</div>
      <div class="metric-delta <{dc}>">annual lift from repricing</div>
    </div>""", unsafe_allow_html=True)
with c2:
    price_delta = median_recommended - median_current
    delta_pct = price_delta / median_current * 100
    dc = "delta-pos" if price_delta >= 0 else "delta-neg"
    st.markdown(f"""<div class="metric-card">
      <div class="metric-label">Median recommended price</div>
      <div class="metric-value">${median_recommended:,.0f}</div>
      <div class="metric-delta">vs <span class="{dc}">${median_current:,.0f} current ({delta_pct:+.1f}%)</span></div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card">
      <div class="metric-label">Mispriced listings</div>
      <div class="metric-value">{pct_overpriced + pct_underpriced:.0f}%</div>
      <div class="metric-delta">
        <span class="delta-neg">{pct_overpriced:.0f}% overpriced</span> ·
        <span class="delta-pos">{pct_underpriced:.0f}% underpriced</span>
      </div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card">
      <div class="metric-label">Avg portfolio occupancy</div>
      <div class="metric-value">{avg_occupancy:.1f}%</div>
      <div class="metric-delta">median demand score: {median_demand:.0f}/100</div>
    </div>""", unsafe_allow_html=True)

occ_vs_target = avg_occupancy - occupancy_target
occ_label = f"{abs(occ_vs_target):.1f} pts {'above' if occ_vs_target >= 0 else 'below'}"
lift_word = "gained" if total_lift >= 0 else "recovered"

st.markdown(f"""
<div class="explanation-box">
  <strong>What this means:</strong> Of the {total_listings:,} listings analyzed,
  <strong>{n_overpriced:,} ({pct_overpriced:.0f}%)</strong> are priced more than 5% above what the model recommends —
  likely sitting vacant longer than necessary.
  Another <strong>{n_underpriced:,} ({pct_underpriced:.0f}%)</strong> are priced more than 5% below the recommendation,
  leaving revenue on the table. Full repricing adoption would generate an estimated
  <strong>${abs(total_lift)/1e6:.2f}M</strong> {lift_word} annually.
  Average occupancy is <strong>{avg_occupancy:.1f}%</strong> — {occ_label} the {occupancy_target}% target.
</div>
""", unsafe_allow_html=True)

# ── Pricing distribution ───────────────────────────────────────────────────
st.markdown('<div class="section-header">Pricing distribution by segment</div>', unsafe_allow_html=True)

seg_summary = df.groupby("segment", observed=True).agg(
    listings=("current_price", "count"),
    median_current=("current_price", "median"),
    median_recommended=("recommended_price", "median"),
    total_lift=("annual_revenue_lift", "sum"),
    avg_occupancy=("occupancy_rate", "mean"),
).reindex(["Budget", "Mid-Market", "Premium"])

biggest_seg = seg_summary["total_lift"].abs().idxmax() if not seg_summary.empty else "Mid-Market"
biggest_lift = seg_summary.loc[biggest_seg, "total_lift"] if biggest_seg in seg_summary.index else 0

st.markdown(f"""
<div class="explanation-box">
  <strong>How to read this:</strong> Each box shows the spread of rents within a segment.
  The center line is the median price; the box covers the middle 50% of listings; whiskers show the full range.
  <strong>Darker boxes</strong> are current prices — <strong>lighter boxes</strong> are what the model recommends.
  Where the lighter box sits below the darker one, that segment is likely overpriced relative to what the market
  will bear, costing occupancy without a revenue gain.
  The <strong>{biggest_seg}</strong> segment has the largest pricing gap —
  estimated <strong>${abs(biggest_lift)/1e3:.0f}K</strong> annual impact if corrected.
</div>
""", unsafe_allow_html=True)

col_box, col_seg = st.columns([3, 2])
segment_order = ["Budget", "Mid-Market", "Premium"]
seg_colors = {"Budget": "#6ea8d8", "Mid-Market": "#4a90c4", "Premium": "#1d5f9e"}

with col_box:
    fig_box = go.Figure()
    for seg in segment_order:
        seg_df = df[df["segment"] == seg]
        if len(seg_df) == 0:
            continue
        fig_box.add_trace(go.Box(
            y=seg_df["current_price"],
            name=f"{seg} (current)",
            marker_color=seg_colors[seg],
            opacity=0.8,
            legendgroup=seg,
            boxmean=True,
        ))
        fig_box.add_trace(go.Box(
            y=seg_df["recommended_price"],
            name=f"{seg} (recommended)",
            marker_color=seg_colors[seg],
            opacity=0.35,
            legendgroup=seg,
            boxmean=True,
        ))
    fig_box.update_layout(
        **LAYOUT_BASE,
        height=360,
        yaxis=dict(**axis("Monthly rent ($)"), zeroline=False),
        xaxis=dict(showgrid=False, tickfont=dict(size=12, color="#212529")),
    )
    st.plotly_chart(fig_box, use_container_width=True)

with col_seg:
    for seg in segment_order:
        if seg not in seg_summary.index:
            continue
        row = seg_summary.loc[seg]
        lift = row["total_lift"]
        dp = (row["median_recommended"] - row["median_current"]) / row["median_current"] * 100
        dc = "delta-pos" if lift >= 0 else "delta-neg"
        ls = f"${lift/1e3:+.0f}K" if abs(lift) < 1e6 else f"${lift/1e6:+.2f}M"
        st.markdown(f"""
        <div class="seg-card">
          <div style="display:flex;justify-content:space-between;align-items:baseline;">
            <span style="font-weight:700;font-size:14px;color:#212529;">{seg}</span>
            <span style="font-size:12px;color:#6c757d;">{int(row['listings']):,} listings</span>
          </div>
          <div style="font-size:13px;color:#343a40;margin-top:5px;">
            ${row['median_current']:,.0f} → <strong>${row['median_recommended']:,.0f}</strong>
            <span class="{dc}"> ({dp:+.1f}%)</span>
          </div>
          <div style="font-size:12px;color:#495057;margin-top:3px;">
            Annual lift: <span class="{dc}">{ls}</span> ·
            Occupancy: {row['avg_occupancy']*100:.1f}%
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── Opportunity map ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Demand score vs price gap — opportunity map</div>', unsafe_allow_html=True)

hd_op = len(df[(df["demand_score"] > 60) & (df["price_gap_pct"] < -5)])
ld_up = len(df[(df["demand_score"] < 40) & (df["price_gap_pct"] > 5)])

st.markdown(f"""
<div class="explanation-box">
  <strong>How to read this:</strong> Each dot is one listing.
  The <strong>horizontal axis</strong> is neighborhood demand — higher means more renters competing for fewer units.
  The <strong>vertical axis</strong> shows whether the current price is above or below the model's recommendation.
  Dots <em>above</em> the green line are underpriced (could charge more without losing occupancy).
  Dots <em>below</em> the red line are overpriced (likely sitting vacant).
  Dot size reflects square footage. The <strong>top-right quadrant</strong> is the highest-priority opportunity:
  high demand and currently underpriced. Right now <strong>{hd_op:,}</strong> listings are in high-demand
  areas but still overpriced, and <strong>{ld_up:,}</strong> are underpriced despite weak demand —
  both are mispricing patterns worth a closer look.
</div>
""", unsafe_allow_html=True)

col_sc, col_city = st.columns([3, 2])

with col_sc:
    sample_df = df.sample(min(600, len(df)), random_state=42)
    fig_sc = px.scatter(
        sample_df, x="demand_score", y="price_gap_pct",
        color="segment", size="sqft",
        hover_data={"city": True, "property_type": True, "current_price": ":,",
                    "recommended_price": ":,", "days_on_market": True, "sqft": False},
        color_discrete_map=seg_colors,
        labels={"demand_score": "Demand score",
                "price_gap_pct": "Price gap vs recommended (%)", "segment": "Segment"},
        opacity=0.65,
    )
    fig_sc.add_hline(y=5, line_dash="dash", line_color="#198754", line_width=1.5,
                     annotation_text="Underpriced (+5%)", annotation_position="top right",
                     annotation_font=dict(size=12, color="#198754"))
    fig_sc.add_hline(y=-5, line_dash="dash", line_color="#dc3545", line_width=1.5,
                     annotation_text="Overpriced (−5%)", annotation_position="bottom right",
                     annotation_font=dict(size=12, color="#dc3545"))
    fig_sc.add_hline(y=0, line_color="#adb5bd", line_width=0.75)
    fig_sc.update_layout(
        **LAYOUT_BASE, height=360,
        yaxis=dict(**axis("Price gap vs recommended (%)")),
        xaxis=dict(**axis("Demand score")),
    )
    st.plotly_chart(fig_sc, use_container_width=True)

with col_city:
    city_sum = df.groupby("city").agg(
        pct_overpriced=("price_gap_pct", lambda x: (x < -5).mean() * 100),
        pct_underpriced=("price_gap_pct", lambda x: (x > 5).mean() * 100),
    ).sort_values("pct_overpriced", ascending=True)

    worst = city_sum["pct_overpriced"].idxmax() if not city_sum.empty else ""
    best = city_sum["pct_underpriced"].idxmax() if not city_sum.empty else ""
    st.markdown(f"""
    <div class="explanation-box" style="font-size:12.5px;">
      <strong>{worst}</strong> has the most overpriced listings — prioritize downward corrections here.
      <strong>{best}</strong> has the most underpriced listings — biggest near-term upside.
    </div>
    """, unsafe_allow_html=True)

    fig_city = go.Figure()
    fig_city.add_trace(go.Bar(
        y=city_sum.index, x=city_sum["pct_overpriced"],
        name="Overpriced", orientation="h", marker_color="#e07070",
    ))
    fig_city.add_trace(go.Bar(
        y=city_sum.index, x=city_sum["pct_underpriced"],
        name="Underpriced", orientation="h", marker_color="#6ab06a",
    ))
    fig_city.update_layout(
        **LAYOUT_BASE, height=300, barmode="group",
        xaxis=dict(**axis("% of listings")),
        yaxis=dict(showgrid=False, tickfont=dict(size=12, color="#212529")),
    )
    st.plotly_chart(fig_city, use_container_width=True)

# ── ROI explorer ───────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Self-serve ROI explorer</div>', unsafe_allow_html=True)

ec1, ec2, ec3 = st.columns(3)
with ec1:
    sel_city = st.selectbox("Market", ["All"] + sorted(df["city"].unique()))
with ec2:
    sel_type = st.selectbox("Property type", ["All", "Studio", "1BR", "2BR", "3BR+"])
with ec3:
    adopt = st.slider("Model adoption rate (%)", 10, 100, 60, step=5,
                      help="What % of property managers adopt the recommended price")

roi_df = df.copy()
if sel_city != "All":
    roi_df = roi_df[roi_df["city"] == sel_city]
if sel_type != "All":
    roi_df = roi_df[roi_df["property_type"] == sel_type]

eligible = roi_df[roi_df["annual_revenue_lift"] > 0]
adopters = eligible.sample(frac=adopt/100, random_state=99) if len(eligible) > 0 else pd.DataFrame()
proj_lift = adopters["annual_revenue_lift"].sum() if len(adopters) > 0 else 0
avg_lift = adopters["annual_revenue_lift"].mean() if len(adopters) > 0 else 0
n_adopt = len(adopters)

city_str = sel_city if sel_city != "All" else "all markets"
type_str = sel_type if sel_type != "All" else "all property types"

st.markdown(f"""
<div class="explanation-box">
  <strong>How to use this:</strong> In practice, not every property manager will update their price overnight.
  This section lets you model the business case at different rollout speeds.
  Of {len(roi_df):,} listings in <strong>{city_str}</strong> ({type_str}),
  <strong>{len(eligible):,}</strong> stand to gain from repricing.
  At a <strong>{adopt}% adoption rate</strong>, roughly <strong>{n_adopt:,} listings</strong> would reprice,
  generating an estimated <strong>${proj_lift/1e3:.0f}K</strong> in added annual revenue
  — about <strong>${avg_lift:,.0f}/listing/year</strong>.
  Use the slider to stress-test conservative vs. optimistic rollout assumptions.
</div>
""", unsafe_allow_html=True)

r1, r2, r3, r4 = st.columns(4)
with r1: st.metric("Listings in scope", f"{len(roi_df):,}")
with r2: st.metric("Projected adopters", f"{n_adopt:,}")
with r3: st.metric("Portfolio lift",
                   f"${proj_lift/1e3:.0f}K" if proj_lift < 1e6 else f"${proj_lift/1e6:.2f}M")
with r4: st.metric("Avg lift / listing",
                   f"${avg_lift:,.0f}/yr" if avg_lift and not np.isnan(avg_lift) else "—")

if len(roi_df) > 0:
    lift_seg = roi_df.groupby("segment", observed=True)["annual_revenue_lift"].sum().reindex(
        ["Budget", "Mid-Market", "Premium"]
    )
    fig_roi = go.Figure(go.Bar(
        x=lift_seg.index,
        y=lift_seg.values / 1e3,
        marker_color=["#6ea8d8", "#4a90c4", "#1d5f9e"],
        text=[f"${v/1e3:.0f}K" if abs(v) < 1e6 else f"${v/1e6:.2f}M" for v in lift_seg.values],
        textposition="outside",
        textfont=dict(size=13, color="#212529"),
    ))
    fig_roi.update_layout(
        **LAYOUT_BASE, height=260,
        yaxis=dict(**axis("Annual revenue lift ($K)"), zeroline=True, zerolinecolor="#dee2e6"),
        xaxis=dict(showgrid=False, tickfont=dict(size=13, color="#212529")),
        showlegend=False,
    )
    st.plotly_chart(fig_roi, use_container_width=True)

# ── Listing-level detail ───────────────────────────────────────────────────
st.markdown('<div class="section-header">Listing-level detail</div>', unsafe_allow_html=True)

top_lift_val = df["annual_revenue_lift"].quantile(0.9)
st.markdown(f"""
<div class="explanation-box">
  <strong>How to read this:</strong> Each row is one listing. Sort by <em>Revenue lift</em>
  to find the highest-priority repricing opportunities — large price gaps in high-demand neighborhoods.
  <em>DOM</em> (days on market) is one of the clearest real-world signals of overpricing:
  listings sitting vacant 30+ days are almost always priced above what renters will pay.
  The top 10% of listings by opportunity each stand to gain more than
  <strong>${top_lift_val:,.0f}/year</strong> from a price correction.
</div>
""", unsafe_allow_html=True)

sort_col = st.selectbox("Sort by", [
    "annual_revenue_lift", "price_gap_pct", "demand_score", "days_on_market", "current_price"
], format_func=lambda x: {
    "annual_revenue_lift": "Revenue lift (highest first)",
    "price_gap_pct": "Price gap %",
    "demand_score": "Demand score",
    "days_on_market": "Days on market",
    "current_price": "Current price",
}[x])

disp = df.sort_values(sort_col, ascending=False).head(200)[[
    "city", "property_type", "segment", "sqft",
    "current_price", "recommended_price", "price_gap_pct",
    "demand_score", "occupancy_rate", "days_on_market", "annual_revenue_lift",
]].rename(columns={
    "city": "City", "property_type": "Type", "segment": "Segment", "sqft": "Sq ft",
    "current_price": "Current ($)", "recommended_price": "Recommended ($)",
    "price_gap_pct": "Gap (%)", "demand_score": "Demand", "occupancy_rate": "Occupancy",
    "days_on_market": "DOM", "annual_revenue_lift": "Annual lift ($)",
})
disp["Occupancy"] = (disp["Occupancy"] * 100).round(1).astype(str) + "%"

st.dataframe(
    disp, use_container_width=True, height=320, hide_index=True,
    column_config={
        "Current ($)": st.column_config.NumberColumn(format="$%d"),
        "Recommended ($)": st.column_config.NumberColumn(format="$%d"),
        "Gap (%)": st.column_config.NumberColumn(format="%.1f%%"),
        "Annual lift ($)": st.column_config.NumberColumn(format="$%d"),
        "Demand": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
    },
)

st.caption(
    "Pricing recommendations generated via demand-weighted regression on comparable transactions, "
    "demand score, and distance-to-center. Revenue lift assumes partial occupancy adjustment. "
    "Synthetic dataset — 2,000 listings across 5 U.S. markets."
)
