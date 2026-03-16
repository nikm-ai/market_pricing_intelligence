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
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global styles ──────────────────────────────────────────────────────────
# All colors use Streamlit CSS variables so they work in light AND dark mode.
st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1180px; }

  /* ── Section dividers ── */
  .sec-header {
    font-size: 11px; font-weight: 700; letter-spacing: 0.09em;
    text-transform: uppercase; color: var(--text-color);
    opacity: 0.6;
    margin: 1.75rem 0 0.6rem; padding-bottom: 6px;
    border-bottom: 1px solid rgba(128,128,128,0.25);
  }

  /* ── KPI cards ── */
  .kpi-card {
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 8px; padding: 1rem 1.2rem;
    background: rgba(128,128,128,0.04);
  }
  .kpi-label {
    font-size: 10px; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; opacity: 0.55; color: var(--text-color);
    margin-bottom: 5px;
  }
  .kpi-value {
    font-size: 26px; font-weight: 700; line-height: 1.15;
    color: var(--text-color);
  }
  .kpi-sub { font-size: 12px; margin-top: 3px; opacity: 0.7; color: var(--text-color); }
  .pos { color: #27a862 !important; opacity: 1 !important; font-weight: 600; }
  .neg { color: #e5534b !important; opacity: 1 !important; font-weight: 600; }

  /* ── Explanation callouts ── */
  .callout {
    border-left: 3px solid #4a90c4;
    border-radius: 0 6px 6px 0;
    padding: 0.65rem 1rem;
    margin-bottom: 1rem;
    font-size: 13px; line-height: 1.65;
    color: var(--text-color);
    background: rgba(74,144,196,0.08);
  }
  .callout b { color: #4a90c4; }

  /* ── Segment summary cards ── */
  .seg-card {
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 9px;
    background: rgba(128,128,128,0.04);
  }
  .seg-name { font-size: 14px; font-weight: 700; color: var(--text-color); }
  .seg-count { font-size: 12px; opacity: 0.55; color: var(--text-color); }
  .seg-price { font-size: 13px; color: var(--text-color); margin-top: 4px; }
  .seg-lift  { font-size: 12px; opacity: 0.75; color: var(--text-color); margin-top: 2px; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] { background: rgba(128,128,128,0.04); }
</style>
""", unsafe_allow_html=True)

# ── Chart helpers ──────────────────────────────────────────────────────────
# Charts always use a white bg so fonts are always dark — no CSS variable ambiguity.
FONT = dict(size=13, color="#1a1a1a", family="Arial, sans-serif")
LEGEND = dict(
    orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
    font=dict(size=12, color="#1a1a1a"), bgcolor="rgba(0,0,0,0)",
)

def ax(title, grid=True):
    return dict(
        title=dict(text=title, font=dict(size=13, color="#1a1a1a")),
        tickfont=dict(size=12, color="#1a1a1a"),
        gridcolor="#ebebeb" if grid else "rgba(0,0,0,0)",
        showgrid=grid,
        zeroline=False,
    )

BASE = dict(
    plot_bgcolor="white", paper_bgcolor="white",
    font=FONT, margin=dict(l=4, r=4, t=14, b=4),
    legend=LEGEND,
)

# ── Data ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return generate_marketplace_data(n=2000)

df_full = load_data()

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Filters")
    cities = st.multiselect("City", sorted(df_full["city"].unique()),
                             default=sorted(df_full["city"].unique()))
    prop_types = st.multiselect("Property type", ["Studio","1BR","2BR","3BR+"],
                                 default=["Studio","1BR","2BR","3BR+"])
    segments = st.multiselect("Market segment", ["Budget","Mid-Market","Premium"],
                               default=["Budget","Mid-Market","Premium"])
    demand_range = st.slider("Demand score", 0, 100, (0, 100))
    st.markdown("---")
    st.markdown("### Model settings")
    elasticity = st.slider("Demand elasticity", 0.5, 2.0, 1.0, 0.1,
                            help="Higher = demand more sensitive to price changes")
    occ_target = st.slider("Target occupancy %", 70, 98, 88)
    st.markdown("---")
    st.caption("Synthetic data · 2,000 listings · 5 markets")

# ── Filter & recompute ─────────────────────────────────────────────────────
df = df_full[
    df_full["city"].isin(cities) &
    df_full["property_type"].isin(prop_types) &
    df_full["segment"].isin(segments) &
    df_full["demand_score"].between(demand_range[0], demand_range[1])
].copy()

if elasticity != 1.0 or occ_target != 88:
    adj = (occ_target / 100 - 0.88)
    df["recommended_price"] = (df["recommended_price"] * (1 + adj * 0.5 / elasticity)).round(0).astype(int)
    df["annual_revenue_recommended"] = (df["recommended_price"] * (occ_target / 100) * 12).round(0).astype(int)
    df["annual_revenue_lift"] = df["annual_revenue_recommended"] - df["annual_revenue_current"]
    df["price_gap_pct"] = ((df["recommended_price"] - df["current_price"]) / df["current_price"] * 100).round(1)

# ── Page header ────────────────────────────────────────────────────────────
st.markdown("## Marketplace Pricing Intelligence")
st.markdown(f"Analyzing **{len(df):,}** listings across **{len(cities)}** {'market' if len(cities)==1 else 'markets'}")

if len(df) == 0:
    st.warning("No listings match the current filters.")
    st.stop()

# ── Project overview — native Streamlit, no raw HTML ──────────────────────
st.markdown('<div class="sec-header">Project overview</div>', unsafe_allow_html=True)

with st.expander("Expand to read about the client situation, problem, and methodology", expanded=True):
    oc1, oc2, oc3, oc4 = st.columns(4)

    with oc1:
        st.markdown("**Client situation**")
        st.caption(
            "A regional property management company operates 2,000+ rental listings across five "
            "U.S. markets. Their pricing team set rents manually — using comparable listings and "
            "gut feel — with no systematic process for incorporating demand signals or measuring "
            "pricing accuracy at scale."
        )

    with oc2:
        st.markdown("**The problem**")
        st.caption(
            "Without a data-driven baseline, the portfolio had two simultaneous issues: overpriced "
            "listings sitting vacant too long (lost revenue from empty units), and underpriced listings "
            "in high-demand areas (revenue left on the table). Leadership had no visibility into which "
            "situation applied to which listings, or what it was costing them."
        )

    with oc3:
        st.markdown("**How we solved it**")
        st.caption(
            "We built a demand-weighted regression model using comparable transaction prices, "
            "neighborhood demand scores, and distance-to-center to generate a recommended price per listing. "
            "We quantified the revenue gap, segmented the portfolio by market and property type, and built "
            "this self-serve dashboard so non-technical managers could explore findings independently."
        )

    with oc4:
        st.markdown("**Technical approach**")
        st.caption(
            "Synthetic dataset of 2,000 listings with realistic price distributions, demand scores, "
            "occupancy rates, and days-on-market. Pricing model uses demand-weighted regression on comps, "
            "demand score, and distance. Revenue lift estimated via price gap × occupancy adjustment."
        )
        st.markdown(
            "`Python` `pandas` `scikit-learn` `Streamlit` `Plotly` `Regression` `Segmentation`",
        )

# ── Compute stats ──────────────────────────────────────────────────────────
total_lift    = df["annual_revenue_lift"].sum()
med_current   = df["current_price"].median()
med_rec       = df["recommended_price"].median()
med_demand    = df["demand_score"].median()
pct_over      = (df["price_gap_pct"] < -5).mean() * 100
pct_under     = (df["price_gap_pct"] > 5).mean() * 100
avg_occ       = df["occupancy_rate"].mean() * 100
n_over        = int((df["price_gap_pct"] < -5).sum())
n_under       = int((df["price_gap_pct"] > 5).sum())
n_total       = len(df)
price_delta   = med_rec - med_current
delta_pct     = price_delta / med_current * 100
occ_gap       = avg_occ - occ_target
lift_word     = "gained" if total_lift >= 0 else "recovered"

# ── KPI cards ──────────────────────────────────────────────────────────────
st.markdown('<div class="sec-header">Portfolio summary</div>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
with k1:
    dc = "pos" if total_lift >= 0 else "neg"
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-label">Total revenue opportunity</div>
      <div class="kpi-value">${total_lift/1e6:.2f}M</div>
      <div class="kpi-sub <{dc}>">annual lift from repricing</div>
    </div>""", unsafe_allow_html=True)
with k2:
    dc = "pos" if price_delta >= 0 else "neg"
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-label">Median recommended price</div>
      <div class="kpi-value">${med_rec:,.0f}</div>
      <div class="kpi-sub">vs <span class="{dc}">${med_current:,.0f} current ({delta_pct:+.1f}%)</span></div>
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
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-label">Avg portfolio occupancy</div>
      <div class="kpi-value">{avg_occ:.1f}%</div>
      <div class="kpi-sub">median demand score: {med_demand:.0f}/100</div>
    </div>""", unsafe_allow_html=True)

occ_dir = f"{abs(occ_gap):.1f} pts {'above' if occ_gap >= 0 else 'below'}"
st.markdown(f"""<div class="callout">
  <b>What this means:</b> Of {n_total:,} listings analyzed,
  <b>{n_over:,} ({pct_over:.0f}%)</b> are priced more than 5% above the model's recommendation —
  likely sitting vacant longer than necessary. Another
  <b>{n_under:,} ({pct_under:.0f}%)</b> are priced more than 5% below, leaving revenue on the table.
  Full repricing adoption would generate an estimated <b>${abs(total_lift)/1e6:.2f}M</b> {lift_word} annually.
  Average occupancy is <b>{avg_occ:.1f}%</b> — {occ_dir} the {occ_target}% target.
</div>""", unsafe_allow_html=True)

# ── Pricing distribution ───────────────────────────────────────────────────
st.markdown('<div class="sec-header">Pricing distribution by segment</div>', unsafe_allow_html=True)

seg_sum = df.groupby("segment", observed=True).agg(
    listings=("current_price","count"),
    med_cur=("current_price","median"),
    med_rec=("recommended_price","median"),
    total_lift=("annual_revenue_lift","sum"),
    avg_occ=("occupancy_rate","mean"),
).reindex(["Budget","Mid-Market","Premium"])

big_seg  = seg_sum["total_lift"].abs().idxmax() if not seg_sum.empty else "Mid-Market"
big_lift = seg_sum.loc[big_seg,"total_lift"] if big_seg in seg_sum.index else 0

st.markdown(f"""<div class="callout">
  <b>How to read this:</b> Each box shows the spread of rents within a segment.
  The center line is the median; the box covers the middle 50% of listings; whiskers show the full range.
  <b>Darker boxes</b> are current prices — <b>lighter boxes</b> are what the model recommends.
  Where the lighter box sits below the darker one, that segment is likely overpriced relative to what
  the market will bear. The <b>{big_seg}</b> segment has the largest pricing gap —
  estimated <b>${abs(big_lift)/1e3:.0f}K</b> annual impact if corrected.
</div>""", unsafe_allow_html=True)

seg_order  = ["Budget","Mid-Market","Premium"]
seg_colors = {"Budget":"#6ea8d8","Mid-Market":"#4a90c4","Premium":"#1d5f9e"}

col_box, col_seg = st.columns([3, 2])
with col_box:
    fig_box = go.Figure()
    for seg in seg_order:
        sd = df[df["segment"] == seg]
        if len(sd) == 0: continue
        fig_box.add_trace(go.Box(
            y=sd["current_price"], name=f"{seg} (current)",
            marker_color=seg_colors[seg], opacity=0.82, legendgroup=seg, boxmean=True,
        ))
        fig_box.add_trace(go.Box(
            y=sd["recommended_price"], name=f"{seg} (recommended)",
            marker_color=seg_colors[seg], opacity=0.35, legendgroup=seg, boxmean=True,
        ))
    fig_box.update_layout(
        **BASE, height=360,
        yaxis=dict(**ax("Monthly rent ($)")),
        xaxis=dict(showgrid=False, tickfont=dict(size=12, color="#1a1a1a")),
    )
    st.plotly_chart(fig_box, use_container_width=True)

with col_seg:
    for seg in seg_order:
        if seg not in seg_sum.index: continue
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
          <div class="seg-price">${row['med_cur']:,.0f} → <b>${row['med_rec']:,.0f}</b>
            <span class="{dc}"> ({dp:+.1f}%)</span></div>
          <div class="seg-lift">Annual lift: <span class="{dc}">{ls}</span> ·
            Occupancy: {row['avg_occ']*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)

# ── Opportunity map ────────────────────────────────────────────────────────
st.markdown('<div class="sec-header">Demand score vs price gap — opportunity map</div>', unsafe_allow_html=True)

hd_op = len(df[(df["demand_score"] > 60) & (df["price_gap_pct"] < -5)])
ld_up = len(df[(df["demand_score"] < 40) & (df["price_gap_pct"] > 5)])

st.markdown(f"""<div class="callout">
  <b>How to read this:</b> Each dot is one listing.
  The <b>horizontal axis</b> is neighborhood demand — higher means more renters competing for fewer units.
  The <b>vertical axis</b> shows whether the current price is above or below the model's recommendation.
  Dots <em>above</em> the green line are underpriced (could charge more without losing occupancy).
  Dots <em>below</em> the red line are overpriced (likely sitting vacant).
  Dot size reflects square footage. The <b>top-right quadrant</b> is the highest-priority opportunity:
  high demand and currently underpriced. Currently <b>{hd_op:,}</b> listings are in high-demand areas
  but still overpriced, and <b>{ld_up:,}</b> are underpriced despite weak demand — both worth investigating.
</div>""", unsafe_allow_html=True)

col_sc, col_city = st.columns([3, 2])
with col_sc:
    samp = df.sample(min(600, len(df)), random_state=42)
    fig_sc = px.scatter(
        samp, x="demand_score", y="price_gap_pct", color="segment", size="sqft",
        hover_data={"city":True,"property_type":True,"current_price":":,",
                    "recommended_price":":,","days_on_market":True,"sqft":False},
        color_discrete_map=seg_colors,
        labels={"demand_score":"Demand score","price_gap_pct":"Price gap vs recommended (%)","segment":"Segment"},
        opacity=0.65,
    )
    fig_sc.add_hline(y=5,  line_dash="dash", line_color="#27a862", line_width=1.5,
                     annotation_text="Underpriced (+5%)", annotation_position="top right",
                     annotation_font=dict(size=12, color="#27a862"))
    fig_sc.add_hline(y=-5, line_dash="dash", line_color="#e5534b", line_width=1.5,
                     annotation_text="Overpriced (−5%)", annotation_position="bottom right",
                     annotation_font=dict(size=12, color="#e5534b"))
    fig_sc.add_hline(y=0, line_color="#cccccc", line_width=0.75)
    fig_sc.update_layout(**BASE, height=360,
        yaxis=dict(**ax("Price gap vs recommended (%)")),
        xaxis=dict(**ax("Demand score")),
    )
    st.plotly_chart(fig_sc, use_container_width=True)

with col_city:
    city_s = df.groupby("city").agg(
        pct_over=("price_gap_pct",  lambda x: (x < -5).mean() * 100),
        pct_under=("price_gap_pct", lambda x: (x > 5).mean()  * 100),
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

# ── ROI explorer ───────────────────────────────────────────────────────────
st.markdown('<div class="sec-header">Self-serve ROI explorer</div>', unsafe_allow_html=True)

ec1, ec2, ec3 = st.columns(3)
with ec1: sel_city = st.selectbox("Market", ["All"] + sorted(df["city"].unique()))
with ec2: sel_type = st.selectbox("Property type", ["All","Studio","1BR","2BR","3BR+"])
with ec3: adopt = st.slider("Model adoption rate (%)", 10, 100, 60, 5,
                             help="% of property managers who adopt the recommended price")

roi_df   = df[df["city"]==sel_city].copy() if sel_city!="All" else df.copy()
if sel_type != "All": roi_df = roi_df[roi_df["property_type"]==sel_type]

eligible  = roi_df[roi_df["annual_revenue_lift"] > 0]
adopters  = eligible.sample(frac=adopt/100, random_state=99) if len(eligible) > 0 else pd.DataFrame()
proj_lift = adopters["annual_revenue_lift"].sum() if len(adopters) > 0 else 0
avg_lift  = adopters["annual_revenue_lift"].mean() if len(adopters) > 0 else 0
n_adopt   = len(adopters)

city_str = sel_city if sel_city != "All" else "all markets"
type_str = sel_type if sel_type != "All" else "all property types"

st.markdown(f"""<div class="callout">
  <b>How to use this:</b> In practice, not every property manager updates their price overnight.
  This section models the business case at different rollout speeds.
  Of {len(roi_df):,} listings in <b>{city_str}</b> ({type_str}),
  <b>{len(eligible):,}</b> stand to gain from repricing. At a <b>{adopt}% adoption rate</b>,
  roughly <b>{n_adopt:,} listings</b> would reprice, generating an estimated
  <b>${proj_lift/1e3:.0f}K</b> in added annual revenue —
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
    lift_seg = roi_df.groupby("segment", observed=True)["annual_revenue_lift"].sum().reindex(
        ["Budget","Mid-Market","Premium"])
    fig_roi = go.Figure(go.Bar(
        x=lift_seg.index, y=lift_seg.values / 1e3,
        marker_color=["#6ea8d8","#4a90c4","#1d5f9e"],
        text=[f"${v/1e3:.0f}K" if abs(v)<1e6 else f"${v/1e6:.2f}M" for v in lift_seg.values],
        textposition="outside", textfont=dict(size=13, color="#1a1a1a"),
    ))
    fig_roi.update_layout(**BASE, height=260, showlegend=False,
        yaxis={**ax("Annual revenue lift ($K)"), "zeroline": True, "zerolinecolor": "#dddddd"},
        xaxis=dict(showgrid=False, tickfont=dict(size=13, color="#1a1a1a")),
    )
    st.plotly_chart(fig_roi, use_container_width=True)

# ── Listing-level detail ───────────────────────────────────────────────────
st.markdown('<div class="sec-header">Listing-level detail</div>', unsafe_allow_html=True)

top_thresh = df["annual_revenue_lift"].quantile(0.9)
st.markdown(f"""<div class="callout">
  <b>How to read this:</b> Each row is one listing. Sort by <em>Revenue lift</em> to surface
  the highest-priority repricing opportunities — large price gaps in high-demand neighborhoods.
  <em>DOM</em> (days on market) is one of the clearest signals of overpricing: listings vacant
  for 30+ days are almost always priced above what renters will pay. The top 10% of listings
  by opportunity each stand to gain more than <b>${top_thresh:,.0f}/year</b> from a price correction.
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

st.dataframe(disp, use_container_width=True, height=320, hide_index=True,
    column_config={
        "Current ($)":     st.column_config.NumberColumn(format="$%d"),
        "Recommended ($)": st.column_config.NumberColumn(format="$%d"),
        "Gap (%)":         st.column_config.NumberColumn(format="%.1f%%"),
        "Annual lift ($)": st.column_config.NumberColumn(format="$%d"),
        "Demand":          st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
    },
)

st.caption(
    "Pricing recommendations via demand-weighted regression on comparable transactions, "
    "demand score, and distance-to-center. Revenue lift assumes partial occupancy adjustment. "
    "Synthetic dataset — 2,000 listings · 5 U.S. markets."
)
