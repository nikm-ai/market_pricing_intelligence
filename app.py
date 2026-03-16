import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

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
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        border: 1px solid #e9ecef;
    }
    .metric-label { font-size: 12px; color: #495057; font-weight: 600; margin-bottom: 4px; letter-spacing: 0.04em; }
    .metric-value { font-size: 24px; font-weight: 600; color: #212529; }
    .metric-delta { font-size: 12px; margin-top: 2px; color: #343a40; }
    .delta-pos { color: #198754; font-weight: 500; }
    .delta-neg { color: #dc3545; font-weight: 500; }
    .section-header {
        font-size: 14px;
        font-weight: 600;
        color: #212529;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin: 1.5rem 0 0.5rem;
        padding-bottom: 6px;
        border-bottom: 1px solid #dee2e6;
    }
    .explanation-box {
        background: #f0f4f8;
        border-left: 3px solid #4a90c4;
        border-radius: 0 6px 6px 0;
        padding: 0.65rem 1rem;
        margin-bottom: 0.85rem;
        font-size: 13px;
        color: #212529;
        line-height: 1.55;
    }
    .explanation-box strong { color: #1d5f9e; }
    div[data-testid="stSidebar"] { background: #f8f9fa; }
    .stSelectbox label, .stMultiSelect label, .stSlider label { font-size: 13px; font-weight: 500; color: #212529; }
</style>
""", unsafe_allow_html=True)

FONT = dict(size=13, color="#212529", family="Arial, sans-serif")
AXIS_STYLE = dict(
    tickfont=dict(size=12, color="#212529"),
    titlefont=dict(size=13, color="#212529"),
    gridcolor="#e9ecef",
)


@st.cache_data
def load_data():
    return generate_marketplace_data(n=2000)


df_full = load_data()

with st.sidebar:
    st.markdown("### Filters")

    cities = st.multiselect(
        "City",
        options=sorted(df_full["city"].unique()),
        default=sorted(df_full["city"].unique()),
    )

    prop_types = st.multiselect(
        "Property type",
        options=["Studio", "1BR", "2BR", "3BR+"],
        default=["Studio", "1BR", "2BR", "3BR+"],
    )

    segments = st.multiselect(
        "Market segment",
        options=["Budget", "Mid-Market", "Premium"],
        default=["Budget", "Mid-Market", "Premium"],
    )

    demand_range = st.slider(
        "Demand score range",
        min_value=0, max_value=100,
        value=(0, 100), step=1,
    )

    st.markdown("---")
    st.markdown("### Model settings")
    price_sensitivity = st.slider(
        "Demand elasticity",
        min_value=0.5, max_value=2.0, value=1.0, step=0.1,
        help="How strongly demand responds to price changes. Higher = more elastic market."
    )
    occupancy_target = st.slider(
        "Target occupancy %",
        min_value=70, max_value=98, value=88, step=1
    )

    st.markdown("---")
    st.caption("Synthetic data · 2,000 listings · 5 markets")


df = df_full[
    df_full["city"].isin(cities) &
    df_full["property_type"].isin(prop_types) &
    df_full["segment"].isin(segments) &
    df_full["demand_score"].between(demand_range[0], demand_range[1])
].copy()

if price_sensitivity != 1.0 or occupancy_target != 88:
    base_target = 0.88
    occ_adjust = (occupancy_target / 100 - base_target)
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


st.markdown("## Marketplace Pricing Intelligence")
st.markdown(
    f"Analyzing **{len(df):,}** listings across **{len(cities)}** "
    f"{'market' if len(cities) == 1 else 'markets'}"
)

if len(df) == 0:
    st.warning("No listings match the current filters.")
    st.stop()

total_lift = df["annual_revenue_lift"].sum()
median_current = df["current_price"].median()
median_recommended = df["recommended_price"].median()
median_demand = df["demand_score"].median()
pct_overpriced = (df["price_gap_pct"] < -5).mean() * 100
pct_underpriced = (df["price_gap_pct"] > 5).mean() * 100
avg_occupancy = df["occupancy_rate"].mean() * 100
total_listings = len(df)
n_overpriced = int((df["price_gap_pct"] < -5).sum())
n_underpriced = int((df["price_gap_pct"] > 5).sum())

col1, col2, col3, col4 = st.columns(4)

with col1:
    delta_sign = "+" if total_lift >= 0 else ""
    delta_class = "delta-pos" if total_lift >= 0 else "delta-neg"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">TOTAL REVENUE OPPORTUNITY</div>
        <div class="metric-value">${total_lift/1e6:.2f}M</div>
        <div class="metric-delta {delta_class}">{delta_sign}annual lift from repricing</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    price_delta = median_recommended - median_current
    delta_pct = price_delta / median_current * 100
    delta_class = "delta-pos" if price_delta >= 0 else "delta-neg"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">MEDIAN RECOMMENDED PRICE</div>
        <div class="metric-value">${median_recommended:,.0f}</div>
        <div class="metric-delta {delta_class}">vs ${median_current:,.0f} current ({delta_pct:+.1f}%)</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">MISPRICED LISTINGS</div>
        <div class="metric-value">{pct_overpriced + pct_underpriced:.0f}%</div>
        <div class="metric-delta"><span class="delta-neg">{pct_overpriced:.0f}% overpriced</span> · <span class="delta-pos">{pct_underpriced:.0f}% underpriced</span></div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">AVG PORTFOLIO OCCUPANCY</div>
        <div class="metric-value">{avg_occupancy:.1f}%</div>
        <div class="metric-delta">median demand score: {median_demand:.0f}/100</div>
    </div>
    """, unsafe_allow_html=True)

occ_vs_target = avg_occupancy - occupancy_target
occ_label = f"{abs(occ_vs_target):.1f} points {'above' if occ_vs_target >= 0 else 'below'}"

st.markdown(f"""
<div class="explanation-box">
  <strong>What this means:</strong> Of the {total_listings:,} listings currently being analyzed,
  <strong>{n_overpriced:,} ({pct_overpriced:.0f}%)</strong> are priced more than 5% above what
  the model recommends — they're likely sitting vacant longer than necessary. Another
  <strong>{n_underpriced:,} ({pct_underpriced:.0f}%)</strong> are priced more than 5% below the
  recommendation, leaving revenue on the table. If property managers adopted the model's suggested
  prices, the estimated net impact across the portfolio would be
  <strong>${abs(total_lift)/1e6:.2f}M {'gained' if total_lift >= 0 else 'recovered'}</strong> annually.
  Average occupancy is currently <strong>{avg_occupancy:.1f}%</strong> — {occ_label} the {occupancy_target}% target.
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

most_mispriced_seg = seg_summary["total_lift"].abs().idxmax() if not seg_summary.empty else "Mid-Market"
most_mispriced_lift = seg_summary.loc[most_mispriced_seg, "total_lift"] if most_mispriced_seg in seg_summary.index else 0

st.markdown(f"""
<div class="explanation-box">
  <strong>How to read this:</strong> Each box shows the spread of rental prices within a segment —
  the middle line is the median, the box covers the middle 50% of listings, and the whiskers extend
  to the full range. Darker boxes are current prices; lighter boxes are what the model recommends.
  Where the recommended box sits <em>below</em> the current box, that segment is likely overpriced
  relative to what the market will support — meaning landlords are losing occupancy without a compensating
  revenue gain. The <strong>{most_mispriced_seg}</strong> segment has the largest pricing gap,
  with an estimated <strong>${abs(most_mispriced_lift)/1e3:.0f}K</strong> annual impact if corrected.
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([3, 2])

with col_left:
    segment_order = ["Budget", "Mid-Market", "Premium"]
    seg_colors = {"Budget": "#6ea8d8", "Mid-Market": "#4a90c4", "Premium": "#1d5f9e"}

    fig_box = go.Figure()
    for seg in segment_order:
        seg_df = df[df["segment"] == seg]
        fig_box.add_trace(go.Box(
            y=seg_df["current_price"],
            name=f"{seg} (current)",
            marker_color=seg_colors[seg],
            opacity=0.75,
            legendgroup=seg,
            boxmean=True,
        ))
        fig_box.add_trace(go.Box(
            y=seg_df["recommended_price"],
            name=f"{seg} (recommended)",
            marker_color=seg_colors[seg],
            marker_symbol="diamond",
            legendgroup=seg,
            boxmean=True,
            opacity=0.4,
        ))

    fig_box.update_layout(
        height=340,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                    font=dict(size=12, color="#212529")),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=FONT,
        yaxis=dict(**AXIS_STYLE, title="Monthly rent ($)"),
        xaxis=dict(showgrid=False, tickfont=dict(size=12, color="#212529")),
        boxgroupgap=0.2,
        boxgap=0.15,
    )
    st.plotly_chart(fig_box, use_container_width=True)

with col_right:
    for seg in segment_order:
        if seg not in seg_summary.index:
            continue
        row = seg_summary.loc[seg]
        lift = row["total_lift"]
        delta_pct = (row["median_recommended"] - row["median_current"]) / row["median_current"] * 100
        delta_class = "delta-pos" if lift >= 0 else "delta-neg"
        lift_str = f"${lift/1e3:+.0f}K" if abs(lift) < 1e6 else f"${lift/1e6:+.2f}M"
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <span style="font-weight: 600; font-size: 14px; color: #212529;">{seg}</span>
                <span style="font-size: 12px; color: #495057;">{int(row['listings']):,} listings</span>
            </div>
            <div style="margin-top: 6px; font-size: 13px; color: #343a40;">
                ${row['median_current']:,.0f} → <strong>${row['median_recommended']:,.0f}</strong>
                <span class="{delta_class}"> ({delta_pct:+.1f}%)</span>
            </div>
            <div style="font-size: 12px; margin-top: 4px; color: #495057;">
                Annual lift: <span class="{delta_class}">{lift_str}</span> ·
                Occupancy: {row['avg_occupancy']*100:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Opportunity map ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Demand score vs price gap — opportunity map</div>', unsafe_allow_html=True)

high_demand_overpriced = df[(df["demand_score"] > 60) & (df["price_gap_pct"] < -5)]
low_demand_underpriced = df[(df["demand_score"] < 40) & (df["price_gap_pct"] > 5)]

st.markdown(f"""
<div class="explanation-box">
  <strong>How to read this:</strong> Each dot is a listing. The horizontal axis measures how much
  rental demand exists in that neighborhood — higher scores mean more renters competing for fewer units.
  The vertical axis shows whether the current price is above or below the model's recommendation.
  Dots <em>above</em> the green dashed line are underpriced (charging less than the market would support).
  Dots <em>below</em> the red dashed line are overpriced (likely sitting vacant as a result).
  Dot size reflects square footage. The highest-priority repricing opportunities are in the
  <strong>top-right quadrant</strong> — high demand and currently underpriced, meaning a rent increase
  is both supported by data and unlikely to hurt occupancy. Currently
  <strong>{len(high_demand_overpriced):,}</strong> listings are in high-demand areas but still overpriced,
  and <strong>{len(low_demand_underpriced):,}</strong> are underpriced despite being in weaker-demand areas —
  both are signals worth investigating.
</div>
""", unsafe_allow_html=True)

col_scatter, col_city = st.columns([3, 2])

with col_scatter:
    sample_df = df.sample(min(600, len(df)), random_state=42)

    fig_scatter = px.scatter(
        sample_df,
        x="demand_score",
        y="price_gap_pct",
        color="segment",
        size="sqft",
        hover_data={
            "city": True,
            "property_type": True,
            "current_price": ":,",
            "recommended_price": ":,",
            "days_on_market": True,
            "sqft": False,
        },
        color_discrete_map={"Budget": "#6ea8d8", "Mid-Market": "#4a90c4", "Premium": "#1d5f9e"},
        labels={
            "demand_score": "Demand score",
            "price_gap_pct": "Price gap vs recommended (%)",
            "segment": "Segment",
        },
        opacity=0.65,
    )
    fig_scatter.add_hline(y=5, line_dash="dash", line_color="#198754", line_width=1.5,
                          annotation_text="Underpriced threshold (+5%)",
                          annotation_position="top right",
                          annotation_font=dict(size=12, color="#198754"))
    fig_scatter.add_hline(y=-5, line_dash="dash", line_color="#dc3545", line_width=1.5,
                          annotation_text="Overpriced threshold (−5%)",
                          annotation_position="bottom right",
                          annotation_font=dict(size=12, color="#dc3545"))
    fig_scatter.add_hline(y=0, line_color="#adb5bd", line_width=0.75)

    fig_scatter.update_layout(
        height=360,
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=FONT,
        yaxis=dict(**AXIS_STYLE, title="Price gap vs recommended (%)"),
        xaxis=dict(**AXIS_STYLE, title="Demand score"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                    font=dict(size=12, color="#212529")),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_city:
    city_summary = df.groupby("city").agg(
        listings=("current_price", "count"),
        median_demand=("demand_score", "median"),
        pct_overpriced=("price_gap_pct", lambda x: (x < -5).mean() * 100),
        pct_underpriced=("price_gap_pct", lambda x: (x > 5).mean() * 100),
        total_lift=("annual_revenue_lift", "sum"),
    ).sort_values("median_demand", ascending=False)

    worst_city = city_summary["pct_overpriced"].idxmax() if not city_summary.empty else ""
    best_opp_city = city_summary["pct_underpriced"].idxmax() if not city_summary.empty else ""

    st.markdown(f"""
    <div class="explanation-box" style="font-size:12px;">
      <strong>By market:</strong> <strong>{worst_city}</strong> has the highest share of overpriced
      listings — worth prioritizing for downward price corrections.
      <strong>{best_opp_city}</strong> has the most underpriced listings, representing the largest
      near-term revenue upside.
    </div>
    """, unsafe_allow_html=True)

    fig_city = go.Figure()
    fig_city.add_trace(go.Bar(
        y=city_summary.index,
        x=city_summary["pct_overpriced"],
        name="Overpriced",
        orientation="h",
        marker_color="#e08080",
    ))
    fig_city.add_trace(go.Bar(
        y=city_summary.index,
        x=city_summary["pct_underpriced"],
        name="Underpriced",
        orientation="h",
        marker_color="#80b080",
    ))
    fig_city.update_layout(
        height=320,
        barmode="group",
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=FONT,
        xaxis=dict(**AXIS_STYLE, title="% of listings"),
        yaxis=dict(showgrid=False, tickfont=dict(size=12, color="#212529")),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                    font=dict(size=12, color="#212529")),
    )
    st.plotly_chart(fig_city, use_container_width=True)

# ── ROI explorer ───────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Self-serve ROI explorer</div>', unsafe_allow_html=True)

exp_col1, exp_col2, exp_col3 = st.columns(3)
with exp_col1:
    selected_city = st.selectbox("Market", options=["All"] + sorted(df["city"].unique()))
with exp_col2:
    selected_type = st.selectbox("Property type", options=["All"] + ["Studio", "1BR", "2BR", "3BR+"])
with exp_col3:
    adoption_rate = st.slider("Model adoption rate (%)", 10, 100, 60, step=5,
                               help="What % of property managers adopt the recommended price")

roi_df = df.copy()
if selected_city != "All":
    roi_df = roi_df[roi_df["city"] == selected_city]
if selected_type != "All":
    roi_df = roi_df[roi_df["property_type"] == selected_type]

adoption = adoption_rate / 100
eligible = roi_df[roi_df["annual_revenue_lift"] > 0]
adopters = eligible.sample(frac=adoption, random_state=99) if len(eligible) > 0 else pd.DataFrame()

projected_lift = adopters["annual_revenue_lift"].sum() if len(adopters) > 0 else 0
avg_lift_per_listing = adopters["annual_revenue_lift"].mean() if len(adopters) > 0 else 0
n_adopters = len(adopters)
n_eligible = len(eligible)

city_str = selected_city if selected_city != "All" else "all markets"
type_str = selected_type if selected_type != "All" else "all property types"

st.markdown(f"""
<div class="explanation-box">
  <strong>How to use this:</strong> In practice, not every property manager will update their price
  overnight — this section models what the business case looks like at different rollout speeds.
  Of the {len(roi_df):,} listings in <strong>{city_str}</strong> ({type_str}),
  <strong>{n_eligible:,}</strong> stand to gain revenue from repricing. At a
  <strong>{adoption_rate}% adoption rate</strong>, roughly <strong>{n_adopters:,} listings</strong>
  would reprice, generating an estimated <strong>${projected_lift/1e3:.0f}K</strong> in additional
  annual revenue — an average of <strong>${avg_lift_per_listing:,.0f} per listing per year</strong>.
  Use the slider to stress-test the business case at conservative vs. optimistic rollout assumptions.
</div>
""", unsafe_allow_html=True)

r1, r2, r3, r4 = st.columns(4)
with r1:
    st.metric("Listings in scope", f"{len(roi_df):,}")
with r2:
    st.metric("Projected adopters", f"{n_adopters:,}")
with r3:
    st.metric("Portfolio revenue lift",
              f"${projected_lift/1e3:.0f}K" if projected_lift < 1e6 else f"${projected_lift/1e6:.2f}M")
with r4:
    st.metric("Avg lift per listing",
              f"${avg_lift_per_listing:,.0f}/yr" if not np.isnan(avg_lift_per_listing) else "—")

if len(roi_df) > 0:
    lift_by_seg = roi_df.groupby("segment", observed=True)["annual_revenue_lift"].sum().reindex(
        ["Budget", "Mid-Market", "Premium"]
    )

    fig_roi = go.Figure(go.Bar(
        x=lift_by_seg.index,
        y=lift_by_seg.values / 1e3,
        marker_color=["#6ea8d8", "#4a90c4", "#1d5f9e"],
        text=[f"${v/1e3:.0f}K" if abs(v) < 1e6 else f"${v/1e6:.2f}M" for v in lift_by_seg.values],
        textposition="outside",
        textfont=dict(size=13, color="#212529"),
    ))
    fig_roi.update_layout(
        height=260,
        margin=dict(l=0, r=0, t=20, b=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=FONT,
        yaxis=dict(**AXIS_STYLE, title="Annual revenue lift ($K)"),
        xaxis=dict(showgrid=False, tickfont=dict(size=13, color="#212529")),
        showlegend=False,
    )
    st.plotly_chart(fig_roi, use_container_width=True)

# ── Listing-level detail ───────────────────────────────────────────────────
st.markdown('<div class="section-header">Listing-level detail</div>', unsafe_allow_html=True)

top_lift = df["annual_revenue_lift"].quantile(0.9)

st.markdown(f"""
<div class="explanation-box">
  <strong>How to read this table:</strong> Each row is one listing. Sort by <em>Revenue lift</em>
  to surface the highest-priority repricing opportunities — these are the listings where the gap
  between current and recommended price is large <em>and</em> demand is strong enough to support
  the change without hurting occupancy. The <em>Demand score</em> bar shows how competitive
  that neighborhood is (out of 100). <em>DOM</em> (days on market) is one of the clearest
  real-world signals of overpricing: listings sitting vacant for 30+ days are almost always
  asking more than the market will pay. The top 10% of listings by opportunity each stand to
  gain more than <strong>${top_lift:,.0f}/year</strong> from a price correction.
</div>
""", unsafe_allow_html=True)

sort_col = st.selectbox(
    "Sort by",
    options=["annual_revenue_lift", "price_gap_pct", "demand_score", "days_on_market", "current_price"],
    format_func=lambda x: {
        "annual_revenue_lift": "Revenue lift (highest first)",
        "price_gap_pct": "Price gap %",
        "demand_score": "Demand score",
        "days_on_market": "Days on market",
        "current_price": "Current price",
    }[x],
)

display_df = df.sort_values(sort_col, ascending=False).head(200)[[
    "city", "property_type", "segment", "sqft",
    "current_price", "recommended_price", "price_gap_pct",
    "demand_score", "occupancy_rate", "days_on_market", "annual_revenue_lift",
]].rename(columns={
    "city": "City",
    "property_type": "Type",
    "segment": "Segment",
    "sqft": "Sq ft",
    "current_price": "Current ($)",
    "recommended_price": "Recommended ($)",
    "price_gap_pct": "Gap (%)",
    "demand_score": "Demand",
    "occupancy_rate": "Occupancy",
    "days_on_market": "DOM",
    "annual_revenue_lift": "Annual lift ($)",
})

display_df["Occupancy"] = (display_df["Occupancy"] * 100).round(1).astype(str) + "%"

st.dataframe(
    display_df,
    use_container_width=True,
    height=320,
    column_config={
        "Current ($)": st.column_config.NumberColumn(format="$%d"),
        "Recommended ($)": st.column_config.NumberColumn(format="$%d"),
        "Gap (%)": st.column_config.NumberColumn(format="%.1f%%"),
        "Annual lift ($)": st.column_config.NumberColumn(format="$%d"),
        "Demand": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
    },
    hide_index=True,
)

st.caption(
    "Pricing recommendations generated via demand-weighted regression on comparable transactions, "
    "demand score, and distance-to-center. Revenue lift assumes partial occupancy adjustment. "
    "Synthetic dataset — 2,000 listings across 5 U.S. markets."
)
