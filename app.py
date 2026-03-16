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
    .metric-label { font-size: 12px; color: #6c757d; font-weight: 500; margin-bottom: 4px; }
    .metric-value { font-size: 24px; font-weight: 600; color: #212529; }
    .metric-delta { font-size: 12px; margin-top: 2px; }
    .delta-pos { color: #198754; }
    .delta-neg { color: #dc3545; }
    .section-header {
        font-size: 14px;
        font-weight: 600;
        color: #495057;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin: 1.5rem 0 0.75rem;
        padding-bottom: 6px;
        border-bottom: 1px solid #dee2e6;
    }
    div[data-testid="stSidebar"] { background: #f8f9fa; }
    .stSelectbox label, .stMultiSelect label, .stSlider label { font-size: 13px; font-weight: 500; }
</style>
""", unsafe_allow_html=True)


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
        <div class="metric-delta delta-neg">{pct_overpriced:.0f}% overpriced · <span class="delta-pos">{pct_underpriced:.0f}% underpriced</span></div>
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

st.markdown('<div class="section-header">Pricing distribution by segment</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([3, 2])

with col_left:
    segment_order = ["Budget", "Mid-Market", "Premium"]
    seg_colors = {"Budget": "#6ea8d8", "Mid-Market": "#4a90c4", "Premium": "#1d5f9e"}

    fig_box = go.Figure()
    for seg in segment_order:
        seg_df = df[df["segment"] == seg]
        fig_box.add_trace(go.Box(
            y=seg_df["current_price"],
            name=f"{seg}<br>(current)",
            marker_color=seg_colors[seg],
            opacity=0.6,
            legendgroup=seg,
            boxmean=True,
        ))
        fig_box.add_trace(go.Box(
            y=seg_df["recommended_price"],
            name=f"{seg}<br>(recommended)",
            marker_color=seg_colors[seg],
            marker_symbol="diamond",
            legendgroup=seg,
            boxmean=True,
            opacity=0.45,
        ))

    fig_box.update_layout(
        height=340,
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis_title="Monthly rent ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=12),
        yaxis=dict(gridcolor="#f0f0f0"),
        xaxis=dict(showgrid=False),
        boxgroupgap=0.2,
        boxgap=0.15,
    )
    st.plotly_chart(fig_box, use_container_width=True)

with col_right:
    seg_summary = df.groupby("segment", observed=True).agg(
        listings=("current_price", "count"),
        median_current=("current_price", "median"),
        median_recommended=("recommended_price", "median"),
        total_lift=("annual_revenue_lift", "sum"),
        avg_occupancy=("occupancy_rate", "mean"),
    ).reindex(["Budget", "Mid-Market", "Premium"])

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
                <span style="font-weight: 600; font-size: 14px;">{seg}</span>
                <span style="font-size: 12px; color: #6c757d;">{int(row['listings']):,} listings</span>
            </div>
            <div style="margin-top: 6px; font-size: 13px; color: #495057;">
                ${row['median_current']:,.0f} → <strong>${row['median_recommended']:,.0f}</strong>
                <span class="{delta_class}"> ({delta_pct:+.1f}%)</span>
            </div>
            <div style="font-size: 12px; margin-top: 4px;">
                Annual lift: <span class="{delta_class}">{lift_str}</span> ·
                Occupancy: {row['avg_occupancy']*100:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="section-header">Demand score vs price gap — opportunity map</div>', unsafe_allow_html=True)

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
    fig_scatter.add_hline(y=5, line_dash="dash", line_color="#198754", line_width=1,
                          annotation_text="Underpriced threshold", annotation_position="top right",
                          annotation_font_size=11)
    fig_scatter.add_hline(y=-5, line_dash="dash", line_color="#dc3545", line_width=1,
                          annotation_text="Overpriced threshold", annotation_position="bottom right",
                          annotation_font_size=11)
    fig_scatter.add_hline(y=0, line_color="#adb5bd", line_width=0.75)

    fig_scatter.update_layout(
        height=360,
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=12),
        yaxis=dict(gridcolor="#f0f0f0", zeroline=False),
        xaxis=dict(gridcolor="#f0f0f0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
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
        height=360,
        barmode="group",
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="% of listings",
        yaxis_title="",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=12),
        xaxis=dict(gridcolor="#f0f0f0"),
        yaxis=dict(showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    st.plotly_chart(fig_city, use_container_width=True)

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
adopters = roi_df[roi_df["annual_revenue_lift"] > 0].sample(
    frac=adoption, random_state=99
) if len(roi_df[roi_df["annual_revenue_lift"] > 0]) > 0 else pd.DataFrame()

projected_lift = adopters["annual_revenue_lift"].sum() if len(adopters) > 0 else 0
avg_lift_per_listing = adopters["annual_revenue_lift"].mean() if len(adopters) > 0 else 0
n_adopters = len(adopters)

r1, r2, r3, r4 = st.columns(4)
with r1:
    st.metric("Listings in scope", f"{len(roi_df):,}")
with r2:
    st.metric("Projected adopters", f"{n_adopters:,}")
with r3:
    st.metric("Portfolio revenue lift", f"${projected_lift/1e3:.0f}K" if projected_lift < 1e6 else f"${projected_lift/1e6:.2f}M")
with r4:
    st.metric("Avg lift per listing", f"${avg_lift_per_listing:,.0f}/yr" if not np.isnan(avg_lift_per_listing) else "—")

if len(roi_df) > 0:
    lift_by_seg = roi_df.groupby("segment", observed=True)["annual_revenue_lift"].sum().reindex(["Budget", "Mid-Market", "Premium"])

    fig_roi = go.Figure(go.Bar(
        x=lift_by_seg.index,
        y=lift_by_seg.values / 1e3,
        marker_color=["#6ea8d8", "#4a90c4", "#1d5f9e"],
        text=[f"${v/1e3:.0f}K" if abs(v) < 1e6 else f"${v/1e6:.2f}M" for v in lift_by_seg.values],
        textposition="outside",
    ))
    fig_roi.update_layout(
        height=260,
        margin=dict(l=0, r=0, t=20, b=0),
        yaxis_title="Annual revenue lift ($K)",
        xaxis_title="",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=12),
        yaxis=dict(gridcolor="#f0f0f0", zeroline=True, zerolinecolor="#dee2e6"),
        xaxis=dict(showgrid=False),
        showlegend=False,
    )
    st.plotly_chart(fig_roi, use_container_width=True)

st.markdown('<div class="section-header">Listing-level detail</div>', unsafe_allow_html=True)

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
