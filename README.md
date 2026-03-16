# Marketplace Pricing Intelligence

An interactive dashboard for identifying pricing inefficiencies across a rental marketplace and quantifying the revenue opportunity from model-recommended repricing.

**Live demo:** [link to your Streamlit deployment]

---

## What it does

This project simulates the kind of pricing analytics work I did for a regional marketplace client — building an optimization model on comparable transaction data and geographic features, then surfacing recommendations through a self-serve dashboard that lets non-technical stakeholders explore ROI estimates by segment.

The app covers a full analytics workflow:

1. **Data generation** — Synthetic dataset of 2,000 rental listings across 5 U.S. markets, with realistic price distributions, demand scores, occupancy rates, and days-on-market
2. **Pricing model** — Demand-weighted regression on comparable transaction prices, neighborhood demand score, and distance to city center to generate recommended prices per listing
3. **Segmentation** — Budget / Mid-Market / Premium segmentation with distribution comparisons (current vs. recommended)
4. **Opportunity map** — Scatter plot of demand score vs. price gap to identify under- and over-priced clusters
5. **Self-serve ROI explorer** — Filter by market and property type, adjust model adoption rate, and see projected portfolio revenue lift
6. **Listing-level detail** — Sortable table with full model outputs for each listing

---

## Technical approach

| Component | Details |
|---|---|
| Data | Synthetic (2,000 listings, 5 cities, 3 segments) |
| Pricing model | Demand-weighted regression on comps, demand score, distance |
| Revenue lift | Estimated via price gap × partial occupancy adjustment |
| Stack | Python, Streamlit, Plotly, pandas, NumPy |

---

## Running locally

```bash
git clone https://github.com/nikm-ai/marketplace-pricing
cd marketplace-pricing
pip install -r requirements.txt
streamlit run app.py
```

---

## Project context

Built as part of a portfolio of applied data science projects. The underlying scenario — identifying mispriced listings via comparable transaction data and surfacing segment-level recommendations through a self-serve dashboard — mirrors the kind of work described in my consulting experience at Manohar Analytics.
