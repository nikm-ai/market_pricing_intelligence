import numpy as np
import pandas as pd

def generate_marketplace_data(n=2000, seed=42):
    rng = np.random.default_rng(seed)

    cities = ["Austin", "Chicago", "Denver", "Seattle", "Nashville"]
    city_base_price = {"Austin": 1850, "Chicago": 2200, "Denver": 2050, "Seattle": 2600, "Nashville": 1650}
    city_demand_mult = {"Austin": 1.15, "Chicago": 1.05, "Denver": 1.10, "Seattle": 1.25, "Nashville": 1.08}

    property_types = ["Studio", "1BR", "2BR", "3BR+"]
    type_size_map = {"Studio": 480, "1BR": 720, "2BR": 1050, "3BR+": 1450}
    type_price_mult = {"Studio": 0.65, "1BR": 1.0, "2BR": 1.45, "3BR+": 2.0}

    segments = ["Budget", "Mid-Market", "Premium"]

    city = rng.choice(cities, n)
    prop_type = rng.choice(property_types, n, p=[0.15, 0.40, 0.30, 0.15])

    sqft = np.array([type_size_map[t] for t in prop_type]) + rng.integers(-80, 80, n)
    base = np.array([city_base_price[c] for c in city])
    type_mult = np.array([type_price_mult[t] for t in prop_type])
    demand_mult = np.array([city_demand_mult[c] for c in city])

    # Comparable transaction price (market anchor)
    comp_price = base * type_mult * (1 + rng.normal(0, 0.08, n))

    # Neighborhood demand score 0-100
    demand_score = np.clip(
        50 * demand_mult + rng.normal(0, 12, n), 10, 98
    )

    # Distance to city center (miles)
    distance_km = np.abs(rng.normal(4, 2.5, n))

    # Current listed price (some over/under-priced relative to comps)
    pricing_error = rng.normal(0, 0.12, n)
    current_price = comp_price * (1 + pricing_error)

    # Days on market (higher = harder to rent = overpriced or low demand)
    days_on_market = np.clip(
        30 - 0.15 * demand_score + 60 * np.abs(pricing_error) + rng.exponential(8, n),
        1, 120
    ).astype(int)

    # Occupancy rate %
    occupancy = np.clip(
        0.95 - 0.003 * days_on_market + 0.002 * demand_score + rng.normal(0, 0.05, n),
        0.40, 0.99
    )

    # Segment assignment
    price_pct = pd.qcut(current_price, q=3, labels=["Budget", "Mid-Market", "Premium"])

    # Model-recommended price (what pricing model would suggest)
    recommended_price = comp_price * (1 + 0.04 * (demand_score - 50) / 50) * (1 - 0.015 * distance_km)

    # Estimated revenue lift from adopting recommended price
    price_gap = (recommended_price - current_price) / current_price
    occupancy_lift = np.where(price_gap < 0, -price_gap * 0.4, price_gap * 0.15)
    revenue_current = current_price * occupancy * 12
    revenue_recommended = recommended_price * np.clip(occupancy + occupancy_lift, 0, 0.99) * 12
    annual_lift = revenue_recommended - revenue_current

    df = pd.DataFrame({
        "city": city,
        "property_type": prop_type,
        "segment": price_pct,
        "sqft": sqft,
        "current_price": current_price.round(0).astype(int),
        "comp_price": comp_price.round(0).astype(int),
        "recommended_price": recommended_price.round(0).astype(int),
        "demand_score": demand_score.round(1),
        "distance_miles": distance_km.round(1),
        "days_on_market": days_on_market,
        "occupancy_rate": occupancy.round(3),
        "annual_revenue_current": revenue_current.round(0).astype(int),
        "annual_revenue_recommended": revenue_recommended.round(0).astype(int),
        "annual_revenue_lift": annual_lift.round(0).astype(int),
        "price_gap_pct": (price_gap * 100).round(1),
    })

    return df


if __name__ == "__main__":
    df = generate_marketplace_data()
    df.to_csv("data/listings.csv", index=False)
    print(f"Generated {len(df)} listings")
    print(df.describe())
