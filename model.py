"""
model.py
Fits a GradientBoostingRegressor to predict market-clearing rent price
from observable listing features (sqft, demand score, distance, city,
property type). Comparable transaction price is held out as a validation
signal — not used as a feature — so the model learns from the underlying
market drivers rather than just echoing the comp.

Exports fit_model() which returns all artifacts needed for the
Model Performance tab.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import streamlit as st

# Features available at listing time (no oracle comp price)
NUM_FEATURES = ["sqft", "demand_score", "distance_miles"]
CAT_FEATURES = ["city", "property_type"]
ALL_FEATURES  = NUM_FEATURES + CAT_FEATURES

FEATURE_DISPLAY_NAMES = {
    "sqft":          "Square footage",
    "demand_score":  "Neighborhood demand score",
    "distance_miles":"Distance to city center (mi)",
    "city":          "City market",
    "property_type": "Property type",
}


def _build_pipeline(model):
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_FEATURES),
    ])
    return Pipeline([("pre", pre), ("model", model)])


@st.cache_resource
def fit_model(df: pd.DataFrame) -> dict:
    """
    Fit GradientBoostingRegressor to predict recommended_price.
    Evaluates with 5-fold CV and returns full artifact dict.
    """
    X = df[ALL_FEATURES]
    y = df["recommended_price"].values.astype(float)

    # ── GBM pipeline ───────────────────────────────────────────────────────
    gbm = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.06,
        min_samples_leaf=15,
        subsample=0.8,
        random_state=42,
    )
    gbm_pipe = _build_pipeline(gbm)

    # ── Ridge baseline ──────────────────────────────────────────────────────
    ridge_pipe = _build_pipeline(Ridge(alpha=10.0))

    # ── 5-fold CV ──────────────────────────────────────────────────────────
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_r2   = cross_val_score(gbm_pipe, X, y, cv=kf, scoring="r2")
    cv_mae  = -cross_val_score(gbm_pipe, X, y, cv=kf, scoring="neg_mean_absolute_error")
    cv_rmse = np.sqrt(-cross_val_score(gbm_pipe, X, y, cv=kf, scoring="neg_mean_squared_error"))

    ridge_cv_r2  = cross_val_score(ridge_pipe, X, y, cv=kf, scoring="r2")
    ridge_cv_mae = -cross_val_score(ridge_pipe, X, y, cv=kf, scoring="neg_mean_absolute_error")

    # ── Fit on full data for display charts ────────────────────────────────
    gbm_pipe.fit(X, y)
    y_pred     = gbm_pipe.predict(X)
    residuals  = y - y_pred

    # ── Feature importances from fitted GBM ────────────────────────────────
    fitted_gbm = gbm_pipe.named_steps["model"]
    raw_importances = fitted_gbm.feature_importances_

    # Column order after ColumnTransformer: num cols first, then cat cols
    ordered_features = NUM_FEATURES + CAT_FEATURES
    display_names    = [FEATURE_DISPLAY_NAMES[f] for f in ordered_features]

    imp_df = pd.DataFrame({
        "Feature":    display_names,
        "Importance": raw_importances,
    }).sort_values("Importance", ascending=True).reset_index(drop=True)

    # ── Build prediction dataframe ─────────────────────────────────────────
    df_pred = df.copy()
    df_pred["predicted_price"] = y_pred.round(0).astype(int)
    df_pred["residual"]        = residuals.round(0).astype(int)
    df_pred["abs_error"]       = np.abs(residuals).round(0).astype(int)
    df_pred["pct_error"]       = (np.abs(residuals) / y * 100).round(1)

    # ── Fold-level summary for CV chart ────────────────────────────────────
    cv_df = pd.DataFrame({
        "Fold":  [f"Fold {i+1}" for i in range(5)],
        "R²":    cv_r2.round(3),
        "MAE":   cv_mae.round(0),
        "RMSE":  cv_rmse.round(0),
    })

    return {
        "model_pipe":     gbm_pipe,

        # Raw arrays
        "y_true":         y,
        "y_pred":         y_pred,
        "residuals":      residuals,

        # Prediction dataframe
        "df_pred":        df_pred,

        # CV metrics
        "cv_r2":          cv_r2,
        "cv_mae":         cv_mae,
        "cv_rmse":        cv_rmse,
        "cv_df":          cv_df,

        # Summary scalars
        "r2_mean":        cv_r2.mean(),
        "r2_std":         cv_r2.std(),
        "mae_mean":       cv_mae.mean(),
        "mae_std":        cv_mae.std(),
        "rmse_mean":      cv_rmse.mean(),
        "rmse_std":       cv_rmse.std(),

        # In-sample (for predicted vs actual chart)
        "insample_r2":    r2_score(y, y_pred),
        "insample_mae":   mean_absolute_error(y, y_pred),

        # Baseline
        "baseline_r2_mean":  ridge_cv_r2.mean(),
        "baseline_r2_std":   ridge_cv_r2.std(),
        "baseline_mae_mean": ridge_cv_mae.mean(),

        # Feature importances
        "imp_df":         imp_df,
    }
