"""
lead_time_forecaster.py

Forecasting hardware lead times as a time-series problem.

Observation: if you treat historical PO data as a time series,
you can often predict whether lead times are trending up or down
before the vendor tells you.

During the 2024 GPU crunch, lead times expanded gradually for
about 6 weeks before vendors started quoting the longer times.
If you catch that early, you order before the queue gets long.

Approach: per-vendor, per-hardware-category regression on rolling
features (recent lead times, order volume, macro signals if available).

Still very early  -  only have synthetic data to test on.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class LeadTimeFeatures:
    """Features for the lead time forecasting model."""
    # rolling stats on recent lead times for this vendor/hw combo
    rolling_mean_30d: float
    rolling_std_30d: float
    rolling_mean_90d: float
    # order volume signals
    orders_last_30d: int
    orders_last_90d: int
    # trend
    lead_time_delta_30d: float   # how much has lead time changed in 30d
    # time features
    month: int
    quarter: int
    # vendor-specific baseline (encoded as numeric  -  not ideal but fine for now)
    vendor_id: int


class LeadTimeForecaster:
    """
    Predicts expected lead time for a future hardware order.

    Input: historical PO data (from supply chain generator or real DCIM)
    Output: predicted lead time in days, with uncertainty estimate
    """

    def __init__(self, horizon_days: int = 30):
        self.horizon_days = horizon_days
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        self._vendor_map: dict[str, int] = {}
        self._fitted = False

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build features from PO history DataFrame.

        Expects columns: order_date, vendor, hardware_type,
                         actual_lead_days, quantity
        """
        df = df.copy()
        df["order_date"] = pd.to_datetime(df["order_date"])
        df = df.sort_values("order_date")

        # encode vendors
        vendors = df["vendor"].unique()
        self._vendor_map = {v: i for i, v in enumerate(vendors)}
        df["vendor_id"] = df["vendor"].map(self._vendor_map).fillna(-1).astype(int)

        feature_rows = []
        for _, grp in df.groupby(["vendor", "hardware_type"]):
            grp = grp.sort_values("order_date").copy()
            grp["rolling_mean_30d"] = grp["actual_lead_days"].rolling(window=5, min_periods=1).mean()
            grp["rolling_std_30d"] = grp["actual_lead_days"].rolling(window=5, min_periods=1).std().fillna(0)
            grp["rolling_mean_90d"] = grp["actual_lead_days"].rolling(window=15, min_periods=1).mean()
            grp["lead_time_delta_30d"] = grp["actual_lead_days"].diff(periods=3).fillna(0)
            grp["orders_last_30d"] = grp["order_date"].apply(
                lambda d: ((grp["order_date"] >= d - pd.Timedelta(days=30)) & (grp["order_date"] < d)).sum()
            )
            grp["orders_last_90d"] = grp["order_date"].apply(
                lambda d: ((grp["order_date"] >= d - pd.Timedelta(days=90)) & (grp["order_date"] < d)).sum()
            )
            grp["month"] = grp["order_date"].dt.month
            grp["quarter"] = grp["order_date"].dt.quarter
            feature_rows.append(grp)

        return pd.concat(feature_rows).dropna(subset=["actual_lead_days"])

    FEATURE_COLS = [
        "rolling_mean_30d", "rolling_std_30d", "rolling_mean_90d",
        "orders_last_30d", "orders_last_90d", "lead_time_delta_30d",
        "month", "quarter", "vendor_id",
    ]

    def fit(self, po_df: pd.DataFrame):
        features_df = self._build_features(po_df)
        # only train on delivered orders (have actual lead times)
        features_df = features_df[features_df["actual_lead_days"].notna()]

        X = features_df[self.FEATURE_COLS].values
        y = features_df["actual_lead_days"].values

        # time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=4)
        cv_maes = []
        for train_idx, val_idx in tscv.split(X):
            self.model.fit(X[train_idx], y[train_idx])
            preds = self.model.predict(X[val_idx])
            cv_maes.append(mean_absolute_error(y[val_idx], preds))

        logger.info(f"CV MAE: {np.mean(cv_maes):.1f} ± {np.std(cv_maes):.1f} days")

        # refit on full data
        self.model.fit(X, y)
        self._fitted = True

        # feature importance
        importances = sorted(
            zip(self.FEATURE_COLS, self.model.feature_importances_),
            key=lambda x: -x[1]
        )
        logger.info("Feature importances: " + ", ".join(f"{k}={v:.3f}" for k, v in importances[:5]))

    def predict(self, vendor: str, hardware_type: str, recent_history: pd.DataFrame) -> dict:
        if not self._fitted:
            raise RuntimeError("Call fit() first")

        features_df = self._build_features(recent_history)
        last_row = features_df[(features_df["vendor"] == vendor)].tail(1)

        if last_row.empty:
            # no history for this vendor  -  return naive estimate
            return {"predicted_lead_days": 60, "confidence": "low", "note": "no vendor history"}

        X = last_row[self.FEATURE_COLS].values
        pred = float(self.model.predict(X)[0])

        return {
            "vendor": vendor,
            "hardware_type": hardware_type,
            "predicted_lead_days": round(pred, 1),
            "confidence": "medium",  # TODO: proper uncertainty from quantile regression
        }
