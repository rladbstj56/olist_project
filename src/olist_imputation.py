"""Training-only imputation for A/B classification, including cross-validation folds."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

KEYS = ['order_id', 'order_item_id']
COORDS = ['customer_lat', 'customer_lng', 'seller_lat', 'seller_lng']
SIZES = ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']
RAW_FIELDS = ['customer_city', 'seller_city', *COORDS, *SIZES]
RAW_INPUT_COLS = ['raw__' + c for c in RAW_FIELDS]


def attach_raw_imputation_columns(frame: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """One-to-one order/item join; retain model row order/index and add prefixed raw fields."""
    if not frame.index.is_unique:
        raise ValueError('Model row index must be unique.')
    if frame.duplicated(KEYS).any() or raw.duplicated(KEYS).any():
        raise ValueError('Order/item keys must be unique in both inputs.')
    if set(RAW_INPUT_COLS) & set(frame.columns):
        raise ValueError('Raw imputation columns already attached.')
    joined = frame[KEYS].merge(raw[KEYS + RAW_FIELDS], on=KEYS, how='left', sort=False,
                              validate='one_to_one', indicator=True)
    if not joined['_merge'].eq('both').all():
        raise ValueError('Every model row must match an unimputed source row.')
    if not np.array_equal(joined[KEYS].to_numpy(), frame[KEYS].to_numpy()):
        raise ValueError('Raw join changed model row order.')
    out = frame.copy()
    for field in RAW_FIELDS:
        out['raw__' + field] = joined[field].to_numpy()
    return out


def haversine_distance(frame: pd.DataFrame) -> pd.Series:
    """Coordinate columns in degrees -> existing Haversine distance in km."""
    lat1, lon1, lat2, lon2 = map(np.radians, [frame[c] for c in COORDS])
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def distance_category(distances):
    """Distance Series in km -> existing right-closed distance categories."""
    return np.select(
        [distances <= 50, distances <= 250, distances <= 750, distances <= 1500],
        ['Urban/Last-Mile', 'Short-Haul', 'Mid-Haul', 'Long-Haul'], default='Continental')


class TrainOnlyImputer(TransformerMixin, BaseEstimator):
    """Estimate from fit rows only; emit fixed model features without raw helper columns."""

    def __init__(self, feature_cols):
        self.feature_cols = feature_cols

    def _fill_coordinates(self, X):
        raw = X[RAW_INPUT_COLS].rename(columns=dict(zip(RAW_INPUT_COLS, RAW_FIELDS))).copy()
        for field in COORDS:
            city = field.split('_')[0] + '_city'
            raw[field] = raw[field].fillna(raw[city].map(self.coordinate_modes_[field]))
        return raw

    def fit(self, X, y=None):
        """Augmented training rows -> fitted city modes and product/distance medians."""
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        self.coordinate_modes_ = {}
        for field in COORDS:
            city = 'raw__' + field.split('_')[0] + '_city'
            modes = X.groupby(city)['raw__' + field].agg(
                lambda values: values.mode().iloc[0] if not values.mode().empty else np.nan)
            self.coordinate_modes_[field] = modes.dropna().to_dict()
        self.product_medians_ = {c: float(X['raw__' + c].median()) for c in SIZES}
        self.distance_median_ = float(haversine_distance(self._fill_coordinates(X)).median())
        if not np.isfinite(self.distance_median_) or not all(np.isfinite(v) for v in self.product_medians_.values()):
            raise ValueError('Training rows must provide finite product and distance medians.')
        return self

    def transform(self, X):
        """Augmented rows -> fixed feature frame using only previously fitted statistics."""
        check_is_fitted(self, ['coordinate_modes_', 'product_medians_', 'distance_median_'])
        out = X[list(self.feature_cols)].copy()
        raw = self._fill_coordinates(X)
        for field in SIZES:
            out[field] = raw[field].fillna(self.product_medians_[field])
        distances = haversine_distance(raw).fillna(self.distance_median_)
        # Keep CSV roundoff identical to the audited comparison on unchanged distances.
        same = np.isclose(distances, out.distance_km, rtol=1e-12, atol=1e-9)
        distances.loc[same] = out.loc[same, 'distance_km']
        out['distance_km'] = distances
        out['distance_cat'] = distance_category(distances)
        if 'delivery_days' in out:
            if 'delivery_speed' in out:
                out['delivery_speed'] = distances / out.delivery_days.replace(0, np.nan)
            if 'day_per_km' in out:
                out['day_per_km'] = out.delivery_days / distances.replace(0, np.nan)
            if 'delivery_distance' in out:
                out['delivery_distance'] = out.delivery_days * distances
        out = out.replace([np.inf, -np.inf], np.nan)
        for field in self.feature_cols:
            out[field] = out[field].astype(X[field].dtype)
        return out

    def transform_prepared_features(self, X):
        """Manual pre-order features -> features using supplied distance, without raw coordinates."""
        check_is_fitted(self, ['product_medians_', 'distance_median_'])
        out = X[list(self.feature_cols)].copy().replace([np.inf, -np.inf], np.nan)
        for field in SIZES:
            out[field] = out[field].fillna(self.product_medians_[field])
        out['distance_km'] = out.distance_km.fillna(self.distance_median_)
        out['distance_cat'] = distance_category(out.distance_km)
        return out

    def get_feature_names_out(self, input_features=None):
        """Return model feature names, excluding raw helper columns."""
        check_is_fitted(self, 'distance_median_')
        return np.asarray(self.feature_cols, dtype=object)
