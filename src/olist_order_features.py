"""Order-level inputs from all items, with train-only statistics and no item-sequence feature."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .olist_delivery_models import prepare_corrected_model_frame
from .olist_imputation import KEYS, COORDS, SIZES, RAW_INPUT_COLS, TrainOnlyImputer

COMMON = {
    'order_purchase_dayofweek': 'purchase_dayofweek',
    'order_purchase_month': 'purchase_month',
    'expected_delivery_days': 'expected_delivery_days',
    'is_sp_customer': 'is_sp_customer',
}
PHYSICAL = dict(zip(SIZES, ['weight', 'length', 'height', 'width']))
ITEM_FEATURES = [*COMMON, 'price', 'freight_value', *SIZES,
                 'distance_km', 'cross_state', 'is_sp_seller', 'main_category']
SOURCE_COMMON = ['source__customer_state', 'source__purchase_timestamp', 'source__expected_timestamp']


def _check_keys(items):
    """Item table -> reject missing or repeated order/item keys before any grouping."""
    if items.empty or items[KEYS].isna().any().any() or items.duplicated(KEYS).any():
        raise ValueError('Nonempty, nonmissing, unique order/item keys are required.')
    if not items.index.is_unique:
        raise ValueError('Item row index must be unique.')


def prepare_order_items(data, raw_data):
    """Existing model CSV + unimputed source -> key-aligned input/label table without fitting statistics."""
    frame = prepare_corrected_model_frame(data, raw_data=raw_data)
    source = raw_data[KEYS + ['product_id', 'seller_id', 'customer_state',
                             'order_purchase_timestamp', 'order_estimated_delivery_date']].rename(columns={
        'customer_state': SOURCE_COMMON[0], 'order_purchase_timestamp': SOURCE_COMMON[1],
        'order_estimated_delivery_date': SOURCE_COMMON[2],
    })
    joined = frame[KEYS].merge(source, on=KEYS, how='left', validate='one_to_one', indicator=True)
    if not joined['_merge'].eq('both').all() or not np.array_equal(joined[KEYS], frame[KEYS]):
        raise ValueError('Every item must match its source without reordering.')
    out = frame[KEYS + ITEM_FEATURES + RAW_INPUT_COLS + ['review_score', 'review_label']].copy()
    for col in ['product_id', 'seller_id', *SOURCE_COMMON]:
        out[col] = joined[col].to_numpy()
    _check_keys(out)
    return out


def make_order_labels(items):
    """Consistent 1/2/4/5 review scores -> binary Series indexed by unique sorted order_id."""
    _check_keys(items)
    if not items.review_score.isin([1, 2, 4, 5]).all():
        raise ValueError('Order labels require the frozen review-3-excluded policy.')
    groups = items.groupby('order_id', sort=True)
    if groups.review_score.nunique(dropna=False).gt(1).any():
        raise ValueError('Review scores disagree within an order.')
    expected = items.review_score.ge(4).astype(int)
    if 'review_label' in items and not items.review_label.eq(expected).all():
        raise ValueError('Stored review label disagrees with review score.')
    return groups.review_score.min().ge(4).astype(int).rename('review_label')


class OrderFeatureTransformer(TransformerMixin, BaseEstimator):
    """Reduce item rows to order rows; align classifier labels by the returned order_id index."""

    def _validate_items(self, items):
        """Item table -> enforce required identifiers, shared fields and aggregatable money/flags."""
        _check_keys(items)
        shared = [*COMMON, *SOURCE_COMMON]
        if items[shared + ['product_id', 'seller_id']].isna().any().any():
            raise ValueError('Order context and product/seller identifiers cannot be missing.')
        if items.groupby('order_id')[shared].nunique(dropna=False).gt(1).any().any():
            raise ValueError('Shared order attributes disagree within an order.')
        money = items[['price', 'freight_value']]
        if not np.isfinite(money.to_numpy(dtype=float)).all() or money.lt(0).any().any():
            raise ValueError('Item price/freight must be finite and nonnegative.')
        if not items[['cross_state', 'is_sp_seller', 'is_sp_customer']].isin([0, 1]).all().all():
            raise ValueError('Region indicators must be binary.')

    def fit(self, X, y=None):
        """Training items -> fitted item imputer, category vocabulary and order-level fallback medians."""
        self._validate_items(X)
        self.item_imputer_ = TrainOnlyImputer(tuple(ITEM_FEATURES)).fit(X)
        self.categories_ = sorted(X.main_category.dropna().unique().tolist())
        self.category_columns_ = {value: f'category_share_{i:02d}' for i, value in enumerate(self.categories_)}
        aggregated = self._aggregate(X)
        self.feature_names_out_ = np.asarray(aggregated.columns, dtype=object)
        self.order_medians_ = aggregated.select_dtypes(include=[np.number]).median()
        if not np.isfinite(self.order_medians_.to_numpy()).all():
            raise ValueError('A feature has no finite training-order median; review the feature contract.')
        return self

    def _aggregate(self, items):
        """Validated items + fitted item statistics -> deterministic aggregates, retaining shared categorical types before final fill."""
        values = self.item_imputer_.transform(items)
        values['order_id'] = items.order_id
        values['product_id'] = items.product_id
        values['seller_id'] = items.seller_id
        for raw_field, name in PHYSICAL.items():
            values[f'missing_{name}_share'] = items['raw__' + raw_field].isna().astype(float)
        values['missing_distance_share'] = items[['raw__' + c for c in COORDS]].isna().any(axis=1).astype(float)
        # Canonical numeric order keeps floating-point reductions independent of row order and identifiers.
        numeric = ['price', 'freight_value', *SIZES, 'distance_km']
        values = values.sort_values(['order_id', *numeric], kind='stable')
        groups = values.groupby('order_id', sort=True)
        out = groups[list(COMMON)].min().rename(columns=COMMON)
        out['distinct_product_count'] = groups.product_id.nunique()
        out['distinct_seller_count'] = groups.seller_id.nunique()
        for source, prefix in [('price', 'price'), ('freight_value', 'freight')]:
            for op in ['sum', 'mean', 'max']:
                out[f'{prefix}_{op}'] = groups[source].agg(op)
            if source == 'price':
                out['price_std'] = groups[source].std(ddof=0)
        out['freight_to_price_ratio'] = out.freight_sum / out.price_sum.replace(0, np.nan)
        for source, prefix in PHYSICAL.items():
            out[f'{prefix}_mean'] = groups[source].mean()
            out[f'{prefix}_max'] = groups[source].max()
        for op in ['mean', 'max']:
            out[f'distance_{op}'] = groups.distance_km.agg(op)
        out['distance_std'] = groups.distance_km.std(ddof=0)
        out['cross_state_share'] = groups.cross_state.mean()
        out['seller_sp_share'] = groups.is_sp_seller.mean()
        for category, col in self.category_columns_.items():
            out[col] = values.main_category.eq(category).fillna(False).groupby(values.order_id).mean()
        out['category_share_unknown'] = (~values.main_category.isin(self.categories_)).groupby(values.order_id).mean()
        for name in [*[f'missing_{v}_share' for v in PHYSICAL.values()], 'missing_distance_share']:
            out[name] = groups[name].mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def transform(self, X):
        """All items of each requested order -> features indexed by order_id using frozen statistics."""
        check_is_fitted(self, ['item_imputer_', 'categories_', 'order_medians_', 'feature_names_out_'])
        self._validate_items(X)
        out = self._aggregate(X)
        return out.loc[:, self.feature_names_out_].fillna(self.order_medians_)

    def get_feature_names_out(self, input_features=None):
        """Optional unused input names -> copy of the learned output column names."""
        check_is_fitted(self, 'feature_names_out_')
        return self.feature_names_out_.copy()
