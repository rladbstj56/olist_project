"""Inference contract for experimental order classifiers; no training or UI side effects."""
import pandas as pd


def predict_order_model(bundle, items):
    """Bundle and all items per order -> indexed positive/risk scores and strict risk flag.

    Reuse fitted aggregation and encoding; count rows rather than item sequence values.
    Callers must supply every item in each requested order.
    """
    features = bundle['transformer'].transform(items)
    if bundle['include_item_count']:
        features['item_count'] = items.groupby('order_id').size().reindex(features.index)
    if list(features.columns) != bundle['feature_columns']:
        raise ValueError('Order feature schema differs from the saved model')
    model = bundle['model']
    if list(model.classes_) != [0, 1]:
        raise ValueError('Expected risk=0 and positive=1 classes')
    positive = model.predict_proba(features)[:, 1]
    return pd.DataFrame({'positive_probability': positive, 'risk_score': 1-positive,
                         'risk_flag': positive < bundle['positive_threshold']}, index=features.index)
