"""One predeclared Track B temporal fit; keep production and historical outputs intact."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import platform
import subprocess
import sys

import numpy as np
import pandas as pd
import sklearn
import lightgbm
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.olist_delivery_models import (
    PRE_ORDER_COLS, RAW_INPUT_COLS, prepare_corrected_model_frame,
    build_corrected_track_pipeline, select_positive_threshold,
    risk_classification_metrics,
)

KEYS = ['order_id', 'order_item_id']
DESIGN = ROOT / 'outputs/experiments/temporal-design-20261003-005953'
PERIODS = [('train', None, '2018-01-01', '2018-01-01'),
           ('valid', '2018-01-01', '2018-04-01', '2018-04-01'),
           ('test', '2018-04-01', '2018-08-01', '2018-11-01')]


def evaluate(frame, threshold):
    """Prediction rows + fixed positive threshold -> metrics, counts and risk confusion cells."""
    result = risk_classification_metrics(frame.review_label, frame.positive_probability.to_numpy(), threshold)
    truth = frame.review_label.eq(0)
    flagged = frame.positive_probability.lt(threshold)
    tn, fp, fn, tp = confusion_matrix(truth, flagged, labels=[False, True]).ravel()
    result.update(n=len(frame), orders=frame.order_id.nunique(), risk_n=int(truth.sum()),
                  risk_rate=float(truth.mean()), flagged_n=int(flagged.sum()),
                  tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    if truth.nunique() < 2:
        result['risk_pr_auc'] = None
    return result


def order_predictions(frame, aggregation):
    """Item predictions + max/mean risk rule -> one row per order, retaining item count."""
    assert frame.groupby('order_id').review_label.nunique().eq(1).all()
    return frame.groupby('order_id', sort=False).agg(
        review_label=('review_label', 'first'),
        positive_probability=('positive_probability', 'min' if aggregation == 'max' else 'mean'),
        item_count=('order_item_id', 'size'),
        purchase_month=('purchase_month', 'first'),
    ).reset_index()


def main():
    """Frozen CSVs/design -> one train-only fit and a new directory of auditable predictions/results."""
    design_summary = json.loads((DESIGN / 'summary.json').read_text())
    input_hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                    for p in design_summary['sha256']}
    assert input_hashes == design_summary['sha256'], 'Frozen inputs or design code changed'
    raw = pd.read_csv(ROOT / 'data/processed/merged_data.csv')
    model = prepare_corrected_model_frame(pd.read_csv(ROOT / 'data/processed/ml_data.csv'), raw_data=raw)
    dates = ['order_purchase_timestamp', 'order_delivered_customer_date', 'review_answer_timestamp']
    metadata = model[KEYS].merge(raw[KEYS + dates], on=KEYS, how='left', validate='one_to_one')
    assert np.array_equal(metadata[KEYS].to_numpy(), model[KEYS].to_numpy())
    metadata.index = model.index
    for col in dates:
        metadata[col] = pd.to_datetime(metadata[col])
    assert not metadata[dates].isna().any().any()
    metadata['available_proxy'] = metadata[dates[1:]].max(axis=1)
    metadata['invalid_time'] = metadata.review_answer_timestamp.lt(metadata.order_purchase_timestamp)
    metadata['proposed_split'] = 'outside_proposed_period'
    metadata['eligible'] = False
    for name, start, end, freeze in PERIODS:
        mask = metadata.order_purchase_timestamp.lt(pd.Timestamp(end))
        if start:
            mask &= metadata.order_purchase_timestamp.ge(pd.Timestamp(start))
        metadata.loc[mask, 'proposed_split'] = name
        metadata.loc[mask & ~metadata.invalid_time & metadata.available_proxy.lt(pd.Timestamp(freeze)), 'eligible'] = True
    designed = pd.read_csv(DESIGN / 'proposed_assignments.csv')
    check = metadata.merge(designed, on=KEYS, validate='one_to_one', suffixes=('', '_design'))
    assert len(check) == len(model) == len(designed)
    for col in ['proposed_split', 'eligible', 'invalid_time']:
        assert check[col].eq(check[col + '_design']).all(), col
    assert check.available_proxy.eq(pd.to_datetime(check.available_proxy_design)).all()
    split_masks = {name: metadata.proposed_split.eq(name) & metadata.eligible for name, *_ in PERIODS}
    for expected in design_summary['periods']:
        part = model.loc[split_masks[expected['split']]]
        assert len(part) == expected['eligible_item_rows']
        assert part.order_id.nunique() == expected['eligible_orders']
    order_sets = {name: set(model.loc[mask, 'order_id']) for name, mask in split_masks.items()}
    assert all(not order_sets[a] & order_sets[b] for a, b in [('train', 'valid'), ('train', 'test'), ('valid', 'test')])
    out = ROOT / 'outputs/experiments' / datetime.now().strftime('temporal-validation-%Y%m%d-%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    metadata.to_csv(out / 'assignments.csv', index_label='row_id')
    profile = metadata.copy()
    profile['risk'] = model.review_label.eq(0)
    profile['delivery_days'] = (profile.order_delivered_customer_date - profile.order_purchase_timestamp).dt.total_seconds() / 86400
    profile['review_days'] = (profile.review_answer_timestamp - profile.order_purchase_timestamp).dt.total_seconds() / 86400
    profile['status'] = np.select([profile.invalid_time, profile.eligible], ['invalid_time', 'eligible'], default='pending_or_outside')
    profile.drop_duplicates('order_id').groupby(['proposed_split', 'status']).agg(
        orders=('order_id', 'size'), risk_rate=('risk', 'mean'),
        median_delivery_days=('delivery_days', 'median'), median_review_days=('review_days', 'median'),
    ).to_csv(out / 'cohort_profile.csv')
    X = model[PRE_ORDER_COLS + RAW_INPUT_COLS]
    train = split_masks['train']
    pipeline = build_corrected_track_pipeline(X.loc[train], 'B')
    print(f'Fitting once: {int(train.sum())} train rows -> {out}', flush=True)
    pipeline.fit(X.loc[train], model.loc[train, 'review_label'])
    assert list(pipeline.named_steps['classifier'].classes_) == [0, 1]
    imputer = pipeline.named_steps['train_only_imputation']
    statistics = {'coordinate_modes': imputer.coordinate_modes_, 'product_medians': imputer.product_medians_,
                  'distance_median': imputer.distance_median_}
    (out / 'imputation_statistics.json').write_text(json.dumps(statistics, ensure_ascii=False, indent=2) + '\n')
    predictions = []
    for name in ['valid', 'test']:
        mask = split_masks[name]
        part = model.loc[mask, KEYS + ['review_label']].copy()
        part['row_id'] = part.index
        part['split'] = name
        part['purchase_month'] = metadata.loc[mask, 'order_purchase_timestamp'].dt.strftime('%Y-%m')
        part['positive_probability'] = pipeline.predict_proba(X.loc[mask])[:, 1]
        predictions.append(part)
    predictions = pd.concat(predictions)
    valid = predictions[predictions.split.eq('valid')]
    threshold = select_positive_threshold(valid.review_label, valid.positive_probability.to_numpy())
    grid = pd.DataFrame([risk_classification_metrics(valid.review_label, valid.positive_probability.to_numpy(), float(t))
                         for t in np.arange(.10, .91, .01)])
    grid['gap'] = (grid.risk_recall - grid.balanced_acc).abs()
    grid['selected'] = grid.positive_threshold.eq(threshold)
    assert grid.loc[grid.gap.idxmin(), 'positive_threshold'] == threshold
    grid.to_csv(out / 'validation_thresholds.csv', index=False)
    predictions['positive_threshold'] = threshold
    predictions.to_csv(out / 'predictions.csv', index=False)
    metrics, monthly, sizes, order_outputs = [], [], [], []
    for name in ['valid', 'test']:
        part = predictions[predictions.split.eq(name)]
        for unit, evaluated in [('item', part), ('order_max', order_predictions(part, 'max')),
                                ('order_mean', order_predictions(part, 'mean'))]:
            metrics.append(dict(split=name, unit=unit, **evaluate(evaluated, threshold)))
            for month, group in evaluated.groupby('purchase_month'):
                monthly.append(dict(split=name, unit=unit, month=month, **evaluate(group, threshold)))
            if unit != 'item':
                evaluated = evaluated.assign(split=name, aggregation=unit)
                order_outputs.append(evaluated)
                evaluated['item_count_group'] = np.select([evaluated.item_count.eq(1), evaluated.item_count.eq(2)], ['1', '2'], default='3+')
                for size, group in evaluated.groupby('item_count_group'):
                    sizes.append(dict(split=name, unit=unit, item_count_group=size, **evaluate(group, threshold)))
    pd.DataFrame(metrics).to_csv(out / 'metrics.csv', index=False)
    pd.DataFrame(monthly).to_csv(out / 'monthly_metrics.csv', index=False)
    pd.DataFrame(sizes).to_csv(out / 'item_count_metrics.csv', index=False)
    pd.concat(order_outputs).to_csv(out / 'order_predictions.csv', index=False)
    # Round-trip the exact stored scores before treating the reported metrics as reproducible.
    saved = pd.read_csv(out / 'predictions.csv', float_precision='round_trip')
    for result in metrics:
        part = saved[saved.split.eq(result['split'])]
        if result['unit'] != 'item':
            part = order_predictions(part, result['unit'].split('_')[1])
        recomputed = evaluate(part, threshold)
        assert all(np.isclose(recomputed[k], v, rtol=0, atol=1e-12) for k, v in result.items() if k not in ['split', 'unit'])
    assert input_hashes == {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in input_hashes}
    manifest = {'status': 'complete', 'fit_count': 1, 'track': 'B', 'periods': design_summary['periods'],
                'positive_threshold': threshold, 'risk_threshold': 1-threshold,
                'threshold_selection': 'validation item rows only; existing minimum abs(risk_recall-balanced_acc) rule',
                'order_rule': 'max risk = min positive probability; mean risk is sensitivity only',
                'feature_columns': PRE_ORDER_COLS, 'classifier_params': pipeline.named_steps['classifier'].get_params(),
                'input_sha256': input_hashes, 'design_assignment_sha256': hashlib.sha256((DESIGN / 'proposed_assignments.csv').read_bytes()).hexdigest(),
                'code_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in ['scripts/audit_temporal_model.py', 'src/olist_delivery_models.py', 'src/olist_imputation.py']},
                'head_before_run': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'lightgbm': lightgbm.__version__},
                'checks': ['design assignments identical', 'order intersections zero', 'eligible counts identical', 'saved-score metric roundtrip', 'input hashes unchanged'],
                'limitations': ['final completed/review-non3 cohort is retrospective', 'historical feature availability unverified', 'availability uses recorded timestamp proxy', 'no untouched new dataset', 'no model binary saved; scores and fitted imputation statistics saved'],
                'output_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())}}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)
    print(out, flush=True)


if __name__ == '__main__':
    main()
