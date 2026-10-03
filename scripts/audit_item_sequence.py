"""Diagnose saved temporal scores by item sequence; no fitting or new model predictions."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import subprocess
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.olist_delivery_models import PRE_ORDER_COLS, RAW_INPUT_COLS, prepare_corrected_model_frame
from src.olist_imputation import TrainOnlyImputer
RUN = ROOT / 'outputs/experiments/temporal-validation-20261003-202427'
KEYS = ['order_id', 'order_item_id']


def main():
    """Frozen inputs, stored scores/statistics -> sequence, paired-input and order-composition diagnostics."""
    manifest = json.loads((RUN / 'manifest.json').read_text())
    files = [ROOT / p for p in manifest['input_sha256']] + list(RUN.glob('*'))
    before = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    for name, digest in manifest['input_sha256'].items():
        assert before[name] == digest
    for name, digest in manifest['output_sha256'].items():
        assert hashlib.sha256((RUN / name).read_bytes()).hexdigest() == digest
    for name, digest in manifest['code_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    raw = pd.read_csv(ROOT / 'data/processed/merged_data.csv')
    model = prepare_corrected_model_frame(pd.read_csv(ROOT / 'data/processed/ml_data.csv'), raw_data=raw)
    stats = json.loads((RUN / 'imputation_statistics.json').read_text())
    imputer = TrainOnlyImputer(tuple(PRE_ORDER_COLS))
    imputer.coordinate_modes_ = stats['coordinate_modes']
    imputer.product_medians_ = stats['product_medians']
    imputer.distance_median_ = stats['distance_median']
    corrected = imputer.transform(model[PRE_ORDER_COLS + RAW_INPUT_COLS])
    f = model[KEYS + ['review_label']].join(corrected.drop(columns='order_item_id'))
    assignment = pd.read_csv(RUN / 'assignments.csv')
    f = f.merge(assignment[KEYS + ['proposed_split', 'eligible']], on=KEYS, validate='one_to_one')
    f = f[f.eligible].rename(columns={'proposed_split': 'split'})
    f = f.merge(raw[KEYS + ['product_id', 'seller_id']], on=KEYS, validate='one_to_one')
    f['item_count'] = f.groupby('order_id').order_id.transform('size')
    f['item_group'] = np.select([f.item_count.eq(1), f.item_count.eq(2)], ['1', '2'], default='3+')
    f['truth'] = f.review_label.eq(0)
    assert f.groupby('order_id').review_label.nunique().eq(1).all()
    assert f.groupby('order_id').order_item_id.min().eq(1).all()
    assert f.groupby('order_id').order_item_id.max().eq(f.groupby('order_id').size()).all()
    p = pd.read_csv(RUN / 'predictions.csv', float_precision='round_trip')
    x = f.merge(p[KEYS + ['split', 'review_label', 'positive_probability']], on=KEYS + ['split', 'review_label'], validate='one_to_one')
    assert len(x) == len(p)
    x['risk'] = 1 - x.positive_probability
    x['flag'] = x.positive_probability.lt(manifest['positive_threshold'])
    out = ROOT / 'outputs/experiments' / datetime.now().strftime('item-sequence-diagnostic-%Y%m%d-%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    x.groupby(['split', 'item_group', 'order_item_id']).agg(
        rows=('risk', 'size'), risk_mean=('risk', 'mean'), risk_median=('risk', 'median'),
        flagged_n=('flag', 'sum'), flagged_rate=('flag', 'mean'), truth_rate=('truth', 'mean'),
    ).to_csv(out / 'sequence_scores.csv')
    orders = f.groupby(['split', 'order_id']).agg(item_count=('item_count', 'first'), item_group=('item_group', 'first'),
        truth=('truth', 'first'), unique_products=('product_id', 'nunique'), unique_sellers=('seller_id', 'nunique'))
    orders['multi_seller'] = orders.unique_sellers.gt(1)
    orders['multi_product'] = orders.unique_products.gt(1)
    orders.groupby(['split', 'item_group']).agg(orders=('truth', 'size'), risk_rate=('truth', 'mean'),
        mean_items=('item_count', 'mean'), multi_seller_rate=('multi_seller', 'mean'),
        multi_product_rate=('multi_product', 'mean')).to_csv(out / 'order_composition.csv')
    weighting = []
    for name, group in f.groupby('split'):
        o = orders.loc[name]
        weighting.append(dict(split=name, rows=len(group), orders=len(o), risk_row_rate=group.truth.mean(),
            risk_order_rate=o.truth.mean(), multi_row_share=group.item_count.gt(1).mean(),
            multi_order_share=o.item_count.gt(1).mean(),
            risk_mean_items=o.loc[o.truth, 'item_count'].mean(), positive_mean_items=o.loc[~o.truth, 'item_count'].mean()))
    pd.DataFrame(weighting).to_csv(out / 'training_weight_profile.csv', index=False)
    others = [c for c in PRE_ORDER_COLS if c != 'order_item_id']
    reference = x[x.order_item_id.eq(1)][['order_id', *others, 'risk', 'flag']]
    pairs = x[x.order_item_id.gt(1)].merge(reference, on='order_id', validate='many_to_one', suffixes=('', '_first'))
    difference = pd.DataFrame({c: ~(pairs[c].eq(pairs[c + '_first']) | (pairs[c].isna() & pairs[c + '_first'].isna())) for c in others})
    pairs['different_other_features'] = difference.sum(axis=1)
    pairs['risk_delta'] = pairs.risk - pairs.risk_first
    pairs['later_only_flag'] = pairs.flag & ~pairs.flag_first
    pairs['same_other_21'] = pairs.different_other_features.eq(0)
    pairs[['split', *KEYS, 'item_count', 'order_item_id', 'risk_first', 'risk', 'risk_delta',
           'flag_first', 'flag', 'later_only_flag', 'different_other_features', 'same_other_21']].loc[:,lambda d:~d.columns.duplicated()].to_csv(out / 'paired_scores.csv', index=False)
    paired_summary = pairs.groupby(['split', 'same_other_21']).agg(pairs=('risk', 'size'), orders=('order_id', 'nunique'),
        mean_delta=('risk_delta', 'mean'), median_delta=('risk_delta', 'median'), min_delta=('risk_delta', 'min'),
        max_delta=('risk_delta', 'max'), first_flag_rate=('flag_first', 'mean'), later_flag_rate=('flag', 'mean'),
        later_only_pairs=('later_only_flag', 'sum'))
    paired_summary.to_csv(out / 'paired_summary.csv')
    feature_rows=[]
    for name, group in pairs.groupby('split'):
        for col in others:
            feature_rows.append(dict(split=name, feature=col, pairs=len(group), differing_pairs=int(difference.loc[group.index, col].sum())))
    pd.DataFrame(feature_rows).to_csv(out / 'within_order_feature_differences.csv', index=False)
    numeric = [c for c in PRE_ORDER_COLS if pd.api.types.is_numeric_dtype(f[c])]
    f.groupby(['split', 'item_group'])[numeric].median().to_csv(out / 'input_medians.csv')
    o = x.groupby(['split', 'order_id']).agg(item_count=('item_count', 'first'), truth=('truth', 'first'),
        max_flag=('flag', 'max'), mean_risk=('risk', 'mean'), min_risk=('risk', 'min'), max_risk=('risk', 'max'))
    first = x[x.order_item_id.eq(1)].set_index(['split','order_id'])
    o['first_flag'] = first.flag
    o['added_by_later'] = o.max_flag & ~o.first_flag
    o['score_spread'] = o.max_risk - o.min_risk
    o['group'] = np.where(o.item_count.eq(1), 'single', 'multi')
    o.groupby(['split','group']).agg(orders=('truth','size'), truth_rate=('truth','mean'), first_flag_n=('first_flag','sum'),
        max_flag_n=('max_flag','sum'), added_by_later=('added_by_later','sum'), mean_spread=('score_spread','mean')).to_csv(out / 'aggregation_effect.csv')
    assert before == {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    result = {'fit_count': 0, 'new_predictions': 0, 'threshold_changed': False,
        'method': 'saved scores; restored train-only imputation statistics; exact matching on other 21 pre-encoder inputs within order',
        'limitations': ['not proof of item sequence causing dissatisfaction', 'not estimate of retraining without sequence',
                        'same-input pairs are a selected subset and may repeat first item', 'historical encoder/model binary not saved'],
        'positive_threshold':manifest['positive_threshold'], 'input_sha256':before,
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'output_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())}}
    (out/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(out)
    print(paired_summary.to_string())
    print(pd.read_csv(out/'aggregation_effect.csv').to_string(index=False))
    print(pd.DataFrame(weighting).to_string(index=False))


if __name__ == '__main__':
    main()
