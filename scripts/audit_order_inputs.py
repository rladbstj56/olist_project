"""Validate and preserve the order-level input transformer; never train a classifier."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import platform
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.olist_order_features import prepare_order_items, make_order_labels, OrderFeatureTransformer

BASE = ROOT/'outputs/experiments/temporal-validation-20261003-202427'
KEYS = ['order_id', 'order_item_id']


def main():
    """Frozen CSVs and temporal assignments -> O39 matrices, saved transformer and invariance audit manifest."""
    previous = json.loads((BASE/'manifest.json').read_text())
    files = [ROOT/'data/processed/ml_data.csv', ROOT/'data/processed/merged_data.csv',
             ROOT/'data/raw/olist_order_items_dataset.csv', BASE/'assignments.csv', BASE/'imputation_statistics.json']
    before = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    for name in ['data/processed/ml_data.csv', 'data/processed/merged_data.csv']:
        assert before[name] == previous['input_sha256'][name]
    assert before[str((BASE/'assignments.csv').relative_to(ROOT))] == previous['output_sha256']['assignments.csv']
    data = pd.read_csv(files[0]); raw = pd.read_csv(files[1])
    items = prepare_order_items(data, raw)
    assignments = pd.read_csv(BASE/'assignments.csv')
    aligned = items[KEYS].merge(assignments[KEYS+['eligible', 'proposed_split']], on=KEYS, validate='one_to_one')
    assert np.array_equal(aligned[KEYS], items[KEYS]) and len(aligned) == len(items)
    aligned.index = items.index
    eligible_items = items.loc[aligned.eligible]
    original = pd.read_csv(files[2], usecols=KEYS)
    original = original[original.order_id.isin(eligible_items.order_id)]
    coverage = original.merge(eligible_items[KEYS], on=KEYS, how='outer', validate='one_to_one', indicator=True)
    assert coverage['_merge'].eq('both').all() and len(coverage) == 85927
    review_check = eligible_items[KEYS+['review_score']].merge(raw[KEYS+['review_score']], on=KEYS, validate='one_to_one', suffixes=('_model','_raw'))
    assert review_check.review_score_model.eq(review_check.review_score_raw).all()
    parts = {name: items.loc[aligned.eligible & aligned.proposed_split.eq(name)].copy() for name in ['train','valid','test']}
    order_sets = {name:set(part.order_id) for name,part in parts.items()}
    assert all(not order_sets[a] & order_sets[b] for a,b in [('train','valid'),('train','test'),('valid','test')])
    transformer = OrderFeatureTransformer().fit(parts['train'])
    assert len(transformer.get_feature_names_out()) == 39
    fitted = transformer.item_imputer_
    statistics = {'coordinate_modes':fitted.coordinate_modes_, 'product_medians':fitted.product_medians_, 'distance_median':fitted.distance_median_}
    assert statistics == json.loads((BASE/'imputation_statistics.json').read_text())
    out = ROOT/'outputs/experiments'/datetime.now().strftime('order-input-validation-%Y%m%d-%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    joblib.dump(transformer, out/'order_transformer.joblib')
    restored = joblib.load(out/'order_transformer.joblib')
    transformer_digest = joblib.hash(transformer)
    results = []
    for expected in previous['periods']:
        name = expected['split']; part = parts[name]
        features = transformer.transform(part)
        labels = make_order_labels(part)
        assert features.index.equals(labels.index)
        assert features.shape == (expected['eligible_orders'], 39)
        assert len(part) == expected['eligible_item_rows']
        assert labels.eq(0).sum() == expected['risk_orders']
        assert np.isfinite(features.select_dtypes(include=[np.number]).to_numpy()).all()
        assert not features.isna().any().any()
        assert np.allclose(features.filter(like='category_share_').sum(axis=1),1)
        pd.testing.assert_frame_equal(restored.transform(part), features, check_exact=True)
        shuffled = part.sample(frac=1, random_state=42)
        pd.testing.assert_frame_equal(transformer.transform(shuffled), features, check_exact=True)
        renumbered = shuffled.copy()
        renumbered['order_item_id'] = renumbered.groupby('order_id').cumcount()+100
        pd.testing.assert_frame_equal(transformer.transform(renumbered), features, check_exact=True)
        for col in ['product_id','seller_id']:
            renumbered[col] = renumbered[col].map({v:f'new_{i}' for i,v in enumerate(renumbered[col].unique())})
        pd.testing.assert_frame_equal(transformer.transform(renumbered), features, check_exact=True)
        small_ids = features.index[:100]
        pd.testing.assert_frame_equal(transformer.transform(part[part.order_id.isin(small_ids)]), features.loc[small_ids], check_exact=True)
        features.to_csv(out/f'{name}_features.csv', index=True)
        labels.to_csv(out/f'{name}_labels.csv', index=True)
        item_counts = part.groupby('order_id').size().rename('item_count')
        item_counts.to_csv(out/f'{name}_item_counts.csv')
        assert item_counts.sum() == len(part)
        reread = pd.read_csv(out/f'{name}_features.csv', index_col='order_id', float_precision='round_trip')
        pd.testing.assert_frame_equal(reread,features,check_exact=True)
        results.append({'split':name,'orders':len(features),'item_rows':len(part),'features':features.shape[1],
                        'risk_orders':int(labels.eq(0).sum()),'missing_cells':int(features.isna().sum().sum()),
                        'shuffle_renumber_idrename_restore_batch_checks':'exact equality'})
        print(results[-1],flush=True)
    assert joblib.hash(transformer) == transformer_digest
    schema = {'feature_columns':transformer.get_feature_names_out().tolist(), 'categorical_features':[c for c in features if not pd.api.types.is_numeric_dtype(features[c])], 'category_columns':transformer.category_columns_,
              'imputation_statistics':statistics, 'order_medians':transformer.order_medians_.to_dict(),
              'excluded':['order_id','order_item_id','product_id','seller_id','review_score','review_label','item_count'],
              'numeric_order':'sort by order_id and numerical item values before reductions',
              'std_ddof':0,'count_policy':'all item rows; unique product/seller counts only'}
    (out/'feature_schema.json').write_text(json.dumps(schema,ensure_ascii=False,indent=2)+'\n')
    assert before == {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    result = {'status':'input_transform_complete_no_classifier_training','classifier_fit_count':0,'preprocessing_fit_count':1,
              'splits':results,'raw_item_key_difference':0,'model_raw_review_mismatches':0,'input_sha256':before,
              'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['src/olist_order_features.py','src/olist_imputation.py','src/olist_delivery_models.py','scripts/audit_order_inputs.py','tests/test_order_features.py']},
              'head_before_run':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'versions':{'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__,'sklearn':sklearn.__version__,'joblib':joblib.__version__},
              'transformer_storage':'order_transformer.joblib local only (existing gitignore); hash recorded; restore verified',
              'limitations':['classifier prediction invariance not run: no classifier trained','O40 not implemented/trained in this stage',
                             'transform requires all items of an order; coverage enforced by audit against source',
                             'historical availability/cohort limitations unchanged','category vocabulary size determines feature count; current frozen training has six categories'],
              'output_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())}}
    (out/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(out,flush=True)


if __name__=='__main__':main()
