"""Exactly two planned fits: O39 versus O40 on frozen temporal order inputs."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import platform
import subprocess
import sys

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.olist_delivery_models import build_track_b_pipeline, risk_classification_metrics, select_positive_threshold
from src.olist_order_features import prepare_order_items, make_order_labels
from src.olist_order_models import predict_order_model

BASE = ROOT/'outputs/experiments/order-input-validation-20261010-112545'
OLD = ROOT/'outputs/experiments/equal-capacity-20261004-232050'


def capacity_metrics(scores, variant):
    """Order scores/labels -> predeclared capacity metrics and boundary-tie sensitivity."""
    ranked = scores.reset_index().sort_values([variant, 'order_id'], kind='stable')
    total = int(ranked.review_label.eq(0).sum())
    rows = []
    for fraction in [.05, .10, .20, .30, .40, .50]:
        k = int(np.ceil(len(ranked)*fraction))
        selected = ranked.iloc[:k]
        boundary = selected[variant].iloc[-1]
        strict = ranked[ranked[variant] < boundary]
        ties = ranked[ranked[variant].eq(boundary)]
        slots = k-len(strict)
        strict_tp = int(strict.review_label.eq(0).sum())
        tie_tp = int(ties.review_label.eq(0).sum())
        tp = int(selected.review_label.eq(0).sum())
        low = strict_tp+max(0, slots-(len(ties)-tie_tp))
        high = strict_tp+min(slots, tie_tp)
        assert low <= tp <= high
        rows.append(dict(variant=variant, capacity=fraction, reviewed_orders=k, tp=tp, fp=k-tp,
                         fn=total-tp, precision=tp/k, recall=tp/total,
                         boundary_tie_orders=len(ties), boundary_tie_selected=slots,
                         tp_min_with_ties=low, tp_max_with_ties=high,
                         tp_expected_random_tie=strict_tp+slots*tie_tp/len(ties)))
    return rows


def main():
    """Hash-verified matrices -> two fitted bundles, aligned comparisons and inference audit."""
    prior = json.loads((BASE/'manifest.json').read_text())
    protected = {ROOT/p:h for group in ['input_sha256', 'source_sha256'] for p,h in prior[group].items()}
    protected.update({BASE/p:h for p,h in prior['output_sha256'].items()})
    old_manifest = json.loads((OLD/'manifest.json').read_text())
    protected[OLD/'validation_order_scores.csv'] = old_manifest['output_sha256']['validation_order_scores.csv']
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p,h in protected.items())
    transformer = joblib.load(BASE/'order_transformer.joblib')
    original_state = joblib.hash(transformer)
    schema = json.loads((BASE/'feature_schema.json').read_text())
    matrices = {s:pd.read_csv(BASE/f'{s}_features.csv', index_col='order_id', float_precision='round_trip') for s in ['train','valid','test']}
    labels = {s:pd.read_csv(BASE/f'{s}_labels.csv', index_col='order_id').review_label for s in matrices}
    counts = {s:pd.read_csv(BASE/f'{s}_item_counts.csv', index_col='order_id').item_count for s in matrices}
    for s,x in matrices.items():
        assert x.index.is_unique and x.index.equals(labels[s].index) and x.index.equals(counts[s].index)
        assert list(x.columns) == schema['feature_columns'] and len(x.columns) == 39
        assert set(labels[s].unique()) == {0,1}
    assert all(not set(matrices[a].index) & set(matrices[b].index) for a,b in [('train','valid'),('train','test'),('valid','test')])
    scores = pd.read_csv(OLD/'validation_order_scores.csv', index_col='order_id', float_precision='round_trip')
    pd.testing.assert_series_equal(scores.review_label, labels['valid'])
    pd.testing.assert_series_equal(scores.item_count, counts['valid'])
    items = prepare_order_items(pd.read_csv(ROOT/'data/processed/ml_data.csv'), pd.read_csv(ROOT/'data/processed/merged_data.csv'))
    parts = {s:items[items.order_id.isin(x.index)].copy() for s,x in matrices.items()}
    raw_keys = pd.read_csv(ROOT/'data/raw/olist_order_items_dataset.csv', usecols=['order_id','order_item_id'])
    for s,part in parts.items():
        pd.testing.assert_series_equal(make_order_labels(part), labels[s])
        pd.testing.assert_series_equal(part.groupby('order_id').size().rename('item_count'), counts[s])
        coverage = raw_keys[raw_keys.order_id.isin(matrices[s].index)].merge(part[['order_id','order_item_id']], how='outer', validate='one_to_one', indicator=True)
        assert coverage['_merge'].eq('both').all()
        pd.testing.assert_frame_equal(transformer.transform(part), matrices[s], check_exact=True)
    out = ROOT/'outputs/experiments'/datetime.now().strftime('order-model-comparison-%Y%m%d-%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    bundles = {}; metadata = {}; predictions = []; metrics = []; checks = []; groups = []
    for variant,with_count in [('O39',False),('O40',True)]:
        xtrain = matrices['train'].copy()
        xvalid = matrices['valid'].copy()
        if with_count:
            xtrain['item_count'] = counts['train']; xvalid['item_count'] = counts['valid']
        model = build_track_b_pipeline(xtrain)
        model.fit(xtrain, labels['train'])
        positive = model.predict_proba(xvalid)[:,1]
        threshold = select_positive_threshold(labels['valid'], positive)
        grid = pd.DataFrame([risk_classification_metrics(labels['valid'], positive, float(t)) for t in np.arange(.10,.91,.01)])
        grid['gap'] = (grid.risk_recall-grid.balanced_acc).abs()
        assert threshold == grid.loc[grid.gap.idxmin(),'positive_threshold']
        grid.to_csv(out/f'{variant}_threshold_grid.csv', index=False)
        bundle = dict(transformer=transformer, model=model, include_item_count=with_count,
                      feature_columns=list(xtrain.columns), positive_threshold=threshold)
        joblib.dump(bundle, out/f'{variant}.joblib')
        bundles[variant] = joblib.load(out/f'{variant}.joblib')
        scores[variant] = positive
        metadata[variant] = dict(feature_columns=list(xtrain.columns), positive_threshold=threshold,
                                 encoded_feature_count=len(model['preprocessor'].get_feature_names_out()),
                                 classifier_parameters=model['classifier'].get_params(), fit_count=1)
        print(variant, 'validation AP', average_precision_score(labels['valid'].eq(0),1-positive), 'threshold',threshold,flush=True)
    capacities = pd.DataFrame([r for v in ['baseline','without_sequence','order_weighted','O39','O40'] for r in capacity_metrics(scores,v)])
    capacities.to_csv(out/'validation_capacity_metrics.csv',index=False)
    scores.to_csv(out/'validation_order_scores.csv')
    print(capacities[['variant','capacity','tp','precision','recall']].to_string(index=False),flush=True)
    for variant,bundle in bundles.items():
        for s in ['valid','test']:
            part = parts[s]
            expected_x = matrices[s].copy()
            if bundle['include_item_count']: expected_x['item_count'] = counts[s]
            expected = bundle['model'].predict_proba(expected_x)[:,1]
            state = joblib.hash(bundle)
            result = predict_order_model(bundle,part)
            np.testing.assert_array_equal(result.positive_probability,expected)
            altered = part.sample(frac=1,random_state=42)
            pd.testing.assert_frame_equal(predict_order_model(bundle,altered), result,check_exact=True)
            altered = altered.copy(); altered['order_item_id'] = altered.groupby('order_id').cumcount()+100
            pd.testing.assert_frame_equal(predict_order_model(bundle,altered), result,check_exact=True)
            for col in ['product_id','seller_id']:
                altered[col] = altered[col].map({v:f'new_{i}' for i,v in enumerate(altered[col].unique())})
            pd.testing.assert_frame_equal(predict_order_model(bundle,altered), result,check_exact=True)
            altered['review_score'] = 999; altered['review_label'] = -1
            pd.testing.assert_frame_equal(predict_order_model(bundle,altered), result,check_exact=True)
            ids = result.index[:100]
            pd.testing.assert_frame_equal(predict_order_model(bundle,part[part.order_id.isin(ids)]),result.loc[ids],check_exact=True)
            assert joblib.hash(bundle) == state
            checks.append(dict(variant=variant,split=s,orders=len(result),restore_shuffle_renumber_idrename_labelchange_batch='exact equality',max_probability_difference=0))
            result['review_label'] = labels[s]; result['item_count'] = counts[s]
            result['purchase_month'] = pd.to_datetime(part.groupby('order_id')['source__purchase_timestamp'].min()).dt.strftime('%Y-%m')
            result['seller_count'] = matrices[s].distinct_seller_count.astype(int)
            result['positive_threshold'] = bundle['positive_threshold']; result['variant'] = variant; result['split'] = s
            predictions.append(result.reset_index())
            segments = {'all':pd.Series('all',index=result.index), 'month':result.purchase_month,
                        'item_count':result.item_count.map(lambda n:str(n) if n<3 else '3+'),
                        'seller_count':result.seller_count.map(lambda n:'1' if n==1 else '2+')}
            for kind,segment in segments.items():
                for group,frame in result.groupby(segment):
                    m = risk_classification_metrics(frame.review_label,frame.positive_probability.to_numpy(),bundle['positive_threshold'])
                    risk = frame.review_label.eq(0); flag = frame.risk_flag
                    tn,fp,fn,tp = confusion_matrix(risk,flag,labels=[False,True]).ravel()
                    assert tp == (risk & flag).sum() and fp == (~risk & flag).sum()
                    row = dict(variant=variant,split=s,segment=kind,group=group,orders=len(frame),risk_orders=int(risk.sum()),
                               flagged_orders=int(flag.sum()),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),**m)
                    (metrics if kind=='all' else groups).append(row)
            print(variant,s,metrics[-1],flush=True)
    pd.concat(predictions,ignore_index=True).to_csv(out/'predictions.csv',index=False)
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(groups).to_csv(out/'segment_metrics.csv',index=False)
    assert joblib.hash(transformer) == original_state
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p,h in protected.items())
    manifest = dict(classifier_fit_count=2,aggregation_fit_count=0,adopted_model=None,
                    input_sha256={str(p.relative_to(ROOT)):h for p,h in protected.items()},
                    source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['scripts/audit_order_models.py','src/olist_order_models.py']},
                    head_before_run=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    versions=dict(python=platform.python_version(),pandas=pd.__version__,numpy=np.__version__,sklearn=sklearn.__version__,lightgbm=lightgbm.__version__,joblib=joblib.__version__),
                    models=metadata,invariance_checks=checks,
                    train_order_class_counts={str(k):int(v) for k,v in labels['train'].value_counts().items()},
                    class_weight_policy='balanced from training orders; no item inverse-count sample weights',
                    threshold_policy='validation orders: first minimum abs(risk_recall-balanced_acc), grid .10:.01:.90; risk iff positive < threshold',
                    capacity_policy='validation only, ceil(n*f); ascending positive probability then order_id; label-independent tie handling',
                    storage='joblib bundles local only under existing ignore; hashes and restore checks recorded',
                    limitations=['repeatedly inspected validation/test, not independent final confirmation','no automatic adoption or UI change',
                                 'O39 still contains implicit item-count information','historical input availability and outcome-conditioned cohort unresolved',
                                 'scores uncalibrated; no causal or intervention effect claim','no confidence intervals or significance claims'],
                    output_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())})
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(out,flush=True)


if __name__ == '__main__': main()
