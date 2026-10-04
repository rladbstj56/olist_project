"""One temporal Track B fit with inverse order-size sample weights and unchanged class weights."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.olist_delivery_models import PRE_ORDER_COLS, RAW_INPUT_COLS, prepare_corrected_model_frame, build_corrected_track_pipeline, select_positive_threshold, risk_classification_metrics
from audit_temporal_model import evaluate, order_predictions

BASE = ROOT / 'outputs/experiments/temporal-validation-20261003-202427'
KEYS = ['order_id', 'order_item_id']


def main():
    """Frozen baseline/data -> one order-weighted fit, saved weights, paired metrics and traceable run manifest."""
    previous = json.loads((BASE/'manifest.json').read_text())
    files = [ROOT/p for p in previous['input_sha256']] + list(BASE.iterdir())
    hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    for p,digest in previous['input_sha256'].items(): assert hashes[p] == digest
    for p,digest in previous['output_sha256'].items(): assert hashlib.sha256((BASE/p).read_bytes()).hexdigest() == digest
    for p,digest in previous['code_sha256'].items(): assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == digest
    versions={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'lightgbm':lightgbm.__version__}
    assert versions == previous['versions']
    raw=pd.read_csv(ROOT/'data/processed/merged_data.csv')
    model=prepare_corrected_model_frame(pd.read_csv(ROOT/'data/processed/ml_data.csv'),raw_data=raw)
    assignments=pd.read_csv(BASE/'assignments.csv')
    aligned=model[KEYS].merge(assignments,on=KEYS,validate='one_to_one')
    assert len(aligned)==len(model) and np.array_equal(aligned[KEYS],model[KEYS])
    aligned.index=model.index
    masks={s:aligned.eligible & aligned.proposed_split.eq(s) for s in ['train','valid','test']}
    for p in previous['periods']:
        part=model.loc[masks[p['split']]]
        assert len(part)==p['eligible_item_rows'] and part.order_id.nunique()==p['eligible_orders']
    feature_cols=list(PRE_ORDER_COLS)
    X=model[PRE_ORDER_COLS+RAW_INPUT_COLS]
    pipeline=build_corrected_track_pipeline(X.loc[masks['train']],'B',feature_cols=feature_cols)
    assert pipeline.named_steps['classifier'].get_params()==previous['classifier_params']
    out=ROOT/'outputs/experiments'/datetime.now().strftime('order-weighted-%Y%m%d-%H%M%S')
    out.mkdir(parents=True,exist_ok=False)
    print(f'One fit, 22 features, inverse order-size weights, {int(masks["train"].sum())} train rows: {out}',flush=True)
    train_rows=model.loc[masks['train'],KEYS+['review_label']].copy()
    assert train_rows.groupby('order_id').review_label.nunique().eq(1).all()
    counts=train_rows.groupby('order_id').order_id.transform('size')
    train_rows['order_items']=counts
    train_rows['raw_order_weight']=1.0/counts
    train_rows['sample_weight']=train_rows.raw_order_weight/train_rows.raw_order_weight.mean()
    label_counts=train_rows.review_label.value_counts()
    train_rows['class_weight']=train_rows.review_label.map(len(train_rows)/(2*label_counts))
    train_rows['effective_weight']=train_rows.sample_weight*train_rows.class_weight
    assert np.allclose(train_rows.groupby('order_id').raw_order_weight.sum(),1)
    assert np.isclose(train_rows.sample_weight.mean(),1)
    order_weights=train_rows.groupby('order_id').agg(review_label=('review_label','first'),
        items=('order_items','first'),raw_weight=('raw_order_weight','sum'),
        sample_weight=('sample_weight','sum'),effective_weight=('effective_weight','sum'))
    assert all(np.allclose(g.effective_weight,g.effective_weight.iloc[0]) for _,g in order_weights.groupby('review_label'))
    train_rows.to_csv(out/'training_weights.csv',index_label='row_id')
    order_weights.to_csv(out/'order_weights.csv')
    train_rows.groupby('review_label').agg(rows=('sample_weight','size'),orders=('order_id','nunique'),
        sample_weight_sum=('sample_weight','sum'),effective_weight_sum=('effective_weight','sum')).to_csv(out/'class_weight_totals.csv')
    pipeline.fit(X.loc[masks['train']],model.loc[masks['train'],'review_label'],
                 classifier__sample_weight=train_rows.sample_weight.to_numpy())
    used=[c for name,transformer,cols in pipeline.named_steps['preprocessor'].transformers_ if name!='remainder' for c in cols]
    assert set(used)==set(feature_cols) and len(used)==22
    assert list(pipeline.named_steps['classifier'].classes_)==[0,1]
    imp=pipeline.named_steps['train_only_imputation']
    statistics={'coordinate_modes':imp.coordinate_modes_,'product_medians':imp.product_medians_,'distance_median':imp.distance_median_}
    assert statistics==json.loads((BASE/'imputation_statistics.json').read_text())
    predictions=[]
    for name in ['valid','test']:
        mask=masks[name]
        part=model.loc[mask,KEYS+['review_label']].copy()
        part['row_id']=part.index
        part['split']=name
        part['purchase_month']=pd.to_datetime(aligned.loc[mask,'order_purchase_timestamp']).dt.strftime('%Y-%m')
        part['positive_probability']=pipeline.predict_proba(X.loc[mask])[:,1]
        predictions.append(part)
    predictions=pd.concat(predictions)
    valid=predictions[predictions.split.eq('valid')]
    selected=select_positive_threshold(valid.review_label,valid.positive_probability.to_numpy())
    grid=pd.DataFrame([risk_classification_metrics(valid.review_label,valid.positive_probability.to_numpy(),float(t)) for t in np.arange(.10,.91,.01)])
    grid['gap']=(grid.risk_recall-grid.balanced_acc).abs()
    grid['selected']=grid.positive_threshold.eq(selected)
    grid.to_csv(out/'validation_thresholds.csv',index=False)
    predictions['positive_threshold']=selected
    predictions.to_csv(out/'predictions.csv',index=False)
    baseline=pd.read_csv(BASE/'predictions.csv',float_precision='round_trip')
    comparison=baseline[KEYS+['split','review_label','positive_probability']].merge(predictions[KEYS+['split','review_label','positive_probability']],on=KEYS+['split','review_label'],validate='one_to_one',suffixes=('_baseline','_order_weighted'))
    assert len(comparison)==len(baseline)==len(predictions)
    comparison.to_csv(out/'paired_predictions.csv',index=False)
    metrics=[]; sizes=[]; monthly=[]
    for variant,pred,threshold in [('baseline',baseline,previous['positive_threshold']),('order_weighted_fixed',predictions,previous['positive_threshold']),('order_weighted_selected',predictions,selected)]:
        for split in ['valid','test']:
            part=pred[pred.split.eq(split)]
            for unit,frame in [('item',part),('order_max',order_predictions(part,'max')),('order_mean',order_predictions(part,'mean'))]:
                metrics.append(dict(variant=variant,split=split,unit=unit,**evaluate(frame,threshold)))
                for month,group in frame.groupby('purchase_month'):
                    monthly.append(dict(variant=variant,split=split,unit=unit,month=month,**evaluate(group,threshold)))
                if unit!='item':
                    frame['item_count_group']=np.select([frame.item_count.eq(1),frame.item_count.eq(2)],['1','2'],default='3+')
                    for size,group in frame.groupby('item_count_group'):
                        sizes.append(dict(variant=variant,split=split,unit=unit,item_count_group=size,**evaluate(group,threshold)))
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(sizes).to_csv(out/'item_count_metrics.csv',index=False)
    pd.DataFrame(monthly).to_csv(out/'monthly_metrics.csv',index=False)
    saved=pd.read_csv(out/'predictions.csv',float_precision='round_trip')
    for result in metrics:
        if result['variant']=='baseline':continue
        part=saved[saved.split.eq(result['split'])]
        if result['unit']!='item':part=order_predictions(part,result['unit'].split('_')[1])
        recalc=evaluate(part,result['positive_threshold'])
        assert all(np.isclose(recalc[k],v,rtol=0,atol=1e-12) for k,v in result.items() if k not in ['variant','split','unit'])
    assert hashes=={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    result={'fit_count':1,'track':'B','removed_model_features':[],'sample_weight_formula':'(1 / train order item count) / mean(1 / train order item count)', 'class_weight_policy':'unchanged row-frequency balanced, multiplied by sample weights', 'sample_weight_sum':float(train_rows.sample_weight.sum()),'effective_weight_sum':float(train_rows.effective_weight.sum()),'retained_feature_columns':feature_cols,
        'key_columns_preserved':KEYS,'training_rows':int(masks['train'].sum()),'baseline_positive_threshold':previous['positive_threshold'],
        'selected_positive_threshold':selected,'classifier_params':pipeline.named_steps['classifier'].get_params(),
        'versions':versions,'input_sha256':hashes,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'head_before_run':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'checks':['same rows/order keys/labels','same model parameters and package versions','same fitted imputation statistics',
                  'encoder uses original 22 columns','raw order weight sums equal one','mean sample weight one','equal effective totals per order within each class','saved-score metrics roundtrip','baseline/input hashes unchanged'],
        'limitations':['test already inspected: diagnostic comparison not untouched confirmation','all previous temporal cohort limitations remain',
                       'no automatic production adoption','class-weight multipliers fixed but effective class mass changes','imputation statistics remain row-based','within-class order equality is not between-class equality'],
        'output_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())}}
    (out/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(pd.DataFrame(metrics).query('unit=="order_max"').to_string(index=False),flush=True)
    print(out,flush=True)


if __name__=='__main__':main()
