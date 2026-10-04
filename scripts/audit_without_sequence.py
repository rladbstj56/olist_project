"""One temporal Track B fit excluding sequence, with frozen-threshold and selected-threshold comparisons."""
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
    """Frozen baseline/data -> one 21-feature fit, paired metrics, input audit and traceable run manifest."""
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
    feature_cols=[c for c in PRE_ORDER_COLS if c!='order_item_id']
    X=model[PRE_ORDER_COLS+RAW_INPUT_COLS]
    pipeline=build_corrected_track_pipeline(X.loc[masks['train']],'B',feature_cols=feature_cols)
    assert pipeline.named_steps['classifier'].get_params()==previous['classifier_params']
    out=ROOT/'outputs/experiments'/datetime.now().strftime('without-sequence-%Y%m%d-%H%M%S')
    out.mkdir(parents=True,exist_ok=False)
    print(f'One fit, 21 features, {int(masks["train"].sum())} train rows: {out}',flush=True)
    pipeline.fit(X.loc[masks['train']],model.loc[masks['train'],'review_label'])
    used=[c for name,transformer,cols in pipeline.named_steps['preprocessor'].transformers_ if name!='remainder' for c in cols]
    assert set(used)==set(feature_cols) and len(used)==21
    assert list(pipeline.named_steps['classifier'].classes_)==[0,1]
    imp=pipeline.named_steps['train_only_imputation']
    statistics={'coordinate_modes':imp.coordinate_modes_,'product_medians':imp.product_medians_,'distance_median':imp.distance_median_}
    assert statistics==json.loads((BASE/'imputation_statistics.json').read_text())
    train=imp.transform(X.loc[masks['train']])
    audit=[]
    for col in PRE_ORDER_COLS:
        exact=[other for other in PRE_ORDER_COLS if other!=col and train[col].equals(train[other])]
        audit.append(dict(feature=col,used_in_ablation=col in feature_cols,train_unique_including_na=train[col].nunique(dropna=False),
                          train_missing=int(train[col].isna().sum()),exact_duplicate_columns='|'.join(exact)))
    pd.DataFrame(audit).to_csv(out/'training_feature_audit.csv',index=False)
    derivations={
        'total_price':np.allclose(train.total_price,train.price+train.freight_value,equal_nan=True),
        'freight_ratio':np.allclose(train.freight_ratio,train.freight_value/train.price.replace(0,np.nan),equal_nan=True),
        'sp_route_type':train.sp_route_type.eq(train.is_sp_customer+train.is_sp_seller).all(),
        'sp_route_type_customer':train.sp_route_type_customer.eq(train.is_sp_customer+train.is_sp_customer*train.is_sp_seller).all(),
        'sp_route_type_seller':train.sp_route_type_seller.eq(train.is_sp_seller+train.is_sp_customer*train.is_sp_seller).all(),
    }
    (out/'derived_feature_checks.json').write_text(json.dumps({k:bool(v) for k,v in derivations.items()},indent=2)+'\n')
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
    comparison=baseline[KEYS+['split','review_label','positive_probability']].merge(predictions[KEYS+['split','review_label','positive_probability']],on=KEYS+['split','review_label'],validate='one_to_one',suffixes=('_baseline','_without_sequence'))
    assert len(comparison)==len(baseline)==len(predictions)
    comparison.to_csv(out/'paired_predictions.csv',index=False)
    metrics=[]; sizes=[]; monthly=[]
    for variant,pred,threshold in [('baseline',baseline,previous['positive_threshold']),('without_sequence_fixed',predictions,previous['positive_threshold']),('without_sequence_selected',predictions,selected)]:
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
    # Changing the identifier must not affect the fitted encoder or predictions.
    sample=X.loc[masks['valid']].iloc[:100].copy()
    original=pipeline.predict_proba(sample)[:,1]
    sample['order_item_id']=999
    assert np.array_equal(original,pipeline.predict_proba(sample)[:,1])
    saved=pd.read_csv(out/'predictions.csv',float_precision='round_trip')
    for result in metrics:
        if result['variant']=='baseline':continue
        part=saved[saved.split.eq(result['split'])]
        if result['unit']!='item':part=order_predictions(part,result['unit'].split('_')[1])
        recalc=evaluate(part,result['positive_threshold'])
        assert all(np.isclose(recalc[k],v,rtol=0,atol=1e-12) for k,v in result.items() if k not in ['variant','split','unit'])
    assert hashes=={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    result={'fit_count':1,'track':'B','removed_model_features':['order_item_id'],'retained_feature_columns':feature_cols,
        'key_columns_preserved':KEYS,'training_rows':int(masks['train'].sum()),'baseline_positive_threshold':previous['positive_threshold'],
        'selected_positive_threshold':selected,'classifier_params':pipeline.named_steps['classifier'].get_params(),
        'versions':versions,'input_sha256':hashes,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'head_before_run':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'checks':['same rows/order keys/labels','same model parameters and package versions','same fitted imputation statistics',
                  'encoder uses exactly 21 specified columns','sequence perturbation invariant on 100 valid rows','saved-score metrics roundtrip','baseline/input hashes unchanged'],
        'limitations':['test already inspected: diagnostic comparison not untouched confirmation','all previous temporal cohort limitations remain',
                       'no automatic production adoption','derived features are not proven useless','feature subsampling may change after column removal despite same seed'],
        'output_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())}}
    (out/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(pd.DataFrame(metrics).query('unit=="order_max"').to_string(index=False),flush=True)
    print(out,flush=True)


if __name__=='__main__':main()
