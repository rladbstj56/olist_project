"""Compare UI/Track C imputation against commit 78653c7, without replacing old outputs."""
from pathlib import Path
import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import types
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_pinball_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import olist_delivery_models as current
from src.olist_imputation import RAW_INPUT_COLS


def delivery_metrics(actual, predicted, expected, quantile):
    """Actual/predicted/promised days + quantile -> item-row regression/simulation metrics."""
    actual, predicted, expected = map(np.asarray, (actual, predicted, expected))
    recommended = np.maximum(expected, np.ceil(predicted))
    return {
        'mae': float(mean_absolute_error(actual, predicted)),
        'pinball_loss': float(mean_pinball_loss(actual, predicted, alpha=quantile)),
        'quantile_coverage': float((actual <= predicted).mean()),
        'current_over_3_rate': float(((actual-expected)>3).mean()),
        'recommended_over_3_rate': float(((actual-recommended)>3).mean()),
        'avg_adjustment_days': float((recommended-expected).mean()),
        'adjusted_item_row_share': float((recommended>expected).mean()),
    }


def main():
    """Existing data + optional temporary model cache -> new run directory, ten model fits."""
    parser=argparse.ArgumentParser()
    parser.add_argument('--console-cache', type=Path)
    args=parser.parse_args()
    out=ROOT/'outputs/experiments'/datetime.now().strftime('console-imputation-%Y%m%d-%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    baseline_code=subprocess.check_output(['git','show','78653c7:src/olist_delivery_models.py'],cwd=ROOT,text=True)
    baseline=types.ModuleType('src._console_baseline')
    baseline.__file__=str(ROOT/'src/olist_delivery_models.py')
    sys.modules[baseline.__name__]=baseline
    exec(compile(baseline_code,'78653c7:src/olist_delivery_models.py','exec'),baseline.__dict__)
    data=current.load_ml_data(ROOT/'data/processed/ml_data.csv')
    raw=pd.read_csv(ROOT/'data/processed/merged_data.csv')
    metrics=[];predictions=[];splits=[];checks={};stats={}
    for variant,module in [('legacy',baseline),('train_only',current)]:
        print('UI B/C',variant,flush=True)
        console=module.train_console_artifacts(data)
        frame=(current.prepare_corrected_model_frame(data,raw_data=raw) if variant=='train_only'
               else baseline.prepare_labeled_model_frame(data))
        tr,te,*_=current.split_by_order(frame,frame.review_label,frame.order_id)
        assert not set(tr.order_id)&set(te.order_id)
        inputs=te[current.PRE_ORDER_COLS + (RAW_INPUT_COLS if variant=='train_only' else [])]
        probability=console.models.track_b.predict_proba(inputs)[:,1]
        days=console.models.track_c.predict(inputs)
        metrics.append({'population':'UI','variant':variant,'track':'B','quantile':None,
                        **current.risk_classification_metrics(te.review_label,probability),
                        'high_risk_unique_orders':len(console.high_risk_orders)})
        metrics.append({'population':'UI','variant':variant,'track':'C','quantile':.9,
                        **delivery_metrics(te.delivery_days,days,te.expected_delivery_days,.9)})
        p=te[['order_id','order_item_id','review_label','delivery_days','expected_delivery_days']].copy()
        p.insert(0,'row_id',te.index);p['population']='UI';p['variant']=variant;p['quantile']=.9
        p['positive_probability']=probability;p['predicted_delivery_days']=days;predictions.append(p)
        if variant=='train_only':
            prepared=console.models.track_b[0].transform(inputs)
            np.testing.assert_allclose(console.models.track_b[1:].predict_proba(prepared)[:,1],probability,rtol=0,atol=1e-12)
            np.testing.assert_allclose(console.models.track_c[1:].predict(prepared),days,rtol=0,atol=1e-12)
            display=console.test_orders.set_index(['order_id','order_item_id'])
            aligned=display.loc[pd.MultiIndex.from_frame(te[['order_id','order_item_id']])]
            np.testing.assert_allclose(aligned.review_risk_probability,1-probability,rtol=0,atol=1e-12)
            for index in [0,1,100]:
                one=current.predict_order(console.models,prepared.iloc[[index]])
                np.testing.assert_allclose(one['review_risk_probability'],1-probability[index],rtol=0,atol=1e-12)
            checks['manual_batch_and_display_prediction_parity']=True
            checks['test_item_rows']=len(console.test_orders)
            stats['ui_distance_median']=console.models.track_b[0].distance_median_
            if args.console_cache:
                import joblib
                joblib.dump(console,args.console_cache)
            for name,part in [('train',tr),('test',te)]:
                sp=part[['order_id','order_item_id']].copy();sp.insert(0,'row_id',part.index);sp['population']='UI';sp['split']=name;splits.append(sp)
    frame=current.prepare_corrected_model_frame(data,label_policy=None,raw_data=raw)
    tr,te,*_=current.split_by_order(frame,frame.delivery_days,frame.order_id)
    assert not set(tr.order_id)&set(te.order_id)
    for name,part in [('train',tr),('test',te)]:
        sp=part[['order_id','order_item_id']].copy();sp.insert(0,'row_id',part.index);sp['population']='Track C evaluation';sp['split']=name;splits.append(sp)
    for q in [.8,.9,.95]:
        for variant,module in [('legacy',baseline),('train_only',current)]:
            print('Track C evaluation',q,variant,flush=True)
            columns=current.PRE_ORDER_COLS + (RAW_INPUT_COLS if variant=='train_only' else [])
            model=(current.build_corrected_track_pipeline(tr,'C',quantile=q) if variant=='train_only'
                   else baseline.build_track_c_pipeline(tr[columns],quantile=q))
            model.fit(tr[columns],tr.delivery_days);days=model.predict(te[columns])
            metrics.append({'population':'Track C evaluation','variant':variant,'track':'C','quantile':q,
                            **delivery_metrics(te.delivery_days,days,te.expected_delivery_days,q)})
            p=te[['order_id','order_item_id','delivery_days','expected_delivery_days']].copy()
            p.insert(0,'row_id',te.index);p['population']='Track C evaluation';p['variant']=variant;p['quantile']=q
            p['predicted_delivery_days']=days;predictions.append(p)
            if variant=='train_only': stats['c_evaluation_distance_median']=model[0].distance_median_
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    pd.concat(predictions).to_csv(out/'predictions.csv',index=False)
    pd.concat(splits).to_csv(out/'split_assignments.csv',index=False)
    files=['src/olist_delivery_models.py','src/olist_imputation.py','streamlit_app.py','scripts/audit_console_imputation.py',
           'data/processed/ml_data.csv','data/processed/merged_data.csv']
    manifest={'baseline_commit':'78653c7','head_before_run':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'timestamp':datetime.now().astimezone().isoformat(),'fits':10,'risk_threshold':current.TRACK_B_RISK_THRESHOLD,
              'python':platform.python_version(),'packages':{x:importlib.metadata.version(x) for x in ['numpy','pandas','scikit-learn','lightgbm','streamlit']},
              'checks':checks,'statistics':stats,'input_and_code_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},
              'artifact_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print('COMPLETE',out,flush=True)
    print(pd.DataFrame(metrics).to_string(index=False),flush=True)


if __name__=='__main__': main()
