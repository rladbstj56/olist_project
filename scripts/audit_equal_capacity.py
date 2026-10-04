"""Compare saved validation order rankings at predeclared equal review capacities; no fitting."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import subprocess
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
RUNS={'baseline':'temporal-validation-20261003-202427','without_sequence':'without-sequence-20261003-204301','order_weighted':'order-weighted-20261004-230716'}


def main():
    """Saved validation scores -> aligned order scores, fixed-capacity metrics and boundary-tie ranges."""
    hashes={}; scores=None
    for name,folder in RUNS.items():
        run=ROOT/'outputs/experiments'/folder
        manifest=json.loads((run/'manifest.json').read_text())
        for filename in ['manifest.json','predictions.csv']:
            p=run/filename;hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
        assert hashes[str((run/'predictions.csv').relative_to(ROOT))]==manifest['output_sha256']['predictions.csv']
        p=pd.read_csv(run/'predictions.csv',float_precision='round_trip')
        p=p[p.split.eq('valid')].copy()
        assert not p.duplicated(['order_id','order_item_id']).any()
        assert p.groupby('order_id').review_label.nunique().eq(1).all()
        o=p.groupby('order_id').agg(review_label=('review_label','first'),item_count=('order_item_id','size'),
                                   positive_probability=('positive_probability','min'))
        o=o.rename(columns={'positive_probability':name})
        if scores is None:scores=o
        else:
            assert scores.index.equals(o.index)
            assert scores[['review_label','item_count']].equals(o[['review_label','item_count']])
            scores[name]=o[name]
    assert len(scores)==14621 and scores.review_label.eq(0).sum()==2270
    out=ROOT/'outputs/experiments'/datetime.now().strftime('equal-capacity-%Y%m%d-%H%M%S')
    out.mkdir(parents=True,exist_ok=False)
    scores.to_csv(out/'validation_order_scores.csv')
    n=len(scores); total_risk=int(scores.review_label.eq(0).sum())
    budgets=[(f'{int(f*100)}%',int(np.ceil(n*f))) for f in [.05,.10,.20,.30,.40,.50]]
    budgets.append(('historical_baseline_threshold_count',6553))
    result=[]
    for name in RUNS:
        ordered=scores.reset_index().sort_values([name,'order_id'],ascending=[True,True],kind='stable')
        for label,k in budgets:
            chosen=ordered.iloc[:k]
            boundary=chosen[name].iloc[-1]
            strict=ordered[ordered[name]<boundary]
            tied=ordered[ordered[name].eq(boundary)]
            slots=k-len(strict); tie_risk=int(tied.review_label.eq(0).sum())
            strict_tp=int(strict.review_label.eq(0).sum())
            lower=strict_tp+max(0,slots-(len(tied)-tie_risk))
            upper=strict_tp+min(slots,tie_risk)
            tp=int(chosen.review_label.eq(0).sum())
            assert len(chosen)==k and lower<=tp<=upper
            result.append(dict(variant=name,capacity=label,reviewed_orders=k,reviewed_fraction=k/n,tp=tp,fp=k-tp,
                fn=total_risk-tp,precision=tp/k,recall=tp/total_risk,lift=(tp/k)/(total_risk/n),
                selected_multi_item_orders=int(chosen.item_count.gt(1).sum()),
                positive_score_boundary=float(boundary),boundary_tie_orders=len(tied),boundary_tie_selected=slots,
                tp_min_with_ties=lower,tp_max_with_ties=upper,
                tp_expected_random_tie=strict_tp+slots*tie_risk/len(tied)))
    table=pd.DataFrame(result)
    table.to_csv(out/'capacity_metrics.csv',index=False)
    ref=table[table.variant.eq('baseline')].set_index('capacity').tp
    table['tp_delta_vs_baseline']=table.tp-table.capacity.map(ref)
    table.to_csv(out/'comparison.csv',index=False)
    assert hashes=={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in hashes}
    saved=pd.read_csv(out/'validation_order_scores.csv',float_precision='round_trip')
    for row in result:
        top=saved.sort_values([row['variant'],'order_id']).iloc[:row['reviewed_orders']]
        assert int(top.review_label.eq(0).sum())==row['tp']
    manifest={'fit_count':0,'new_predictions':0,'split':'valid','orders':n,'risk_orders':total_risk,
        'aggregation':'maximum item risk = minimum positive score','capacity_policy':'ceil(n*f), f=5/10/20/30/40/50%; baseline historical count 6553 supplemental',
        'tie_policy':'ascending order_id, label-independent; sensitivity range and random-tie expectation reported',
        'input_sha256':hashes,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'limitations':['validation repeatedly inspected; not final independent confirmation','capacity fractions are scenarios, not chosen operational quotas',
                       'no inference on real-world intervention benefit','no confidence intervals or significance test','ranking does not verify permutation invariance'],
        'output_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())}}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(table[['variant','capacity','reviewed_orders','tp','precision','recall','tp_delta_vs_baseline','tp_min_with_ties','tp_max_with_ties']].to_string(index=False))
    print(out)


if __name__=='__main__':main()
