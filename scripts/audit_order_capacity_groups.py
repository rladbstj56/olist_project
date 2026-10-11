"""Describe saved validation rankings at fixed global capacities; no model loading or fitting."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import subprocess
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'outputs/experiments/order-model-comparison-20261011-084408'
INPUT = ROOT/'outputs/experiments/order-input-validation-20261010-112545'
VARIANTS = ['baseline','without_sequence','order_weighted','O39','O40']


def group_metrics(labels, selected, groups):
    """Aligned order labels, global selection flags and groups -> group counts/rates.

    Selection is fixed before grouping. FPR uses actual non-risk orders as denominator;
    undefined rates remain missing rather than implying measured zero performance.
    """
    assert labels.index.equals(selected.index) and labels.index.equals(groups.index)
    frame = pd.DataFrame({'risk':labels.eq(0),'selected':selected,'group':groups})
    rows=[]
    for group,f in frame.groupby('group'):
        n=len(f); risk=int(f.risk.sum()); normal=n-risk; chosen=int(f.selected.sum())
        tp=int((f.risk & f.selected).sum()); fp=chosen-tp
        rows.append(dict(group=group,orders=n,risk_orders=risk,normal_orders=normal,risk_rate=risk/n,
                         selected_orders=chosen,selection_rate=chosen/n,
                         share_of_all_selected=chosen/int(selected.sum()) if selected.sum() else np.nan,
                         tp=tp,fp=fp,fn=risk-tp,tn=normal-fp,
                         precision=tp/chosen if chosen else np.nan,
                         recall=tp/risk if risk else np.nan,fpr=fp/normal if normal else np.nan))
    return rows


def main():
    """Hash-verified saved scores/features -> segment tables, paired changes and rank audit."""
    files={RUN/'validation_order_scores.csv':json.loads((RUN/'manifest.json').read_text())['output_sha256']['validation_order_scores.csv'],
           RUN/'validation_capacity_metrics.csv':json.loads((RUN/'manifest.json').read_text())['output_sha256']['validation_capacity_metrics.csv'],
           INPUT/'valid_features.csv':json.loads((INPUT/'manifest.json').read_text())['output_sha256']['valid_features.csv']}
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in files.items())
    scores=pd.read_csv(RUN/'validation_order_scores.csv',index_col='order_id',float_precision='round_trip')
    features=pd.read_csv(INPUT/'valid_features.csv',index_col='order_id',float_precision='round_trip')
    prior=pd.read_csv(RUN/'validation_capacity_metrics.csv')
    assert scores.index.is_unique and scores.index.equals(features.index)
    assert len(scores)==14621 and scores.review_label.eq(0).sum()==2270
    assert np.isfinite(scores[VARIANTS]).all().all() and scores[VARIANTS].ge(0).all().all() and scores[VARIANTS].le(1).all().all()
    assert scores.item_count.ge(features.distinct_seller_count).all()
    groups={'all':pd.Series('all',index=scores.index),
            'item_binary':scores.item_count.map(lambda n:'single' if n==1 else 'multi'),
            'item_count':scores.item_count.map(lambda n:str(n) if n<3 else '3+'),
            'seller_binary':features.distinct_seller_count.map(lambda n:'single' if n==1 else 'multi')}
    groups['item_seller']=groups['item_binary']+'_item/'+groups['seller_binary']+'_seller'
    ranks=scores[['review_label','item_count']].copy()
    ranks['seller_count']=features.distinct_seller_count.astype(int)
    rows=[]; exchanges=[]
    for v in VARIANTS:
        ordered=scores.reset_index().sort_values([v,'order_id'],kind='stable')
        ranks[v]=pd.Series(np.arange(1,len(scores)+1),index=ordered.order_id).reindex(scores.index)
    for fraction in [.05,.10,.20,.30,.40,.50]:
        k=int(np.ceil(len(scores)*fraction))
        flags={v:ranks[v].le(k) for v in VARIANTS}
        for v in VARIANTS:
            for dimension,group in groups.items():
                result=group_metrics(scores.review_label,flags[v],group)
                assert sum(r['orders'] for r in result)==len(scores)
                assert sum(r['selected_orders'] for r in result)==k
                assert sum(r['tp'] for r in result)==int((scores.review_label.eq(0)&flags[v]).sum())
                for row in result:rows.append(dict(variant=v,capacity=fraction,global_selected=k,dimension=dimension,**row))
            ref=prior[prior.variant.eq(v)&np.isclose(prior.capacity,fraction)].iloc[0]
            assert int((scores.review_label.eq(0)&flags[v]).sum())==ref.tp
        for reference in ['baseline','order_weighted','O39']:
            added=flags['O40'] & ~flags[reference]; removed=flags[reference] & ~flags['O40']
            assert added.sum()==removed.sum()
            for dimension,group in groups.items():
                for direction,selection in [('added',added),('removed',removed)]:
                    for row in group_metrics(scores.review_label,selection,group):
                        exchanges.append(dict(reference=reference,candidate='O40',capacity=fraction,direction=direction,dimension=dimension,**row))
    table=pd.DataFrame(rows)
    delta=[]
    for reference in ['baseline','order_weighted','O39']:
        keys=['capacity','dimension','group']
        joined=table[table.variant.eq('O40')].merge(table[table.variant.eq(reference)],on=keys,validate='one_to_one',suffixes=('_new','_ref'))
        for _,r in joined.iterrows():
            delta.append(dict(reference=reference,capacity=r.capacity,dimension=r.dimension,group=r['group'],
                              selected_delta=int(r.selected_orders_new-r.selected_orders_ref),tp_delta=int(r.tp_new-r.tp_ref),fp_delta=int(r.fp_new-r.fp_ref)))
    out=ROOT/'outputs/experiments'/datetime.now().strftime('order-capacity-groups-%Y%m%d-%H%M%S')
    out.mkdir(parents=True,exist_ok=False)
    table.to_csv(out/'group_metrics.csv',index=False)
    pd.DataFrame(delta).to_csv(out/'group_deltas.csv',index=False)
    pd.DataFrame(exchanges).to_csv(out/'selection_exchanges.csv',index=False)
    ranks.to_csv(out/'validation_ranks.csv')
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in files.items())
    manifest=dict(fit_count=0,new_predictions=0,split='valid',orders=len(scores),risk_orders=2270,
                  input_sha256={str(p.relative_to(ROOT)):h for p,h in files.items()},
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  policy='global ceil(n*f), f=.05/.10/.20/.30/.40/.50; ascending positive score, then order_id; no subgroup quotas',
                  checks='all subgroup partitions sum to global counts; all 30 global TP counts match prior artifact; added and removed global counts equal; input hashes unchanged',
                  limitations=['descriptive validation analysis; no causal conclusion or significance claim','validation repeatedly inspected','capacities not operationally selected','false positive means review4/5 under frozen label policy; not proof support contact would be unnecessary','item_count means item rows, not distinct products'],
                  output_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())})
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(table[(table.capacity.eq(.1)) & table.dimension.eq('item_binary')][['variant','group','orders','risk_orders','selected_orders','tp','fp','precision','recall','fpr']].to_string(index=False))
    print(out)


if __name__=='__main__':main()
