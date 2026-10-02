"""Inspect proposed chronological cohorts; never fit models or overwrite processed data."""
from pathlib import Path
from datetime import datetime
import hashlib,json,subprocess
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]


def main():
    """Read frozen model/raw CSVs -> dated design counts, monthly profile and row assignments."""
    out=ROOT/'outputs/experiments'/datetime.now().strftime('temporal-design-%Y%m%d-%H%M%S')
    out.mkdir(parents=True,exist_ok=False)
    files=['data/processed/ml_data.csv','data/processed/merged_data.csv','data/raw/olist_order_reviews_dataset.csv']
    base=pd.read_csv(ROOT/files[0],usecols=['order_id','order_item_id','review_score'])
    base=base[base.review_score!=3]
    raw=pd.read_csv(ROOT/files[1],usecols=['order_id','order_item_id','review_score','order_purchase_timestamp','order_delivered_customer_date','review_answer_timestamp'])
    f=base.merge(raw,on=['order_id','order_item_id'],validate='one_to_one',suffixes=('_model',''))
    assert len(f)==len(base) and f.review_score_model.eq(f.review_score).all()
    dates=['order_purchase_timestamp','order_delivered_customer_date','review_answer_timestamp']
    for c in dates:f[c]=pd.to_datetime(f[c])
    assert not f[dates].isna().any().any()
    assert f.groupby('order_id')[dates+['review_score']].nunique().le(1).all().all()
    orders=f.drop_duplicates('order_id').copy()
    orders['risk']=orders.review_score.le(2)
    orders['available_proxy']=orders[dates[1:]].max(axis=1)
    orders['invalid_time']=orders.review_answer_timestamp.lt(orders.order_purchase_timestamp)
    orders['proposed_split']='outside_proposed_period'
    orders['eligible']=False
    summary=[]
    periods=[('train',None,'2018-01-01','2018-01-01'),('valid','2018-01-01','2018-04-01','2018-04-01'),('test','2018-04-01','2018-08-01','2018-11-01')]
    for name,start,end,freeze in periods:
        mask=orders.order_purchase_timestamp.lt(pd.Timestamp(end))
        if start:mask &= orders.order_purchase_timestamp.ge(pd.Timestamp(start))
        orders.loc[mask,'proposed_split']=name
        valid=mask & ~orders.invalid_time
        ready=valid & orders.available_proxy.lt(pd.Timestamp(freeze))
        orders.loc[ready,'eligible']=True
        ids=orders.loc[ready,'order_id']
        summary.append({'split':name,'purchase_start_inclusive':start,'purchase_end_exclusive':end,'available_before':freeze,
                        'orders_before_time_checks':int(mask.sum()),'invalid_time_orders':int((mask&orders.invalid_time).sum()),
                        'pending_orders':int((valid&~ready).sum()),'eligible_orders':int(ready.sum()),
                        'eligible_item_rows':int(f.order_id.isin(ids).sum()),'risk_orders':int(orders.loc[ready,'risk'].sum()),
                        'risk_order_rate':float(orders.loc[ready,'risk'].mean())})
    assert not orders.order_id.duplicated().any()
    monthly=orders.groupby(orders.order_purchase_timestamp.dt.to_period('M')).agg(orders=('order_id','size'),risk_rate=('risk','mean'))
    monthly.to_csv(out/'monthly_profile.csv')
    assignment=f[['order_id','order_item_id']].merge(orders[['order_id','proposed_split','eligible','invalid_time','available_proxy']],on='order_id',validate='many_to_one')
    assignment.to_csv(out/'proposed_assignments.csv',index=False)
    reviews=pd.read_csv(ROOT/files[2],usecols=['order_id','review_answer_timestamp'])
    reviews.review_answer_timestamp=pd.to_datetime(reviews.review_answer_timestamp)
    selected=reviews[reviews.order_id.isin(orders.order_id)]
    prior_counts={}
    for cutoff in ['2018-01-01','2018-04-01']:
        late=orders[(orders.order_purchase_timestamp<pd.Timestamp(cutoff))&(orders.review_answer_timestamp>=pd.Timestamp(cutoff))]
        prior_counts[cutoff]=int(selected[selected.order_id.isin(late.order_id)&selected.review_answer_timestamp.lt(pd.Timestamp(cutoff))].order_id.nunique())
    result={'status':'design_only_no_training','periods':summary,'model_rows':len(f),'model_orders':len(orders),
            'review_before_purchase_orders':int(orders.invalid_time.sum()),
            'review_before_delivery_orders':int(orders.review_answer_timestamp.lt(orders.order_delivered_customer_date).sum()),
            'multiple_review_orders':int(selected.groupby('order_id').size().gt(1).sum()),
            'latest_review_after_cutoff_with_prior_review':prior_counts,
            'latest_recorded_review':str(orders.review_answer_timestamp.max()),
            'assumption':'availability proxy is max(delivery timestamp, selected review answer timestamp); actual system ingestion and historical features unverified',
            'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'pandas_version':pd.__version__,
            'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files+['scripts/audit_temporal_split.py']}}
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(out);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
