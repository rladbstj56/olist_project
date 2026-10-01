"""Isolated, fixed-condition imputation audit. Existing artifacts are read-only."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from src.olist_delivery_models import (
    DEFAULT_LABEL_POLICY, PRE_ORDER_COLS, TRACK_A_COLS,
    add_post_delivery_features, build_track_a_pipeline, build_track_b_pipeline,
    prepare_labeled_model_frame, risk_classification_metrics,
    split_train_valid_test_by_order,
)

KEYS = ['order_id', 'order_item_id']
SIZES = ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']
COORDS = ['customer_lat', 'customer_lng', 'seller_lat', 'seller_lng']
ALLOWED = set(SIZES + ['distance_km', 'distance_cat', 'delivery_speed', 'day_per_km', 'delivery_distance'])
METRICS = ['accuracy', 'balanced_acc', 'risk_precision', 'risk_recall', 'risk_pr_auc', 'flagged_rate', 'macro_f1']


def distance(frame: pd.DataFrame) -> pd.Series:
    """Four coordinate columns in degrees -> existing Haversine distance in km."""
    lat1, lon1, lat2, lon2 = map(np.radians, [frame[c] for c in COORDS])
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def transform_imputation(raw: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Apply fixed training statistics to raw rows; never estimate from these rows."""
    out = raw.copy()
    for c in COORDS:
        city = c.split('_')[0] + '_city'
        out[c] = out[c].fillna(out[city].map(stats['coordinate_modes'][c]))
    for c in SIZES:
        out[c] = out[c].fillna(stats['product_medians'][c])
    out['distance_km'] = distance(out)
    if 'distance_median' in stats:
        out['distance_km'] = out['distance_km'].fillna(stats['distance_median'])
    return out


def fit_imputation(train_raw: pd.DataFrame) -> dict:
    """Only training rows -> coordinate modes and product/distance medians."""
    stats = {'coordinate_modes': {}, 'product_medians': {c: float(train_raw[c].median()) for c in SIZES}}
    for c in COORDS:
        city = c.split('_')[0] + '_city'
        modes = train_raw.groupby(city)[c].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        stats['coordinate_modes'][c] = modes.dropna().to_dict()
    stats['distance_median'] = float(transform_imputation(train_raw, stats).distance_km.median())
    assert np.isfinite(stats['distance_median'])
    assert all(np.isfinite(v) for v in stats['product_medians'].values())
    return stats


def make_candidate(base: pd.DataFrame, transformed: pd.DataFrame) -> pd.DataFrame:
    """Baseline model frame plus transformed raw rows -> aligned candidate frame."""
    out = base.copy()
    out[SIZES] = transformed[SIZES]
    new_distance = transformed.distance_km.copy()
    # Preserve serialized values when the reconstructed distance differs only by roundoff.
    unchanged = np.isclose(new_distance, base.distance_km, rtol=1e-12, atol=1e-9)
    new_distance.loc[unchanged] = base.loc[unchanged, 'distance_km']
    out['distance_km'] = new_distance
    out['distance_cat'] = np.select(
        [new_distance <= 50, new_distance <= 250, new_distance <= 750, new_distance <= 1500],
        ['Urban/Last-Mile', 'Short-Haul', 'Mid-Haul', 'Long-Haul'], default='Continental',
    )
    out = add_post_delivery_features(out)
    for c in out:
        out[c] = out[c].astype(base[c].dtype)
    pd.testing.assert_frame_equal(out[[c for c in base if c not in ALLOWED]], base[[c for c in base if c not in ALLOWED]])
    assert out.index.equals(base.index) and out.columns.equals(base.columns)
    return out


def check_imputation(raw: pd.DataFrame, train_index: pd.Index, stats: dict) -> dict:
    """Check edge cases and prove holdout perturbations cannot change training preprocessing."""
    toy = pd.DataFrame({
        'customer_city': ['tie', 'tie', 'empty'], 'seller_city': ['s', 's', 's'],
        'customer_lat': [1., 2., np.nan], 'customer_lng': [1., 2., np.nan],
        'seller_lat': [0., 0., 0.], 'seller_lng': [0., 0., 0.],
        **{c: [1., 3., np.nan] for c in SIZES},
    })
    toy_stats = fit_imputation(toy)
    assert toy_stats['coordinate_modes']['customer_lat']['tie'] == 1.
    probe = toy.iloc[[2]].copy()
    probe['customer_city'] = 'unseen'
    probe['seller_lat'] = np.nan
    probe['customer_lng'] = np.nan
    applied = transform_imputation(probe, toy_stats)
    assert applied.customer_lat.isna().all()
    assert (applied.distance_km == toy_stats['distance_median']).all()
    assert (applied[SIZES] == 2.).all().all()
    empty_city = transform_imputation(toy.iloc[[2]], toy_stats)
    assert empty_city.customer_lat.isna().all()
    assert empty_city.distance_km.notna().all()
    changed = raw.copy()
    holdout = ~changed.index.isin(train_index)
    changed.loc[holdout, COORDS + SIZES] = 123456.
    changed.loc[holdout, ['customer_city', 'seller_city']] = 'holdout-only-perturbation'
    changed_stats = fit_imputation(changed.loc[train_index])
    assert stats == changed_stats
    pd.testing.assert_frame_equal(
        transform_imputation(raw, stats).loc[train_index],
        transform_imputation(changed, changed_stats).loc[train_index],
    )
    return {'unseen_city': True, 'all_missing_city': True, 'mode_tie': True,
            'distance_fallback': True, 'holdout_perturbation_train_invariance': True}


def measure(y: pd.Series, probability: np.ndarray, threshold: float) -> dict:
    """Labels, positive probabilities and fixed threshold -> item-row metrics and risk confusion counts."""
    assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
    row = risk_classification_metrics(y, probability, threshold)
    matrix = confusion_matrix(y, (probability >= threshold).astype(int), labels=[0, 1])
    row.update(dict(zip(['risk_tp', 'risk_fn', 'risk_fp', 'risk_tn'], map(int, matrix.ravel()))))
    return row


def sha256(path: Path) -> str:
    """File path -> digest for preservation and execution provenance."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Validate inputs, optionally train four isolated models, and write a new experiment directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only', action='store_true', help='Run preprocessing checks without fitting models or writing artifacts.')
    args = parser.parse_args()
    paths = [ROOT/'data/processed/ml_data.csv', ROOT/'data/processed/merged_data.csv',
             ROOT/'outputs/tables/track_a_vs_b_comparison.csv', ROOT/'src/olist_delivery_models.py', Path(__file__)]
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in paths}
    source = pd.read_csv(paths[0])
    raw_source = pd.read_csv(paths[1])
    historical = pd.read_csv(paths[2], float_precision='round_trip')
    assert not source.duplicated(KEYS).any() and not raw_source.duplicated(KEYS).any()
    base = prepare_labeled_model_frame(source)
    assert len(base) == 100107
    joined = base[KEYS].merge(raw_source[KEYS + ['customer_city', 'seller_city'] + COORDS + SIZES],
                              on=KEYS, how='left', sort=False, validate='one_to_one', indicator=True)
    assert joined['_merge'].eq('both').all()
    assert np.array_equal(joined[KEYS].to_numpy(), base[KEYS].to_numpy())
    joined = joined.drop(columns='_merge')
    joined.index = base.index
    split = split_train_valid_test_by_order(base[PRE_ORDER_COLS], base.review_label, base.order_id)
    indices = dict(zip(['train', 'valid', 'test'], [x.index for x in split[:3]]))
    assert [len(v) for v in indices.values()] == [64103, 16019, 19985]
    assert [base.loc[v].order_id.nunique() for v in indices.values()] == [56237, 14060, 17575]
    assert set().union(*[set(v) for v in indices.values()]) == set(base.index)
    names = list(indices)
    for i, name in enumerate(names):
        for other in names[i+1:]:
            assert not set(base.loc[indices[name]].order_id) & set(base.loc[indices[other]].order_id)
    stats = fit_imputation(joined.loc[indices['train']])
    checks = check_imputation(joined, indices['train'], stats)
    transformed = transform_imputation(joined, stats)
    candidate = make_candidate(base, transformed)
    assert candidate[TRACK_A_COLS].dtypes.equals(base[TRACK_A_COLS].dtypes)
    masks = pd.DataFrame({c: ~((base[c] == candidate[c]) | (base[c].isna() & candidate[c].isna())) for c in TRACK_A_COLS})
    audit = {'checks': checks, 'changed_by_feature': masks.sum().astype(int).to_dict(),
             'changed_by_split': {k: int(masks.loc[v].any(axis=1).sum()) for k,v in indices.items()},
             'distance_roundoff_policy': {'rtol': 1e-12, 'atol_km': 1e-9},
             'fitted_statistics': stats, 'imputation_counts': {}}
    for name, idx in indices.items():
        audit['imputation_counts'][name] = {
            c: {'original_missing': int(joined.loc[idx,c].isna().sum()),
                'filled': int((joined.loc[idx,c].isna() & transformed.loc[idx,c].notna()).sum())}
            for c in COORDS + SIZES}
        audit['imputation_counts'][name]['distance_fallback'] = int(distance(transformed.loc[idx]).isna().sum())
    print(json.dumps({'preflight': 'passed', 'changed_by_split': audit['changed_by_split'],
                      'distance_median': stats['distance_median']}, ensure_ascii=False), flush=True)
    if args.validate_only:
        return

    run_dir = ROOT/'outputs/experiments'/datetime.now().astimezone().strftime('imputation-audit-%Y%m%d-%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=False)
    assignments = base[KEYS + ['review_label']].copy()
    assignments.insert(0, 'row_id', base.index)
    for name, idx in indices.items():
        assignments.loc[idx, 'split'] = name
    assignments.to_csv(run_dir/'split_assignments.csv', index=False)
    (run_dir/'imputation_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    manifest = {'status':'running', 'started_at': datetime.now().astimezone().isoformat(),
                'python':platform.python_version(), 'platform':platform.platform(),
                'packages': {p:importlib.metadata.version(p) for p in ['pandas','numpy','scikit-learn','scipy','lightgbm']},
                'input_sha256':hashes, 'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                'git_status_before_training':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
                'label_policy':DEFAULT_LABEL_POLICY, 'split':{'random_state':42,'test_size':.2,'valid_fraction_of_remaining':.2},
                'models':{}, 'note':'Sensitivity comparison on previously inspected test set; no new generalization claim.'}
    (run_dir/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    rows, predictions = [], []
    for track, cols, builder in [('A',TRACK_A_COLS,build_track_a_pipeline),('B',PRE_ORDER_COLS,build_track_b_pipeline)]:
        saved = historical[historical.track.str.startswith('Track '+track)].iloc[0]
        threshold = float(saved.positive_threshold)
        for variant, frame in [('existing',base),('train_only',candidate)]:
            print(f'Training {track}/{variant}', flush=True)
            model = builder(frame.loc[indices['train'],cols])
            config = {'features':cols, 'positive_threshold':threshold, 'classifier':model.named_steps['classifier'].get_params()}
            if track in manifest['models']:
                assert config == manifest['models'][track]
            manifest['models'][track] = config
            model.fit(frame.loc[indices['train'],cols], frame.loc[indices['train'],'review_label'])
            assert list(model.named_steps['classifier'].classes_) == [0,1]
            for name in ['valid','test']:
                idx = indices[name]
                probability = model.predict_proba(frame.loc[idx,cols])[:,1]
                row = {'track':track,'variant':variant,'split':name,'rows':len(idx),'orders':frame.loc[idx].order_id.nunique()}
                row.update(measure(frame.loc[idx,'review_label'],probability,threshold));rows.append(row)
                prediction = assignments.loc[idx].copy()
                prediction['track']=track;prediction['variant']=variant
                prediction['positive_probability']=probability;prediction['risk_probability']=1-probability
                prediction['positive_threshold']=threshold;prediction['predicted_label']=(probability>=threshold).astype(int)
                prediction['feature_changed']=masks.loc[idx,cols].any(axis=1).to_numpy()
                predictions.append(prediction)
            print(f'Finished {track}/{variant}', flush=True)
    metrics=pd.DataFrame(rows);metrics.to_csv(run_dir/'metrics.csv',index=False)
    pd.concat(predictions).to_csv(run_dir/'predictions.csv',index=False)
    stored=pd.read_csv(run_dir/'predictions.csv',float_precision='round_trip')
    assert len(stored)==4*(16019+19985)
    for (track,variant,name), group in stored.groupby(['track','variant','split']):
        assert not group.row_id.duplicated().any()
        assert set(group.row_id)==set(indices[name])
        assert (group.predicted_label==(group.positive_probability>=group.positive_threshold).astype(int)).all()
        measured=measure(group.review_label,group.positive_probability.to_numpy(),float(group.positive_threshold.iloc[0]))
        row=metrics[(metrics.track==track)&(metrics.variant==variant)&(metrics.split==name)].iloc[0]
        for k,v in measured.items():
            assert np.isclose(v,row[k],rtol=0,atol=1e-12),(track,variant,name,k)
    changes=[]; history_deltas=[]
    report=['# 학습 영역 전용 결측치 처리 비교', '', '아이템 행 평가, 고정 임계값, 동일 환경의 전처리 민감도 비교입니다. AP는 average precision입니다.', '',
            '| 모델 | 조건 | 영역 | 정밀도 | 재현율 | AP | 균형 정확도 | 위험 TP/FN/FP/TN |', '|---|---|---|---:|---:|---:|---:|---|']
    for _,r in metrics.iterrows():
        report.append(f'| {r.track} | {r.variant} | {r.split} | {r.risk_precision:.6f} | {r.risk_recall:.6f} | {r.risk_pr_auc:.6f} | {r.balanced_acc:.6f} | {r.risk_tp}/{r.risk_fn}/{r.risk_fp}/{r.risk_tn} |')
    report+=['','## 같은 환경에서 실험군 − 비교군','', '| 모델 | 영역 | 정밀도 차이 | 재현율 차이 | AP 차이 | 예측 라벨 변경 행 |','|---|---|---:|---:|---:|---:|']
    for track in ['A','B']:
        for name in ['valid','test']:
            subset=metrics[(metrics.track==track)&(metrics.split==name)].set_index('variant')
            delta={k:float(subset.loc['train_only',k]-subset.loc['existing',k]) for k in METRICS}
            preds=stored[(stored.track==track)&(stored.split==name)].pivot(index='row_id',columns='variant',values='predicted_label')
            changed=int((preds.existing!=preds.train_only).sum());changes.append({'track':track,'split':name,'metric_delta':delta,'prediction_changes':changed})
            report.append(f'| {track} | {name} | {delta["risk_precision"]:+.6f} | {delta["risk_recall"]:+.6f} | {delta["risk_pr_auc"]:+.6f} | {changed} |')
        old=historical[historical.track.str.startswith('Track '+track)].iloc[0]
        new=metrics[(metrics.track==track)&(metrics.variant=='existing')&(metrics.split=='test')].iloc[0]
        history_deltas.append({'track':track,'new_existing_minus_historical':{k:float(new[k]-old[k]) for k in METRICS}})
    report+=['','## 과거 CSV와 새 비교군','', '차이 방향은 새 비교군 − 과거 CSV이며 전처리 효과가 아닙니다.', '', '```json',json.dumps(history_deltas,ensure_ascii=False,indent=2),'```', '',
             '## 검증과 한계','',f'- 직접 변경된 피처를 가진 행: {audit["changed_by_split"]}',
             '- 저장 확률로 지표·혼동행렬 재계산을 통과했습니다.','- 검증·테스트 교란 시 학습 대체 통계와 변환 결과 불변 검사를 통과했습니다.',
             '- 기존 표본 선택·지리 참조자료와 사전에 선택된 모델 설정은 유지했습니다. 테스트는 이미 열람한 표본이므로 신규 일반화 성능으로 해석하지 않습니다.',
             '- 기존 노트북·성능표·UI 모델은 교체하지 않았습니다.']
    (run_dir/'comparison.md').write_text('\n'.join(report)+'\n')
    assert hashes=={str(p.relative_to(ROOT)):sha256(p) for p in paths}
    manifest.update(status='complete',completed_at=datetime.now().astimezone().isoformat(),
                    prediction_metrics_verified=True,inputs_preserved=True,comparisons=changes,historical_comparison=history_deltas)
    manifest['output_sha256']={p.name:sha256(p) for p in run_dir.iterdir() if p.name!='manifest.json'}
    (run_dir/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('COMPLETE '+str(run_dir),flush=True)


if __name__ == '__main__':
    main()
