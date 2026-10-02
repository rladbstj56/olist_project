"""Run from repository root with the local --console-cache produced by the audit script."""
from pathlib import Path
import sys,json
from unittest.mock import patch
import joblib
import numpy as np
from streamlit.testing.v1 import AppTest
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from src import olist_delivery_models as m
cached=joblib.load(sys.argv[1])
# Reuse fitted models to check final output assembly without another training run.
with patch.object(m,'build_corrected_track_pipeline',side_effect=lambda X,t,**kw: cached.models.track_b if t=='B' else cached.models.track_c), \
     patch.object(cached.models.track_b,'fit',return_value=cached.models.track_b), \
     patch.object(cached.models.track_c,'fit',return_value=cached.models.track_c):
    console=m.train_console_artifacts(m.load_ml_data(ROOT/'data/processed/ml_data.csv'))
np.testing.assert_allclose(console.test_orders.review_risk_probability,cached.test_orders.review_risk_probability,rtol=0,atol=1e-12)
assert {'customer_state','seller_state'}.issubset(console.test_orders.columns)
with patch.object(m,'train_console_artifacts',return_value=console):
    app=AppTest.from_file(str(ROOT/'streamlit_app.py'),default_timeout=60).run()
    assert not app.exception
    initial_metrics=[{'label':x.label,'value':x.value} for x in app.metric]
    next(x for x in app.number_input if x.label=='고객-셀러 거리(km)').set_value(60.).run()
    assert not app.exception
    assert any('Short-Haul' in x.value for x in app.caption)
    app.selectbox[0].select(1).run()
    assert not app.exception and app.selectbox[0].value==1
report={'streamlit_execution':'passed','distance_input_60km':'Short-Haul','item_selection':'passed',
        'final_output_assembly_reused_fitted_models':'passed','initial_metrics':initial_metrics,
        'browser_visual_check':'not_verified: computer-use Accessibility/Screen Recording permission pending',
        'model_retraining_during_ui_check':False,'scope':'Default 90% quantile; existing fitted models reused. Quantile slider retraining not exercised.'}
Path(__file__).with_name('ui_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
