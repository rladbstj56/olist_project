from pathlib import Path
import json,hashlib,subprocess,sys,platform
from datetime import datetime
import importlib.metadata
import numpy as np,pandas as pd
sys.path.insert(0,str(Path.cwd()))
from src.olist_delivery_models import *
from src.olist_imputation import RAW_INPUT_COLS
root=Path.cwd();out=root/'outputs/experiments'/datetime.now().strftime('ab-pipeline-verification-%Y%m%d-%H%M%S');out.mkdir(exist_ok=False,parents=True)
reference=root/'outputs/experiments/imputation-audit-20260930-220848'
old=pd.read_csv(reference/'predictions.csv',float_precision='round_trip');hist=pd.read_csv(root/'outputs/tables/track_a_vs_b_comparison.csv',float_precision='round_trip')
frame=prepare_corrected_model_frame(load_ml_data());s=split_train_valid_test_by_order(frame,frame.review_label,frame.order_id)
rows=[];predictions=[];checks=[]
for track,cols in [('A',TRACK_A_COLS),('B',PRE_ORDER_COLS)]:
 print('Training corrected pipeline',track,flush=True)
 model=build_corrected_track_pipeline(frame,track);model.fit(s[0][cols+RAW_INPUT_COLS],s[3]);valid=model.predict_proba(s[1][cols+RAW_INPUT_COLS])[:,1]
 threshold=select_positive_threshold(s[4],valid);fixed=float(hist.loc[hist.track.str.startswith('Track '+track),'positive_threshold'].iloc[0])
 for name,j,proba in [('valid',1,valid),('test',2,model.predict_proba(s[2][cols+RAW_INPUT_COLS])[:,1])]:
  expected=old[(old.track==track)&(old.variant=='train_only')&(old.split==name)].set_index('row_id').loc[s[j].index]
  delta=float(np.max(np.abs(proba-expected.positive_probability.to_numpy())))
  assert np.allclose(proba,expected.positive_probability,rtol=0,atol=1e-12),(track,name,delta)
  checks.append({'track':track,'split':name,'max_probability_difference_from_audit':delta})
  for rule,th in [('historical_fixed',fixed),('validation_selected',threshold)]:
   metric={'track':track,'split':name,'threshold_rule':rule,**risk_classification_metrics(s[j+3],proba,th)};rows.append(metric)
  p=s[j][['order_id','order_item_id']].copy();p.insert(0,'row_id',s[j].index);p['track']=track;p['split']=name;p['review_label']=s[j+3];p['positive_probability']=proba;p['threshold_validation_selected']=threshold;predictions.append(p)
pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False);pd.concat(predictions).to_csv(out/'predictions.csv',index=False)
tracked=['src/olist_delivery_models.py','src/olist_imputation.py','notebooks/03_ml_classifier.ipynb','tests/test_training_imputation.py']
manifest={'status':'complete','timestamp':datetime.now().astimezone().isoformat(),'git_head_before_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'python':platform.python_version(),'packages':{x:importlib.metadata.version(x) for x in ['pandas','numpy','scikit-learn','lightgbm']},'checks':checks,'code_sha256':{f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in tracked},'reference_manifest_sha256':hashlib.sha256((reference/'manifest.json').read_bytes()).hexdigest()}
(out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');print('VERIFIED',out);print(pd.DataFrame(rows).to_string(index=False))
