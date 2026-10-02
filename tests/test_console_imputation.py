"""Console input contract and Track C routing regression checks."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from src import olist_delivery_models as m
from src.olist_imputation import distance_category


class ConsoleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = m.load_ml_data()
        cls.raw = pd.read_csv('data/processed/merged_data.csv')

    def test_distance_boundaries(self):
        values = pd.Series([0, 50, 50.001, 250, 250.001, 750, 750.001, 1500, 1500.001])
        self.assertEqual(list(distance_category(values)), [
            'Urban/Last-Mile', 'Urban/Last-Mile', 'Short-Haul', 'Short-Haul',
            'Mid-Haul', 'Mid-Haul', 'Long-Haul', 'Long-Haul', 'Continental'])

    def test_console_and_manual_inputs(self):
        original = m.build_corrected_track_pipeline
        seen = []
        def fast(X, track, **kwargs):
            pipe = original(X, track, **kwargs)
            if track == 'B': pipe.set_params(classifier=DummyClassifier(strategy='prior'))
            else: pipe.set_params(regressor=DummyRegressor(strategy='mean'))
            seen.append((track, pipe))
            return pipe
        with patch.object(m, 'build_corrected_track_pipeline', side_effect=fast):
            console = m.train_console_artifacts(self.data, raw_data=self.raw)
            table = m.evaluate_track_c_quantiles(self.data, quantiles=(.8,.9,.95), raw_data=self.raw)
            examples = m.make_recommendation_examples(self.data, raw_data=self.raw, high_risk_only=False)
            full = m.train_models(self.data, raw_data=self.raw)
        self.assertEqual([t for t,_ in seen], ['B','C','C','C','C','B','C','B','C'])
        self.assertTrue({"customer_state", "seller_state"}.issubset(console.test_orders.columns))
        self.assertEqual(len(console.test_orders), 19985)
        self.assertEqual(console.test_orders.order_id.nunique(), 17575)
        self.assertFalse(console.test_orders.duplicated(['order_id','order_item_id']).any())
        self.assertFalse(console.high_risk_orders.order_id.duplicated().any())
        self.assertEqual(len(table),3)
        self.assertFalse(any(c.startswith('raw__') for c in examples))
        self.assertAlmostEqual(seen[0][1][0].distance_median_,428.9974892020914)
        self.assertAlmostEqual(seen[2][1][0].distance_median_,431.950087901222)
        row=console.test_orders.iloc[[0]][m.PRE_ORDER_COLS].copy()
        row['distance_km']=60.; row['distance_cat']='Continental'; row['product_weight_g']=np.nan
        transform=console.models.track_b[0]
        corrected=transform.transform_prepared_features(row)
        self.assertEqual(corrected.distance_cat.iloc[0],'Short-Haul')
        self.assertEqual(corrected.distance_km.iloc[0],60.)
        self.assertEqual(corrected.product_weight_g.iloc[0],transform.product_medians_['product_weight_g'])
        self.assertTrue(np.isfinite(m.predict_order(console.models,row)['predicted_delivery_days']))
        self.assertTrue(np.isfinite(m.predict_order(full,row)['review_risk_probability']))
        self.assertEqual(m.risk_level_from_probability(.46, risk_threshold=.46),'주의')
        self.assertEqual(m.risk_level_from_probability(m.TRACK_B_RISK_THRESHOLD),'주의')
        self.assertEqual(m.risk_level_from_probability(m.TRACK_B_RISK_THRESHOLD + 1e-6),'고위험')
        from streamlit.testing.v1 import AppTest
        with patch.object(m, 'train_console_artifacts', return_value=console):
            app=AppTest.from_file('streamlit_app.py',default_timeout=30).run()
            self.assertEqual(len(app.exception),0)
            self.assertIn('아이템',app.selectbox[0].options[0])
            distance=next(x for x in app.number_input if x.label=='고객-셀러 거리(km)')
            distance.set_value(60.).run()
            self.assertEqual(len(app.exception),0)
            self.assertTrue(any('Short-Haul' in x.value for x in app.caption))
            app.selectbox[0].select(1).run()
            self.assertEqual(len(app.exception),0)
            self.assertEqual(app.selectbox[0].value,1)



if __name__=='__main__': unittest.main()
