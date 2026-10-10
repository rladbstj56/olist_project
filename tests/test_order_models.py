"""Inference boundary and schema tests without classifier fitting."""
import unittest
import numpy as np
from test_order_features import fixture
from src.olist_order_features import OrderFeatureTransformer
from src.olist_order_models import predict_order_model
from scripts.audit_order_models import capacity_metrics
import pandas as pd


class FixedClassifier:
    classes_ = np.array([0,1])

    def predict_proba(self, features):
        """Order features -> fixed scores covering the strict threshold boundary."""
        p = np.array([.49,.50,.51,.2,.8,.9])[:len(features)]
        return np.column_stack([1-p,p])


class OrderPredictionTests(unittest.TestCase):
    def test_threshold_and_count_contract(self):
        items = fixture()
        transformer = OrderFeatureTransformer().fit(items)
        for with_count in [False,True]:
            columns = list(transformer.get_feature_names_out())+(['item_count'] if with_count else [])
            bundle = dict(transformer=transformer,model=FixedClassifier(),include_item_count=with_count,
                          feature_columns=columns,positive_threshold=.5)
            result = predict_order_model(bundle,items)
            self.assertEqual(result.risk_flag.tolist(),[True,False,False,True,False,False])
            np.testing.assert_array_equal(result.risk_score,1-result.positive_probability)
            bundle['feature_columns'] = columns[::-1]
            with self.assertRaises(ValueError): predict_order_model(bundle,items)

    def test_capacity_tie_range(self):
        scores = pd.DataFrame({'review_label':[0,1,0,1],'model':[.1,.1,.1,.9]},index=pd.Index(['a','b','c','d'],name='order_id'))
        rows = capacity_metrics(scores,'model')
        self.assertEqual(rows[0]['reviewed_orders'],1)
        self.assertEqual(rows[0]['tp'],1)
        self.assertEqual(rows[0]['tp_min_with_ties'],0)
        self.assertEqual(rows[0]['tp_max_with_ties'],1)
        self.assertAlmostEqual(rows[0]['tp_expected_random_tie'],2/3)


if __name__ == '__main__': unittest.main()
