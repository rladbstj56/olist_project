"""Regression checks for train-only imputation and A/B routing."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from src import olist_delivery_models as models
from src.olist_imputation import (
    RAW_FIELDS, RAW_INPUT_COLS, COORDS, SIZES, TrainOnlyImputer, attach_raw_imputation_columns,
)


def fixture():
    """Six order/item rows with ties, unknown cities and distinct fold medians."""
    base = pd.DataFrame({c: [1.] * 6 for c in models.PRE_ORDER_COLS})
    base['order_id'] = ['a','b','c','d','e','f']
    base['order_item_id'] = 1
    base['distance_km'] = np.nan
    base['distance_cat'] = 'Mid-Haul'
    raw = pd.DataFrame({
        'order_id':base.order_id, 'order_item_id':1,
        'customer_city':['tie','tie','empty','other','other','unseen'],
        'seller_city':['s']*6, 'customer_lat':[1.,2.,np.nan,10.,11.,np.nan],
        'customer_lng':[1.,2.,np.nan,10.,11.,np.nan],
        'seller_lat':[0.]*6, 'seller_lng':[0.]*6,
        **{c:[1.,2.,3.,4.,1000.,2000.] for c in SIZES},
    })
    return base,raw


class ImputationUnitTests(unittest.TestCase):
    def test_join_preserves_order_and_rejects_missing_or_duplicate_keys(self):
        base,raw=fixture();base=base.iloc[[2,0,1]].copy();base.index=[20,5,9]
        joined=attach_raw_imputation_columns(base,raw)
        self.assertEqual(list(joined.index),[20,5,9])
        self.assertEqual(joined['raw__product_weight_g'].tolist(),[3.,1.,2.])
        with self.assertRaises(ValueError):attach_raw_imputation_columns(base,raw.iloc[1:])
        with self.assertRaises(ValueError):attach_raw_imputation_columns(base,pd.concat([raw,raw.iloc[[0]]]))

    def test_unknown_city_ties_and_missing_distance(self):
        base,raw=fixture();X=attach_raw_imputation_columns(base,raw)
        transformer=TrainOnlyImputer(tuple(models.PRE_ORDER_COLS)).fit(X.iloc[:3])
        self.assertEqual(transformer.coordinate_modes_['customer_lat']['tie'],1.)
        probe=X.iloc[[2,5]].copy();probe.loc[:,'raw__product_weight_g']=np.nan
        result=transformer.transform(probe)
        self.assertTrue(result.distance_km.eq(transformer.distance_median_).all())
        self.assertTrue(result.product_weight_g.eq(2.).all())
        self.assertEqual(list(result.columns),models.PRE_ORDER_COLS)
        self.assertFalse(any(c.startswith('raw__') for c in result.columns))

    def test_holdout_changes_do_not_change_training_statistics(self):
        base,raw=fixture();X=attach_raw_imputation_columns(base,raw)
        original=TrainOnlyImputer(tuple(models.PRE_ORDER_COLS)).fit(X.iloc[:3])
        changed=X.copy();changed.loc[3:,['raw__'+c for c in COORDS+SIZES]]=999.
        refit=clone(original).fit(changed.iloc[:3])
        self.assertEqual(original.coordinate_modes_,refit.coordinate_modes_)
        self.assertEqual(original.product_medians_,refit.product_medians_)
        self.assertEqual(original.distance_median_,refit.distance_median_)
        pd.testing.assert_frame_equal(original.transform(X.iloc[:3]),refit.transform(changed.iloc[:3]))

    def test_cross_validation_fits_each_fold_separately(self):
        base,raw=fixture();X=attach_raw_imputation_columns(base,raw)
        pipe=Pipeline([('impute',TrainOnlyImputer(tuple(models.PRE_ORDER_COLS))),
                       ('encode',models.make_preprocessor(base[models.PRE_ORDER_COLS])),
                       ('classifier',DummyClassifier(strategy='most_frequent'))])
        cv=[(np.array([0,1,2]),np.array([3,4,5])),(np.array([3,4,5]),np.array([0,1,2]))]
        result=cross_validate(pipe,X,np.array([0,1,0,1,0,1]),cv=cv,return_estimator=True,error_score='raise')
        medians=[p.named_steps['impute'].product_medians_['product_weight_g'] for p in result['estimator']]
        self.assertEqual(medians,[2.,1000.])
        self.assertFalse(hasattr(pipe.named_steps['impute'],'product_medians_'))


class ImputationDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source=models.load_ml_data()
        cls.raw=pd.read_csv('data/processed/merged_data.csv')
        cls.frame=models.prepare_corrected_model_frame(cls.source,raw_data=cls.raw)
        cls.base=models.prepare_labeled_model_frame(cls.source)
        cls.parts=models.split_train_valid_test_by_order(cls.frame,cls.frame.review_label,cls.frame.order_id)

    def test_full_features_match_preserved_audit_implementation(self):
        from scripts.audit_imputation_comparison import fit_imputation,transform_imputation,make_candidate
        raw=self.frame[RAW_INPUT_COLS].rename(columns=dict(zip(RAW_INPUT_COLS,RAW_FIELDS)))
        stats=fit_imputation(raw.loc[self.parts[0].index])
        expected=make_candidate(self.base,transform_imputation(raw,stats))
        for track,cols in [('A',models.TRACK_A_COLS),('B',models.PRE_ORDER_COLS)]:
            transformer=models.build_corrected_track_pipeline(self.frame,track).named_steps['train_only_imputation']
            transformer.fit(self.parts[0])
            pd.testing.assert_frame_equal(transformer.transform(self.frame),expected[cols])

    def test_public_evaluators_use_corrected_pipeline(self):
        original=models.build_corrected_track_pipeline
        seen=[]
        def fast_builder(X,track):
            pipeline=original(X,track)
            pipeline.set_params(classifier=DummyClassifier(strategy='most_frequent'))
            seen.append((track,pipeline))
            return pipeline
        with patch.object(models,'build_corrected_track_pipeline',side_effect=fast_builder):
            table=models.evaluate_track_a_vs_b(self.source,raw_data=self.raw)
            single=models.evaluate_track_b(self.source,raw_data=self.raw)
        self.assertEqual([x[0] for x in seen],['A','B','B'])
        self.assertEqual(table.test_rows.tolist(),[19985]*3)
        self.assertEqual(single['test_orders'],17575)
        for _,pipeline in seen:
            self.assertAlmostEqual(pipeline.named_steps['train_only_imputation'].distance_median_,428.20576234972293)


if __name__=='__main__':
    unittest.main()
