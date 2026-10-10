"""Order input contract checks; no classification model is trained."""
import pickle
import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone
from src.olist_imputation import COORDS, SIZES
from src.olist_order_features import ITEM_FEATURES, SOURCE_COMMON, OrderFeatureTransformer, make_order_labels


def fixture():
    """Eight items in six orders -> known sums, population deviations, category mixes and raw missingness."""
    f = pd.DataFrame({c: [1.] * 8 for c in ITEM_FEATURES})
    f['order_id'] = ['a', 'a', 'a', 'b', 'c', 'd', 'e', 'f']
    f['order_item_id'] = [1, 2, 3, 1, 1, 1, 1, 1]
    f['product_id'] = ['p1', 'p2', 'p1', 'p3', 'p4', 'p5', 'p6', 'p7']
    f['seller_id'] = ['s1', 's2', 's1', 's1', 's1', 's1', 's1', 's1']
    f['main_category'] = ['A', 'B', 'A', 'C', 'D', 'E', 'F', 'A']
    f['price'] = [10., 20., 30., 5., 7., 9., 11., 13.]
    f['freight_value'] = [1., 2., 3., 1., 1., 1., 1., 1.]
    f['cross_state'] = [0, 1, 0, 0, 0, 0, 0, 0]
    f['is_sp_seller'] = [1, 0, 1, 1, 1, 1, 1, 1]
    f['distance_km'] = np.nan
    f['review_score'] = [1, 1, 1, 4, 2, 5, 4, 5]
    f['review_label'] = f.review_score.ge(4).astype(int)
    for field in SOURCE_COMMON:
        f[field] = 'consistent'
    for city in ['customer_city', 'seller_city']:
        f['raw__' + city] = 'known_city'
    for field in COORDS:
        f['raw__' + field] = 0. if field.startswith('customer') else 1.
    for field in SIZES:
        f['raw__' + field] = [10., 20., np.nan, 40., 50., 60., 70., 80.]
    return f


class OrderFeatureTests(unittest.TestCase):
    def setUp(self):
        self.items = fixture()
        self.transformer = OrderFeatureTransformer().fit(self.items)

    def test_known_aggregates_and_labels(self):
        out = self.transformer.transform(self.items)
        self.assertEqual(out.shape, (6, 39))
        self.assertEqual(out.index.name, 'order_id')
        self.assertNotIn('order_item_id', out)
        self.assertNotIn('item_count', out)
        self.assertNotIn('review_label', out)
        a = out.loc['a']
        self.assertEqual(a.price_sum, 60)
        self.assertEqual(a.price_mean, 20)
        self.assertAlmostEqual(a.price_std, np.std([10., 20., 30.], ddof=0))
        self.assertAlmostEqual(a.freight_to_price_ratio, .1)
        self.assertEqual(a.distinct_product_count, 2)
        self.assertEqual(a.distinct_seller_count, 2)
        self.assertAlmostEqual(a.weight_mean, (10+20+50)/3)
        self.assertAlmostEqual(a.missing_weight_share, 1/3)
        self.assertAlmostEqual(a.cross_state_share, 1/3)
        self.assertAlmostEqual(a.seller_sp_share, 2/3)
        self.assertAlmostEqual(a[self.transformer.category_columns_['A']], 2/3)
        self.assertTrue(np.allclose(out.filter(like='category_share').sum(axis=1), 1))
        self.assertEqual(out.loc['b', 'price_std'], 0)
        pd.testing.assert_series_equal(make_order_labels(self.items),
            pd.Series([0, 1, 0, 1, 1, 1], index=pd.Index(list('abcdef'), name='order_id'), name='review_label'))

    def test_shuffle_renumber_reidentify_and_target_independence(self):
        expected = self.transformer.transform(self.items)
        changed = self.items.sample(frac=1, random_state=7).copy()
        changed['order_item_id'] = changed.groupby('order_id').cumcount()+101
        for col in ['product_id', 'seller_id']:
            changed[col] = changed[col].map({v: 'renamed_'+v for v in changed[col].unique()})
        changed['review_score'] = 999
        changed['review_label'] = 999
        pd.testing.assert_frame_equal(self.transformer.transform(changed), expected, check_exact=True)
        pd.testing.assert_frame_equal(OrderFeatureTransformer().fit(self.items.iloc[::-1]).transform(self.items), expected, check_exact=True)
        sample = self.items[self.items.order_id.eq('a')]
        pd.testing.assert_frame_equal(self.transformer.transform(sample), expected.loc[['a']], check_exact=True)

    def test_holdout_missing_unknown_and_fit_state_unchanged(self):
        state = pickle.dumps(self.transformer)
        holdout = self.items[self.items.order_id.eq('a')].copy()
        holdout['main_category'] = ['new_category', None, 'A']
        for field in SIZES:
            holdout['raw__'+field] = np.nan
        for field in COORDS:
            holdout['raw__'+field] = np.nan
        holdout['raw__customer_city'] = 'unseen'
        holdout['raw__seller_city'] = 'unseen'
        out = self.transformer.transform(holdout)
        self.assertAlmostEqual(out.category_share_unknown.iloc[0], 2/3)
        self.assertEqual(out.missing_distance_share.iloc[0], 1)
        self.assertEqual(out.missing_weight_share.iloc[0], 1)
        self.assertEqual(out.weight_mean.iloc[0], 50)
        self.assertAlmostEqual(out.distance_mean.iloc[0], self.transformer.item_imputer_.distance_median_)
        self.assertEqual(state, pickle.dumps(self.transformer))
        combined = pd.concat([self.items, holdout.assign(order_id='holdout')], ignore_index=True)
        refit = clone(self.transformer).fit(combined[combined.order_id.ne('holdout')])
        pd.testing.assert_frame_equal(refit.transform(holdout), out, check_exact=True)

    def test_zero_total_uses_training_order_median(self):
        probe = self.items[self.items.order_id.eq('b')].copy()
        probe['price'] = 0.
        out = self.transformer.transform(probe)
        self.assertEqual(out.freight_to_price_ratio.iloc[0], self.transformer.order_medians_.freight_to_price_ratio)
        zero_training = self.items.copy(); zero_training['price'] = 0.
        with self.assertRaisesRegex(ValueError, 'finite training-order median'):
            OrderFeatureTransformer().fit(zero_training)

    def test_invalid_keys_shared_values_and_labels_are_rejected(self):
        duplicate = pd.concat([self.items, self.items.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, 'unique order/item'):
            self.transformer.transform(duplicate)
        for col, value in [('expected_delivery_days', 99), (SOURCE_COMMON[0], 'different')]:
            broken = self.items.copy(); broken.loc[0, col] = value
            with self.assertRaisesRegex(ValueError, 'Shared order attributes'):
                self.transformer.transform(broken)
        missing = self.items.copy(); missing.loc[0, 'seller_id'] = None
        with self.assertRaisesRegex(ValueError, 'identifiers cannot be missing'):
            self.transformer.transform(missing)
        broken = self.items.copy(); broken.loc[0, 'review_score'] = 2
        with self.assertRaisesRegex(ValueError, 'Review scores disagree'):
            make_order_labels(broken)
        broken = self.items.copy(); broken.loc[0, 'review_label'] = 1
        with self.assertRaisesRegex(ValueError, 'Stored review label'):
            make_order_labels(broken)

    def test_shared_calendar_strings_remain_categorical(self):
        items = self.items.copy()
        items['order_purchase_dayofweek'] = 'wednesday'
        items['order_purchase_month'] = 'january'
        out = OrderFeatureTransformer().fit_transform(items)
        self.assertTrue(out.purchase_dayofweek.eq('wednesday').all())
        self.assertTrue(out.purchase_month.eq('january').all())
        self.assertEqual(out.shape, (6, 39))

    def test_serialization_restores_features_without_refitting(self):
        restored = pickle.loads(pickle.dumps(self.transformer))
        pd.testing.assert_frame_equal(restored.transform(self.items), self.transformer.transform(self.items), check_exact=True)
        self.assertEqual(len(restored.get_feature_names_out()), 39)


if __name__ == '__main__':
    unittest.main()
