"""Check subgroup denominators and undefined metrics without model fitting."""
import unittest
import pandas as pd
from scripts.audit_order_capacity_groups import group_metrics


class CapacityGroupTests(unittest.TestCase):
    def test_global_selection_and_denominators(self):
        y=pd.Series([0,1,1,0,1])
        selected=pd.Series([True,True,False,False,True])
        groups=pd.Series(['a','a','a','b','b'])
        rows={r['group']:r for r in group_metrics(y,selected,groups)}
        self.assertEqual(rows['a']['tp'],1)
        self.assertEqual(rows['a']['fp'],1)
        self.assertEqual(rows['a']['fpr'],.5)
        self.assertEqual(rows['a']['recall'],1)
        self.assertEqual(rows['a']['precision'],.5)
        self.assertEqual(rows['a']['share_of_all_selected'],2/3)
        self.assertEqual(rows['b']['fpr'],1)
        self.assertEqual(sum(r['selected_orders'] for r in rows.values()),3)

    def test_undefined_rates_remain_missing(self):
        rows=group_metrics(pd.Series([0,1]),pd.Series([False,False]),pd.Series(['a','b']))
        self.assertTrue(pd.isna(rows[0]['precision']))
        self.assertTrue(pd.isna(rows[0]['fpr']))
        self.assertTrue(pd.isna(rows[1]['recall']))
        self.assertEqual(rows[1]['fpr'],0)


if __name__=='__main__':unittest.main()
