"""Smoke tests for Stability nearest-barrier delta hedge; no network requests."""
import unittest
import numpy as np
import pandas as pd
from stability_delta_hedge import _bs_greeks, propose_warrant_hedges


class HedgeTests(unittest.TestCase):
    def test_put_hedges_positive_delta_near_lower_barrier(self):
        priced = pd.DataFrame([{
            "Code":"TESTL", "Spot":168., "Barriere_basse":140.,
            "Barriere_haute":300., "Maturite":"2027-02-19",
            "Delta_MC_EUR_par_USD":0.12,
            "Gamma_MC_EUR_par_USD2":-0.002,
            "Vega_MC_EUR_par_1pp":-0.03,
            "Sigma_RMS_pct_ann":45.,
        }])
        warrants = pd.DataFrame([{
            "Code":"PUT1","Type":"PUT","Strike":140.,
            "Maturite":pd.Timestamp("2027-03-19"),"Parite":10.,
            "Ask":1., "Bid":0.9,
        }])
        out = propose_warrant_hedges(priced,warrants,quantity=100,eurusd=1.10)
        self.assertEqual(out.loc[0,"Barriere_proche"],"BASSE")
        self.assertEqual(out.loc[0,"Type_couverture"],"PUT")
        self.assertEqual(out.loc[0,"Action"],"ACHETER")
        self.assertAlmostEqual(out.loc[0,"Delta_net_EUR_USD"],0.,places=8)

    def test_call_hedges_negative_delta_near_upper_barrier(self):
        priced = pd.DataFrame([{
            "Code":"TESTU","Spot":168.,"Barriere_basse":100.,
            "Barriere_haute":180.,"Maturite":"2027-01-15",
            "Delta_MC_EUR_par_USD":-0.10,
            "Gamma_MC_EUR_par_USD2":-0.001,
            "Vega_MC_EUR_par_1pp":-0.02,
            "Sigma_RMS_pct_ann":45.,
        }])
        warrants = pd.DataFrame([{
            "Code":"CALL1","Type":"CALL","Strike":180.,
            "Maturite":pd.Timestamp("2027-02-19"),"Parite":10.,
            "Ask":1.,"Bid":0.9,
        }])
        out=propose_warrant_hedges(priced,warrants,quantity=100,eurusd=1.10)
        self.assertEqual(out.loc[0,"Type_couverture"],"CALL")
        self.assertEqual(out.loc[0,"Action"],"ACHETER")
        self.assertAlmostEqual(out.loc[0,"Delta_net_EUR_USD"],0.,places=8)

    def test_missing_warrants_never_claims_hedge(self):
        priced=pd.DataFrame([{
            "Code":"TEST","Spot":168.,"Barriere_basse":140.,
            "Barriere_haute":300.,"Maturite":"2027-02-19",
            "Delta_MC_EUR_par_USD":0.12,
            "Gamma_MC_EUR_par_USD2":-0.002,
            "Vega_MC_EUR_par_1pp":-0.03,
            "Sigma_RMS_pct_ann":45.,
        }])
        out=propose_warrant_hedges(priced,pd.DataFrame(),quantity=100)
        self.assertEqual(out.loc[0,"Action"],"INDISPONIBLE")
        self.assertTrue(np.isnan(out.loc[0,"Quantite_warrants"]))


if __name__ == "__main__":
    unittest.main()
