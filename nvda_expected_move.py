"""Compare NVDA market-implied ±1 sigma with HAR-RV expected moves.

Expected move is a symmetric *linear approximation* S * sigma * sqrt(T).
It is NOT a directional forecast, a hard bound or a barrier-touch probability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compare_expected_moves(iv_frame: pd.DataFrame, iv_meta: dict, signal: dict) -> tuple[pd.DataFrame, dict]:
    if not signal.get("ok"):
        raise ValueError("Prévisions HAR-RV indisponibles.")
    spot = float(signal["nvda_spot"])
    if not np.isfinite(spot) or spot <= 0:
        raise ValueError("Spot NVDA invalide.")

    har = signal.get("nvda_har_rv_forecasts", {})
    long_run = float(signal.get("nvda_har_long_run_vol_pct", np.nan))
    if not all(h in har for h in range(1, 6)):
        raise ValueError("Chemin HAR-RV J+1 à J+5 incomplet.")
    if not np.isfinite(long_run) or long_run <= 0:
        raise ValueError("Volatilité HAR-RV long terme indisponible.")

    now = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    origin = np.datetime64(now.date(), "D")
    rows = []
    for record in iv_frame.to_dict("records"):
        expiry = pd.Timestamp(record["Echeance"])
        cal_days = max(0, int((expiry - now).days))
        trading_days = int(np.busday_count(origin, np.datetime64(expiry.date(), "D")))
        iv = float(record["IV_ATM_pct"])
        if cal_days <= 0 or trading_days <= 0 or not (np.isfinite(iv) and iv > 0):
            continue

        # Market-IV convention: 365 calendar days to expiration.
        implied_move_pct = iv * np.sqrt(cal_days / 365.0)
        # HAR variance aggregation: daily trading-day vol, 252 business days/year.
        variances = [
            (float(har[h]) / 100.0) ** 2 / 252.0
            for h in range(1, min(trading_days, 5) + 1)
        ]
        if trading_days > 5:
            variances += [(long_run / 100.0) ** 2 / 252.0] * (trading_days - 5)
        har_move_pct = np.sqrt(sum(variances)) * 100.0

        implied_usd = spot * implied_move_pct / 100.0
        har_usd = spot * har_move_pct / 100.0
        rows.append({
            "Échéance": expiry.strftime("%Y-%m-%d"),
            "Séances": trading_days,
            "IV ATM (%)": round(iv, 2),
            "IV ± (%)": round(implied_move_pct, 2),
            "IV ± ($)": round(implied_usd, 2),
            "IV bas ($)": round(max(0.0, spot - implied_usd), 2),
            "IV haut ($)": round(spot + implied_usd, 2),
            "HAR ± (%)": round(har_move_pct, 2),
            "HAR ± ($)": round(har_usd, 2),
            "HAR bas ($)": round(max(0.0, spot - har_usd), 2),
            "HAR haut ($)": round(spot + har_usd, 2),
            "Écart IV−HAR (pts)": round(implied_move_pct - har_move_pct, 2),
            "Fiabilité IV": record.get("Qualite", "Non précisée"),
        })
    if not rows:
        raise ValueError("Aucune échéance future commune à l'IV et HAR.")
    return pd.DataFrame(rows), {
        "spot": spot,
        "spot_source": signal.get("nvda_spot_source", "inconnu"),
        "spot_time": str(signal.get("nvda_spot_time", "inconnue")),
        "iv_source": iv_meta.get("source", "inconnue"),
        "iv_date": iv_meta.get("spot_date", "inconnue"),
    }
