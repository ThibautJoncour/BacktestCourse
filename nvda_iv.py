"""NVDA ATM implied-volatility term structure from public Yahoo option chains.

This is an observed market term structure, NOT a forecast of future volatility.
Data may be delayed, incomplete, or unavailable; no artificial fallback values.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import yfinance as yf


def load_nvda_atm_term_structure(max_months: int = 9) -> tuple[pd.DataFrame, dict]:
    ticker = yf.Ticker("NVDA")
    history = ticker.history(period="5d", auto_adjust=False)
    if history.empty or "Close" not in history:
        raise RuntimeError("Spot NVDA indisponible chez Yahoo Finance.")
    spot = float(history["Close"].dropna().iloc[-1])
    if not (spot > 0):
        raise RuntimeError("Spot NVDA invalide.")
    observed = pd.Timestamp(history.index[-1]).strftime("%Y-%m-%d")
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    limit = now + pd.DateOffset(months=max_months)
    expiries = sorted(
        pd.Timestamp(e) for e in ticker.options
        if now.normalize() < pd.Timestamp(e) <= limit
    )
    rows = []
    for expiry in expiries:
        try:
            chain = ticker.option_chain(expiry.strftime("%Y-%m-%d"))
        except Exception:
            continue
        candidates = []
        for side, frame in (("call", chain.calls), ("put", chain.puts)):
            if frame is None or frame.empty:
                continue
            valid = frame.copy()
            for col in ("strike", "impliedVolatility", "bid", "ask"):
                valid[col] = pd.to_numeric(valid[col], errors="coerce")
            valid = valid[
                (valid["impliedVolatility"] > 0.01)
                & (valid["impliedVolatility"] < 5)
                & (valid["bid"] > 0)
                & (valid["ask"] >= valid["bid"])
            ].copy()
            if valid.empty:
                continue
            valid["distance"] = (valid["strike"] - spot).abs()
            nearest = valid.sort_values(["distance", "strike"]).iloc[0]
            candidates.append({
                "side": side,
                "strike": float(nearest["strike"]),
                "iv": float(nearest["impliedVolatility"]) * 100,
                "spread": float(nearest["ask"] - nearest["bid"]),
            })
        if not candidates:
            continue
        # Favor tighter bid-ask quotes; a missing side does not fabricate a second leg.
        weights = [1 / max(c["spread"], 0.01) for c in candidates]
        iv = sum(c["iv"] * w for c, w in zip(candidates, weights)) / sum(weights)
        rows.append({
            "Echeance": expiry.strftime("%Y-%m-%d"),
            "Jours": int((expiry - now.normalize()).days),
            "IV_ATM_pct": round(iv, 2),
            "Strike_call": next((c["strike"] for c in candidates if c["side"] == "call"), float("nan")),
            "Strike_put": next((c["strike"] for c in candidates if c["side"] == "put"), float("nan")),
            "Cotes": "+".join(c["side"] for c in candidates),
        })
    if not rows:
        raise RuntimeError("Aucune cotation optionnelle ATM exploitable (bid/ask + IV) chez Yahoo.")
    return pd.DataFrame(rows).sort_values("Jours"), {
        "spot": spot,
        "spot_date": observed,
        "fetched_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "source": "Yahoo Finance / yfinance (cotations potentiellement differees)",
    }
