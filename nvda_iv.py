"""Observed NVIDIA ATM implied volatility by monthly option expiration.

Yahoo options may have zero bid/ask and model-derived IV. These points are
explicitly tagged as indicative; no synthetic IV is inserted.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf


def _third_friday(year: int, month: int) -> pd.Timestamp:
    first = pd.Timestamp(year=year, month=month, day=1)
    return first + pd.Timedelta(days=(4 - first.weekday()) % 7 + 14)


def _monthly_expirations(available: list[pd.Timestamp], today: pd.Timestamp, months: int):
    """One listed expiration per calendar month, favor standard third Fridays."""
    end = today + pd.DateOffset(months=months)
    candidates = [e for e in available if today < e <= end]
    per_month = {}
    for e in candidates:
        key = (e.year, e.month)
        target = _third_friday(*key)
        if key not in per_month or abs((e - target).days) < abs((per_month[key] - target).days):
            per_month[key] = e
    return sorted(per_month.values())


def _near_atm(frame: pd.DataFrame, spot: float, side: str):
    if frame is None or frame.empty:
        return None
    f = frame.copy()
    for field in ("strike", "impliedVolatility", "bid", "ask"):
        if field not in f:
            f[field] = np.nan
        f[field] = pd.to_numeric(f[field], errors="coerce")
    # Avoid picking a highly liquid but far-out-of-the-money contract.
    f = f[
        f["strike"].between(spot * 0.97, spot * 1.03)
        & f["impliedVolatility"].between(0.05, 3.0)
    ].copy()
    if f.empty:
        return None
    f["liquid"] = (f["bid"] > 0) & (f["ask"] >= f["bid"])
    f["distance"] = abs(f["strike"] - spot)
    f = f.sort_values(["distance", "liquid"], ascending=[True, False])
    nearest = f.iloc[0]
    return {
        "side": side,
        "strike": float(nearest["strike"]),
        "iv": float(nearest["impliedVolatility"]) * 100,
        "liquid": bool(nearest["liquid"]),
    }


def load_nvda_atm_term_structure(max_months: int = 9) -> tuple[pd.DataFrame, dict]:
    ticker = yf.Ticker("NVDA")
    history = ticker.history(period="5d", auto_adjust=False)
    if history.empty or "Close" not in history or history["Close"].dropna().empty:
        raise RuntimeError("Spot NVDA indisponible chez Yahoo Finance.")
    spot = float(history["Close"].dropna().iloc[-1])
    if not np.isfinite(spot) or spot <= 0:
        raise RuntimeError("Spot NVDA invalide.")
    observed = pd.Timestamp(history.index[-1]).strftime("%Y-%m-%d")
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    expirations = _monthly_expirations(
        [pd.Timestamp(e) for e in ticker.options], today, max_months
    )
    rows = []
    for expiry in expirations:
        try:
            chain = ticker.option_chain(expiry.strftime("%Y-%m-%d"))
        except Exception:
            continue
        legs = [
            leg for leg in (
                _near_atm(chain.calls, spot, "call"),
                _near_atm(chain.puts, spot, "put"),
            ) if leg is not None
        ]
        if not legs:
            continue
        # Both sides must be ATM; average the two IVs when available.
        # A zero bid is common on Yahoo: retain it only as "indicative".
        iv = float(np.mean([leg["iv"] for leg in legs]))
        quality = "Bid/ask valides" if all(leg["liquid"] for leg in legs) else "Indicative (bid/ask incomplets)"
        rows.append({
            "Echeance": expiry.strftime("%Y-%m-%d"),
            "Jours": int((expiry - today).days),
            "IV_ATM_pct": round(iv, 2),
            "Strike_call": next((leg["strike"] for leg in legs if leg["side"] == "call"), float("nan")),
            "Strike_put": next((leg["strike"] for leg in legs if leg["side"] == "put"), float("nan")),
            "Cotes": "+".join(leg["side"] for leg in legs),
            "Qualite": quality,
        })
    if not rows:
        raise RuntimeError("Aucune IV ATM exploitable dans les chaînes d'options Yahoo.")
    return pd.DataFrame(rows).sort_values("Jours"), {
        "spot": spot,
        "spot_date": observed,
        "fetched_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "requested_months": max_months,
        "available_months": len(expirations),
        "source": "Yahoo Finance / yfinance (potentiellement différé)",
    }
