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


def _load_yahoo_atm_term_structure(max_months: int = 9) -> tuple[pd.DataFrame, dict]:
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


def _load_optionsband_atm_term_structure(max_months: int = 9) -> tuple[pd.DataFrame, dict]:
    """Public delayed ATM IV series; no API key. Fail transparently on unknown schema."""
    import requests

    response = requests.get(
        "https://optionsband.com/api/v1/expirations/NVDA",
        params={"limit": 24},
        headers={"Accept": "application/json", "User-Agent": "NVDA-IV-Research/1.0"},
        timeout=12,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Format de réponse Options Band inattendu")
    data = payload.get("data", payload)
    if isinstance(data, list):
        entries = data
        info = payload
    elif isinstance(data, dict):
        entries = next(
            (data[k] for k in ("expirations", "rows", "items", "results", "term_structure")
             if isinstance(data.get(k), list)), None
        )
        info = data
    else:
        entries = None
        info = payload
    if not entries:
        raise ValueError("Aucune échéance dans la réponse Options Band")
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    horizon = today + pd.DateOffset(months=max_months)
    rows = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        expiration = next(
            (item[k] for k in ("expiration", "expiry", "expiration_date", "expirationDate", "date")
             if item.get(k)), None
        )
        raw_iv = next(
            (item[k] for k in ("atm_iv", "iv_atm", "atmIv", "atm_iv_pct", "ivAtm")
             if item.get(k) is not None), None
        )
        if expiration is None or raw_iv is None:
            continue
        expiry = pd.to_datetime(expiration, errors="coerce", utc=True)
        if pd.isna(expiry):
            continue
        expiry = expiry.tz_localize(None).normalize()
        if not today < expiry <= horizon:
            continue
        try:
            iv = float(raw_iv)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(iv):
            continue
        # Fractional vol from API: 0.6 -> 60%; percent values already >= 1.
        iv_pct = iv * 100 if iv < 1 else iv
        if not 5 <= iv_pct <= 300:
            continue
        rows.append({
            "Echeance": expiry.strftime("%Y-%m-%d"),
            "Jours": int((expiry - today).days),
            "IV_ATM_pct": round(iv_pct, 2),
            "Strike_call": float("nan"),
            "Strike_put": float("nan"),
            "Cotes": "ATM fournisseur",
            "Qualite": "Bid/ask valides" if item.get("quote_quality") == "valid" else "Indicative (source différée)",
        })
    if not rows:
        raise ValueError("Aucune IV ATM valide extraite d'Options Band")
    frame = pd.DataFrame(rows).drop_duplicates("Echeance").sort_values("Jours")
    # Only show listed monthly expirations.
    frame["month"] = pd.to_datetime(frame["Echeance"]).dt.to_period("M")
    frame = frame.drop_duplicates("month").drop(columns="month")
    spot = info.get("underlying_price", info.get("spot", float("nan")))
    try:
        spot = float(spot)
    except (TypeError, ValueError):
        spot = float("nan")
    return frame, {
        "spot": spot,
        "spot_date": str(payload.get("as_of") or info.get("as_of") or "non communiqué")[:10],
        "fetched_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "available_months": len(frame),
        "source": "Options Band (options Cboe différées ; IV fournisseur)",
    }


def load_nvda_atm_term_structure(max_months: int = 9) -> tuple[pd.DataFrame, dict]:
    """Try a public IV-by-expiry API first; fallback to Yahoo only if necessary."""
    errors = []
    for provider in (_load_optionsband_atm_term_structure, _load_yahoo_atm_term_structure):
        try:
            return provider(max_months=max_months)
        except Exception as exc:
            errors.append(f"{provider.__name__}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Sources IV indisponibles. " + " | ".join(errors))
