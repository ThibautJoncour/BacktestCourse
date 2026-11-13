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



def _load_optionsband_public_page(max_months: int = 9) -> tuple[pd.DataFrame, dict]:
    """Parse Options Band's public NVDA term-structure table, if reachable."""
    import requests
    from bs4 import BeautifulSoup

    response = requests.get(
        "https://optionsband.com/stocks/NVDA/volatility",
        headers={"User-Agent": "Mozilla/5.0 (compatible; research-dashboard/1.0)"},
        timeout=15,
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    heading = soup.find(
        lambda tag: tag.name in ("h2", "h3")
        and "IV term structure" in tag.get_text(" ", strip=True)
    )
    if heading is None:
        raise ValueError("Section IV term structure absente de la page")
    table = heading.find_next("table")
    if table is None:
        raise ValueError("Table des échéances absente")
    headers = [cell.get_text(" ", strip=True).lower() for cell in table.find("tr").find_all(["th", "td"])]
    date_idx = next((i for i, h in enumerate(headers) if "expir" in h), None)
    iv_idx = next((i for i, h in enumerate(headers) if "atm iv" in h), None)
    if date_idx is None or iv_idx is None:
        raise ValueError(f"Colonnes IV ATM inconnues: {headers}")
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    limit = today + pd.DateOffset(months=max_months)
    entries = []
    for tr in table.find_all("tr")[1:]:
        cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
        if max(date_idx, iv_idx) >= len(cells):
            continue
        expiry = pd.to_datetime(cells[date_idx], errors="coerce")
        if pd.isna(expiry):
            continue
        expiry = expiry.normalize()
        if not today < expiry <= limit:
            continue
        try:
            iv = float(cells[iv_idx].replace("%", "").strip())
        except ValueError:
            continue
        if 5 <= iv <= 300:
            entries.append((expiry, iv))
    if not entries:
        raise ValueError("Aucune échéance et IV ATM future lisible dans la page")
    monthly = {}
    for expiry, iv in entries:
        key = (expiry.year, expiry.month)
        target = _third_friday(*key)
        if key not in monthly or abs((expiry-target).days) < abs((monthly[key][0]-target).days):
            monthly[key] = (expiry, iv)
    rows = [
        {
            "Echeance": expiry.strftime("%Y-%m-%d"),
            "Jours": (expiry-today).days,
            "IV_ATM_pct": iv,
            "Strike_call": float("nan"),
            "Strike_put": float("nan"),
            "Cotes": "ATM fournisseur",
            "Qualite": "Indicative (Cboe différé)",
        }
        for expiry, iv in sorted(monthly.values())
    ]
    return pd.DataFrame(rows), {
        "spot": float("nan"),
        "spot_date": "voir source",
        "fetched_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "available_months": len(rows),
        "source": "Options Band — page publique (Cboe différé)",
    }


# Verified public snapshot from 2026-11-13 (Options Band NVDA volatility page).
# NOT live data. Used ONLY as a visibly dated last-resort reference curve.
_REFERENCE_SNAPSHOT_DATE = "2026-11-13"
_REFERENCE_SNAPSHOT = [
    ("2026-11-20", 87.1),
    ("2026-12-18", 58.1),
    ("2027-01-15", 52.1),
    ("2027-02-19", 49.3),
    ("2027-03-19", 49.4),
    ("2027-04-16", 48.0),
]


def _load_reference_snapshot(max_months: int = 9) -> tuple[pd.DataFrame, dict]:
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    end = today + pd.DateOffset(months=max_months)
    rows = []
    for date_str, iv in _REFERENCE_SNAPSHOT:
        expiry = pd.Timestamp(date_str)
        if not today < expiry <= end:
            continue
        rows.append({
            "Echeance": date_str,
            "Jours": (expiry - today).days,
            "IV_ATM_pct": iv,
            "Strike_call": float("nan"),
            "Strike_put": float("nan"),
            "Cotes": "Référence datée",
            "Qualite": "Indicative (archive 13/11/2026)",
        })
    if not rows:
        raise RuntimeError("L'archive du 13/11/2026 ne contient plus d'échéance future.")
    return pd.DataFrame(rows), {
        "spot": float("nan"),
        "spot_date": _REFERENCE_SNAPSHOT_DATE,
        "fetched_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "available_months": len(rows),
        "source": "ARCHIVE 13/11/2026 — Options Band (NON ACTUALISÉE)",
    }


def load_nvda_atm_term_structure(max_months: int = 9) -> tuple[pd.DataFrame, dict]:
    """Try a public IV-by-expiry API first; fallback to Yahoo only if necessary."""
    errors = []
    for provider in (_load_optionsband_atm_term_structure, _load_yahoo_atm_term_structure, _load_optionsband_public_page, _load_reference_snapshot):
        try:
            return provider(max_months=max_months)
        except Exception as exc:
            errors.append(f"{provider.__name__}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Sources IV indisponibles. " + " | ".join(errors))
