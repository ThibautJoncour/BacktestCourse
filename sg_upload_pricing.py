"""Read uploaded SG Bourse NVIDIA Stability exports for scenario Monte Carlo.

Uploaded quote prices are snapshots without guaranteed quote timestamps.
They must not be represented as live/executable prices.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from strategy import _price_nvda_stabilities_mc


_REQUIRED = {
    "Mnémo.": "Code",
    "Code ISIN": "ISIN",
    "Sous-jacent": "Sous_jacent",
    "Barrière": "Barriere_basse",
    "Borne haute": "Barriere_haute",
    "Maturité": "Maturite",
    "Achat": "Achat_export",
    "Vente": "Vente_export",
    "Prix sous-jacent": "Spot_export",
}


def _fr_float(value) -> float:
    if pd.isna(value):
        return float("nan")
    if isinstance(value, (float, int)):
        return float(value)
    s = str(value).replace("\u00a0", " ").replace(" ", "")
    match = re.search(r"[-+]?(?:\d+(?:[.,]\d+)?)", s)
    if not match:
        return float("nan")
    return float(match.group(0).replace(",", "."))


def parse_sg_export(path: str | Path) -> tuple[pd.DataFrame, float]:
    path = Path(path)
    if path.suffix.lower() != ".xlsx":
        raise ValueError("Importer un fichier SG Excel au format .xlsx.")
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("Fichier trop volumineux (limite 8 Mo).")

    source = pd.read_excel(path, engine="openpyxl", sheet_name=0)
    source.columns = [str(column).strip() for column in source.columns]
    missing = sorted(set(_REQUIRED) - set(source.columns))
    if missing:
        raise ValueError("Colonnes SG manquantes : " + ", ".join(missing))

    data = source[list(_REQUIRED)].rename(columns=_REQUIRED).copy()
    data["Sous_jacent"] = data["Sous_jacent"].astype(str).str.strip().str.upper()
    data = data[data["Sous_jacent"].str.contains("NVIDIA|NVDA", regex=True, na=False)].copy()
    if data.empty:
        raise ValueError("Aucun produit NVIDIA dans cet export.")

    data["Code"] = data["Code"].astype(str).str.strip().str.upper()
    data["ISIN"] = data["ISIN"].astype(str).str.strip().str.upper()
    for column in ("Barriere_basse", "Barriere_haute", "Achat_export", "Vente_export", "Spot_export"):
        data[column] = data[column].map(_fr_float)
    data["Maturite"] = pd.to_datetime(data["Maturite"], dayfirst=True, errors="coerce").dt.normalize()
    valid = (
        data["Code"].str.fullmatch(r"[A-Z0-9]{4,8}")
        & data["ISIN"].str.fullmatch(r"[A-Z]{2}[A-Z0-9]{10}")
        & data["Maturite"].notna()
        & np.isfinite(data["Barriere_basse"])
        & np.isfinite(data["Barriere_haute"])
        & (data["Barriere_basse"] < data["Barriere_haute"])
    )
    if not valid.all():
        raise ValueError(f"{int((~valid).sum())} ligne(s) SG invalides : ISIN, barrières ou maturité.")

    spots = data["Spot_export"].dropna()
    spots = spots[spots > 0]
    if spots.empty:
        raise ValueError("Spot NVIDIA manquant dans la colonne Prix sous-jacent.")
    spot = float(spots.median())
    if ((spots - spot).abs() > 0.02).any():
        raise ValueError("Plusieurs spots NVIDIA incompatibles dans l'export.")

    data["Achat_export"] = data["Achat_export"].where(data["Achat_export"] > 0, np.nan)
    data["Vente_export"] = data["Vente_export"].where(data["Vente_export"] > 0, np.nan)
    data = data.drop_duplicates(subset=["Code", "ISIN"], keep="last")
    return data.reset_index(drop=True), spot


def price_uploaded_export(path: str | Path, signal: dict) -> tuple[pd.DataFrame, dict]:
    if not signal.get("ok"):
        raise ValueError("HAR-RV indisponible : " + str(signal.get("error", "calcul impossible")))
    products, spot = parse_sg_export(path)
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    future = products[products["Maturite"] > today].copy()
    excluded = len(products) - len(future)
    if future.empty:
        raise ValueError("L'export ne contient aucun produit d'échéance future.")

    mc_products = future[["Code", "ISIN", "Barriere_basse", "Barriere_haute", "Maturite"]].copy()
    mc_products["Maturite"] = mc_products["Maturite"].dt.strftime("%Y-%m-%d")
    # Export SG quotes are NOT passed as live market prices to the pricing engine.
    mc_products["Prix_marche"] = np.nan
    mc_products["Source_prix"] = "Export utilisateur : cotation non-live"

    priced = _price_nvda_stabilities_mc(
        spot=spot,
        today=today,
        har_forecasts=signal["nvda_har_rv_forecasts"],
        long_run_vol_pct=float(signal["nvda_har_long_run_vol_pct"]),
        products=mc_products,
    )
    priced = priced.merge(
        future[["Code", "ISIN", "Achat_export", "Vente_export"]],
        how="left",
        on=["Code", "ISIN"],
        validate="one_to_one",
    )
    # Comparing today's scenario to undated quotes is diagnostic only.
    priced["Ecart_scenario_achat"] = priced["Prix_theorique_MC"] - priced["Achat_export"]
    priced["Rendement_scenario_pct"] = (
        100.0 * (priced["Prix_theorique_MC"] / priced["Achat_export"] - 1)
    ).replace([np.inf, -np.inf], np.nan)
    priced = priced.sort_values("Ecart_scenario_achat", ascending=False, na_position="last")
    return priced.reset_index(drop=True), {
        "spot_export": spot,
        "spot_har_scenario": signal.get("nvda_spot"),
        "valuation_date": today.strftime("%Y-%m-%d"),
        "count": len(future),
        "excluded_expired": excluded,
        "quoted": int(future["Achat_export"].notna().sum()),
        "quote_asof": "non fourni par l'export",
    }
