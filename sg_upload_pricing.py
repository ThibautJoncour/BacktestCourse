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


from xml.etree import ElementTree as ET
from zipfile import ZipFile

_XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _read_xlsx_without_openpyxl(path: Path) -> pd.DataFrame:
    """Read the first XLSX sheet using Python's standard library only."""
    with ZipFile(path) as archive:
        members = set(archive.namelist())
        if "xl/workbook.xml" not in members:
            raise ValueError("Classeur Excel invalide : workbook.xml absent.")
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheets = workbook.find(f"{_XLSX_NS}sheets")
        if sheets is None or len(sheets) == 0:
            raise ValueError("Le fichier Excel ne contient aucune feuille.")
        first = sheets[0]
        rel_id = first.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        relationship = next(
            (node for node in rels
             if node.get("Id") == rel_id),
            None,
        )
        if relationship is None:
            raise ValueError("Première feuille Excel introuvable.")
        target = relationship.get("Target", "")
        if target.startswith("/"):
            sheet_path = target.lstrip("/")
        elif target.startswith("xl/"):
            sheet_path = target
        else:
            from posixpath import normpath
            sheet_path = normpath("xl/" + target)
        if sheet_path not in members:
            raise ValueError("Feuille Excel absente de l'archive.")

        strings = []
        if "xl/sharedStrings.xml" in members:
            shared = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for si in shared.findall(f"{_XLSX_NS}si"):
                strings.append("".join(
                    element.text or "" for element in si.iter(f"{_XLSX_NS}t")
                ))

        sheet = ET.fromstring(archive.read(sheet_path))
        sheet_data = sheet.find(f"{_XLSX_NS}sheetData")
        if sheet_data is None:
            raise ValueError("Feuille Excel vide.")

        cells_by_row = []
        for row in sheet_data.findall(f"{_XLSX_NS}row"):
            columns = {}
            for cell in row.findall(f"{_XLSX_NS}c"):
                reference = cell.get("r", "")
                letters = re.match(r"[A-Z]+", reference)
                if not letters:
                    continue
                position = 0
                for letter in letters.group(0):
                    position = position * 26 + ord(letter) - ord("A") + 1
                position -= 1
                kind = cell.get("t", "")
                raw = cell.find(f"{_XLSX_NS}v")
                if kind == "inlineStr":
                    inline = cell.find(f"{_XLSX_NS}is")
                    value = (
                        "".join(t.text or "" for t in inline.iter(f"{_XLSX_NS}t"))
                        if inline is not None else None
                    )
                elif raw is None:
                    value = None
                elif kind == "s":
                    index = int(raw.text)
                    value = strings[index]
                elif kind == "b":
                    value = raw.text == "1"
                elif kind in ("str", "e"):
                    value = raw.text
                else:
                    try:
                        value = float(raw.text)
                    except (ValueError, TypeError):
                        value = raw.text
                columns[position] = value
            cells_by_row.append(columns)

        if not cells_by_row:
            raise ValueError("Aucune cellule Excel lisible.")
        width = max((max(row, default=-1) for row in cells_by_row), default=-1) + 1
        matrix = [[row.get(index) for index in range(width)] for row in cells_by_row]
        headers = [str(value).strip() if value is not None else f"Unnamed: {i}"
                   for i, value in enumerate(matrix[0])]
        return pd.DataFrame(matrix[1:], columns=headers)


def _read_sg_xlsx(path: Path) -> pd.DataFrame:
    # Always parse .xlsx as ZIP/XML, independent of installed optional packages.
    return _read_xlsx_without_openpyxl(path)


def parse_sg_export(path: str | Path) -> tuple[pd.DataFrame, float]:
    path = Path(path)
    # Shiny stores uploads in temporary files whose paths may have no .xlsx suffix.
    # The UI checks the original filename; validate the actual ZIP content here.
    if not path.is_file():
        raise ValueError("Fichier d'import temporaire introuvable.")
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("Fichier trop volumineux (limite 8 Mo).")

    source = _read_sg_xlsx(path)
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
