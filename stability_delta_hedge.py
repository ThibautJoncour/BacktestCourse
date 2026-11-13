"""Diagnostic delta-neutral SG Stability NVDA vs SG NVIDIA call/put warrants.

Greeks are MODEL estimates (HAR-RV bump/reprice) and not issuer Greeks.
Quotes or warrant availability are never implied from a failed SG request.
"""
from __future__ import annotations

from datetime import date
from io import BytesIO, StringIO
from math import erf, exp, log, sqrt
import re

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from strategy import _price_nvda_stabilities_mc

SG_WARRANTS_URL = (
    "https://sgbourse.fr/product-search/warrants/"
    "actions%2520am%C3%A9ricaines/nvidia/all-assets?ShowFilter=false"
)

def _num(v):
    if v is None or pd.isna(v):
        return np.nan
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace("\u00a0", "").replace(" ", "").replace("€", "")
    m = re.search(r"-?\d[\d.,]*", s)
    if not m:
        return np.nan
    s = m.group()
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan

def _parity(v):
    if isinstance(v, str) and ":" in v:
        return _num(v.split(":")[0])
    return _num(v)

def _normal_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def _normal_pdf(x):
    return exp(-0.5*x*x) / sqrt(2.0*np.pi)

def _bs_greeks(spot, strike, T, sigma, kind, parity, fx):
    """European proxy in EUR for one USD-underlying warrant; FX=USD per EUR."""
    if min(spot, strike, T, sigma, parity, fx) <= 0:
        return np.nan, np.nan, np.nan
    d1 = (log(spot/strike) + (0.02 + 0.5*sigma*sigma)*T)/(sigma*sqrt(T))
    delta_usd = _normal_cdf(d1) if kind == "CALL" else _normal_cdf(d1)-1.0
    gamma_usd = _normal_pdf(d1)/(spot*sigma*sqrt(T))
    vega_usd_per_pp = spot*_normal_pdf(d1)*sqrt(T)/100.0
    factor = parity*fx
    return delta_usd/factor, gamma_usd/factor, vega_usd_per_pp/factor

def import_warrants(path):
    """Accept SG-style Excel/CSV with code, type, strike, maturity and parity."""
    path = str(path)
    frame = pd.read_csv(path, sep=None, engine="python") if path.lower().endswith(".csv") else pd.read_excel(path)
    names = {
        "mnémo.": "Code", "mnémo": "Code", "code": "Code",
        "code isin": "ISIN", "isin": "ISIN",
        "type": "Type", "type de produit": "Type",
        "prix d'exercice": "Strike", "strike": "Strike",
        "maturité": "Maturite", "maturite": "Maturite",
        "parité": "Parite", "parite": "Parite",
        "achat": "Ask", "vente": "Bid",
    }
    frame.columns = [names.get(str(c).strip().lower(), str(c).strip()) for c in frame.columns]
    if not {"Code","Type","Strike","Maturite","Parite"}.issubset(frame.columns):
        raise ValueError("Colonnes nécessaires : Mnémo., Type, Prix d'exercice, Maturité, Parité.")
    out = frame.copy()
    out["Code"] = out["Code"].astype(str).str.strip().str.upper()
    out["Type"] = out["Type"].astype(str).str.upper().map(
        lambda s: "PUT" if "PUT" in s else ("CALL" if "CALL" in s else ""))
    for k, fn in [("Strike",_num),("Parite",_parity)]:
        out[k] = out[k].map(fn)
    out["Maturite"] = pd.to_datetime(out["Maturite"],dayfirst=True,errors="coerce").dt.normalize()
    out = out[(out["Type"]!="")&(out["Strike"]>0)&(out["Parite"]>0)&out["Maturite"].notna()]
    for k in ("Ask","Bid"):
        out[k] = out[k].map(_num) if k in out else np.nan
    out = out.drop_duplicates("Code").reset_index(drop=True)
    if out.empty:
        raise ValueError("Aucun call/put NVIDIA exploitable dans le fichier.")
    return out

def fetch_sg_warrants():
    """Best-effort public list; return an empty table on JS shell/HTTP failure."""
    cols = ["Code","Type","Strike","Maturite","Parite","Ask","Bid"]
    try:
        response = requests.get(SG_WARRANTS_URL, timeout=12,
            headers={"User-Agent":"Mozilla/5.0","Accept-Language":"fr-FR"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text,"html.parser")
        found = []
        # Only read complete table rows: never infer strikes from arbitrary page text.
        for tr in soup.select("tr"):
            text = tr.get_text(" ",strip=True)
            anchors = tr.select('a[href*="/product-details/"]')
            if not anchors:
                continue
            cells=[t.get_text(" ",strip=True) for t in tr.select("td")]
            kind = "PUT" if re.search(r"\bPut\b",text,re.I) else "CALL" if re.search(r"\bCall\b",text,re.I) else ""
            strike = re.search(r"(\d+[,.]\d+|\d+)\s*USD",text,re.I)
            maturity = re.search(r"\b\d{2}/\d{2}/\d{4}\b",text)
            parity = re.search(r"\b(\d+)\s*:\s*1\b",text)
            if not (kind and strike and maturity and parity and cells):
                continue
            code=anchors[0].get_text(" ",strip=True).upper()
            if not re.fullmatch(r"[A-Z0-9]{4,8}",code):
                continue
            found.append({"Code":code,"Type":kind,"Strike":_num(strike.group(1)),
               "Maturite":pd.to_datetime(maturity.group(),dayfirst=True),
               "Parite":float(parity.group(1)),"Ask":np.nan,"Bid":np.nan})
        return pd.DataFrame(found,columns=cols).drop_duplicates("Code")
    except (requests.RequestException,ValueError) :
        return pd.DataFrame(columns=cols)

def stability_greeks(priced, signal, bump_usd=1.0, vol_bump_pp=1.0, paths=50000):
    """Common-random-number finite differences, per Stability in EUR/USD and EUR/USD².

    Vega reported in EUR per +1 vol percentage point. Uses HAR *realized* sigma
    as scenario input; this is NOT market implied-vol vega.
    """
    if priced.empty:
        return priced.copy()
    spot=float(priced["Spot"].iloc[0])
    base_vol=signal["nvda_har_rv_forecasts"]
    long_vol=float(signal["nvda_har_long_run_vol_pct"])
    today=pd.Timestamp.today().normalize()
    products=priced[["Code","ISIN","Barriere_basse","Barriere_haute","Maturite"]].copy()
    products["Prix_marche"]=np.nan
    def price(s, forecasts=base_vol, long_run=long_vol):
        o=_price_nvda_stabilities_mc(s,today,forecasts,products,long_run,
                                    nb_paths=paths)
        return o.set_index(["Code","ISIN"])["Prix_theorique_MC"]
    base=priced.set_index(["Code","ISIN"]).copy()
    up=price(spot+bump_usd)
    down=price(spot-bump_usd)
    bumped={k:float(v)+vol_bump_pp for k,v in base_vol.items()}
    volup=price(spot,bumped,long_vol+vol_bump_pp)
    idx=base.index
    base["Delta_MC_EUR_par_USD"]=(up.reindex(idx)-down.reindex(idx))/(2*bump_usd)
    base["Gamma_MC_EUR_par_USD2"]=(up.reindex(idx)-2*base["Prix_theorique_MC"]+down.reindex(idx))/(bump_usd**2)
    base["Vega_MC_EUR_par_1pp"]=(volup.reindex(idx)-base["Prix_theorique_MC"])/vol_bump_pp
    return base.reset_index()

def propose_warrant_hedges(priced, warrants, quantity=100, eurusd=1.0):
    """Match nearest barrier's put/call; neutralize DELTA, not KO/gamma/vega.

    FX=USD per EUR; required for accurate cross-currency delta and notionals.
    Never suggest an executable hedge without valid quotes and true delta inputs.
    """
    rows=[]
    now=pd.Timestamp.today().normalize()
    for p in priced.to_dict("records"):
        spot=float(p["Spot"])
        lower=float(p["Barriere_basse"])
        upper=float(p["Barriere_haute"])
        nearest="BASSE" if spot-lower <= upper-spot else "HAUTE"
        target=lower if nearest=="BASSE" else upper
        side="PUT" if nearest=="BASSE" else "CALL"
        d=float(p.get("Delta_MC_EUR_par_USD",np.nan))
        target_delta=-quantity*d
        expiry=pd.Timestamp(p["Maturite"]).normalize()
        eligible=warrants[(warrants["Type"]==side)&(warrants["Maturite"]>=expiry)].copy() if not warrants.empty else pd.DataFrame()
        if not eligible.empty:
            eligible["strike_error"]=(eligible["Strike"]-target).abs()
            eligible["expiry_error"]=(eligible["Maturite"]-expiry).dt.days
            eligible=eligible.sort_values(["strike_error","expiry_error"])
            w=eligible.iloc[0]
            T=max((w["Maturite"]-now).days/365.0,1/365)
            sigma=max(float(p["Sigma_RMS_pct_ann"])/100,0.01)
            dw,gw,vw=_bs_greeks(spot,float(w["Strike"]),T,sigma,side,float(w["Parite"]),eurusd)
            nw=target_delta/dw if np.isfinite(dw) and abs(dw)>1e-8 else np.nan
            action="ACHETER" if nw>0 else "VENDRE" if nw<0 else "NEUTRE"
            code=w["Code"]
            strike=w["Strike"]
            option_expiry=w["Maturite"].date().isoformat()
            residual=quantity*d+nw*dw if np.isfinite(nw) else np.nan
            net_gamma=quantity*float(p["Gamma_MC_EUR_par_USD2"])+nw*gw if np.isfinite(nw) else np.nan
            net_vega=quantity*float(p["Vega_MC_EUR_par_1pp"])+nw*vw if np.isfinite(nw) else np.nan
            status="THEORIQUE — vérifier prix, liquidité, convention et FX"
        else:
            code="—"; strike=np.nan; option_expiry="—"; nw=np.nan
            action="INDISPONIBLE"; residual=np.nan; net_gamma=np.nan;net_vega=np.nan
            status="Aucun warrant importé avec type et échéance compatibles"
        rows.append({"Stability":p["Code"],"Barriere_proche":nearest,
            "Niveau_barriere":target,"Delta_stab_EUR_USD":d,"Quantite_stab":quantity,
            "Warrant":code,"Type_couverture":side,"Strike_warrant":strike,
            "Echeance_warrant":option_expiry,"Action":action,"Quantite_warrants":abs(nw) if np.isfinite(nw) else np.nan,
            "Delta_net_EUR_USD":residual,"Gamma_net_EUR_USD2":net_gamma,
            "Vega_net_EUR_1pp":net_vega,"Statut":status})
    return pd.DataFrame(rows)
