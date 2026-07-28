from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

START_DATE = "2007-01-01"
HORIZON = 21
ENTRY_QUANTILE = 0.70

PRODUCT_NAME = "SG Stability SPX 6500 / 9500"
PRODUCT_ISIN = "DE000FG1E4N0"
LOWER_BARRIER = 6500.0
UPPER_BARRIER = 9500.0
MATURITY = pd.Timestamp("2027-01-15")

FEATURES = [
    "return_1d", "return_5d", "return_21d", "rv_5", "rv_10", "rv_21", "rv_63",
    "vix", "vix3m", "vvix", "skew", "vix_term_structure", "vix_ratio",
    "iv_rv_spread", "drawdown_63", "downside_rv_21",
]

class StrategyError(RuntimeError):
    pass


def _download_close(ticker: str, name: str) -> pd.Series:
    data = yf.download(ticker, start=START_DATE, auto_adjust=True, progress=False, threads=False)
    if data.empty:
        raise StrategyError(f"Aucune donnée reçue pour {ticker}.")
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.rename(name)
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close


def download_market_data() -> pd.DataFrame:
    series = [
        _download_close("^GSPC", "spx"),
        _download_close("^VIX", "vix"),
        _download_close("^VIX3M", "vix3m"),
        _download_close("^VVIX", "vvix"),
        _download_close("^SKEW", "skew"),
    ]
    df = pd.concat(series, axis=1).sort_index().ffill(limit=3)
    if df["spx"].dropna().empty:
        raise StrategyError("La série SPX est vide.")
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["return_1d"] = np.log(out["spx"] / out["spx"].shift(1))
    out["return_5d"] = np.log(out["spx"] / out["spx"].shift(5))
    out["return_21d"] = np.log(out["spx"] / out["spx"].shift(21))
    for window in [5, 10, 21, 63]:
        out[f"rv_{window}"] = out["return_1d"].rolling(window).std() * np.sqrt(252) * 100
    out["vix_term_structure"] = out["vix3m"] - out["vix"]
    out["vix_ratio"] = out["vix"] / out["vix3m"]
    out["iv_rv_spread"] = out["vix"] - out["rv_21"]
    out["drawdown_63"] = out["spx"] / out["spx"].rolling(63).max() - 1
    downside = out["return_1d"].where(out["return_1d"] < 0, 0.0)
    out["downside_rv_21"] = downside.rolling(21).std() * np.sqrt(252) * 100

    future_variance = out["return_1d"].pow(2).shift(-1).rolling(HORIZON).sum().shift(-(HORIZON - 1))
    out["future_rv_21"] = np.sqrt(future_variance * 252 / HORIZON) * 100
    out["vrp_points"] = out["vix"].pow(2) - out["future_rv_21"].pow(2)
    return out.replace([np.inf, -np.inf], np.nan)


def make_ridge_model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=10.0)),
    ])


def make_boosting_model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=250, l2_regularization=1.0, random_state=42)),
    ])


def compute_daily_signal() -> dict:
    raw = download_market_data()
    df = create_features(raw)
    train = df[FEATURES + ["vrp_points"]].dropna().copy()
    latest_features = df[FEATURES].dropna().iloc[[-1]]

    if len(train) < 750:
        raise StrategyError(f"Historique insuffisant : {len(train)} observations.")
    if latest_features.empty:
        raise StrategyError("Aucune feature récente exploitable.")

    X_train, y_train = train[FEATURES], train["vrp_points"]
    ridge, boosting = make_ridge_model(), make_boosting_model()
    ridge.fit(X_train, y_train)
    boosting.fit(X_train, y_train)

    ridge_train = ridge.predict(X_train)
    boosting_train = boosting.predict(X_train)
    ensemble_train = 0.5 * ridge_train + 0.5 * boosting_train
    threshold = float(np.quantile(ensemble_train, ENTRY_QUANTILE))

    ridge_score = float(ridge.predict(latest_features)[0])
    boosting_score = float(boosting.predict(latest_features)[0])
    ensemble_score = 0.5 * ridge_score + 0.5 * boosting_score

    latest_date = latest_features.index[-1]
    current_spx = float(df.loc[latest_date, "spx"])
    percentile = float(np.mean(ensemble_train <= ensemble_score))
    inside_barriers = LOWER_BARRIER < current_spx < UPPER_BARRIER
    product_alive = pd.Timestamp.today().normalize() < MATURITY
    take_position = bool(ensemble_score >= threshold and inside_barriers and product_alive)

    return {
        "ok": True,
        "date": latest_date,
        "last_target_date": train.index[-1],
        "spx": current_spx,
        "ridge_score": ridge_score,
        "boosting_score": boosting_score,
        "ensemble_score": ensemble_score,
        "threshold": threshold,
        "percentile": percentile,
        "take_position": take_position,
        "product_name": PRODUCT_NAME,
        "product_isin": PRODUCT_ISIN,
        "lower_barrier": LOWER_BARRIER,
        "upper_barrier": UPPER_BARRIER,
        "lower_distance": LOWER_BARRIER / current_spx - 1,
        "upper_distance": UPPER_BARRIER / current_spx - 1,
        "maturity": MATURITY,
    }
