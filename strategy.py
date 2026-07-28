from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CODE_VERSION = "VIX3M_FIX_V3_2026-07-28"

START_DATE = "2007-01-01"
HORIZON = 21
ENTRY_QUANTILE = 0.70

PRODUCT_NAME = "SG Stability SPX 6500 / 9500"
PRODUCT_ISIN = "DE000FG1E4N0"
LOWER_BARRIER = 6500.0
UPPER_BARRIER = 9500.0
MATURITY = pd.Timestamp("2027-01-15")

FEATURES = [
    "return_1d",
    "return_5d",
    "return_21d",
    "rv_5",
    "rv_10",
    "rv_21",
    "rv_63",
    "vix",
    "vix3m",
    "vvix",
    "skew",
    "vix_term_structure",
    "vix_ratio",
    "iv_rv_spread",
    "drawdown_63",
    "downside_rv_21",
]


class StrategyError(RuntimeError):
    pass


def _download_close(ticker: str, name: str) -> pd.Series:
    data = yf.download(
        ticker,
        start=START_DATE,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if data.empty:
        raise StrategyError(f"Aucune donnée reçue pour {ticker}.")

    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = pd.to_numeric(close, errors="coerce").rename(name)
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close[~close.index.duplicated(keep="last")].sort_index()

    return close


def download_market_data() -> pd.DataFrame:
    """
    Le calendrier SPX est imposé à toutes les autres séries.
    VIX3M est propagé sans limite sur les dates SPX suivantes afin
    qu'un retard Yahoo de quelques jours ne bloque plus le signal.
    """

    spx = _download_close("^GSPC", "spx")

    df = pd.DataFrame(index=spx.index.copy())
    df["spx"] = spx.reindex(df.index)

    tickers = {
        "vix": "^VIX",
        "vix3m": "^VIX3M",
        "vvix": "^VVIX",
        "skew": "^SKEW",
    }

    original_last_dates = {}

    for name, ticker in tickers.items():
        series = _download_close(ticker, name)
        original_last_dates[name] = series.last_valid_index()

        # Alignement strict sur le calendrier du SPX.
        aligned = series.reindex(df.index)

        # Propagation de la dernière valeur connue.
        # Pas de limit ici : on affiche séparément l'âge réel de la donnée.
        df[name] = aligned.ffill()

    if df[FEATURES[:0]].shape[0] == 0:
        raise StrategyError("Le calendrier SPX est vide.")

    print(f"\n=== CODE VERSION: {CODE_VERSION} ===")
    print("Dernières dates réellement publiées par Yahoo :")
    for name, date in original_last_dates.items():
        print(f"{name}: {date}")

    print("\nDernières valeurs après alignement/ffill :")
    print(df[["spx", "vix", "vix3m", "vvix", "skew"]].tail(5))

    print("\nDernières dates valides après ffill :")
    for column in df.columns:
        print(f"{column}: {df[column].last_valid_index()}")

    # Métadonnées utiles pour signaler qu'une donnée est reportée.
    df.attrs["original_last_dates"] = original_last_dates

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["return_1d"] = np.log(out["spx"] / out["spx"].shift(1))
    out["return_5d"] = np.log(out["spx"] / out["spx"].shift(5))
    out["return_21d"] = np.log(out["spx"] / out["spx"].shift(21))

    for window in [5, 10, 21, 63]:
        out[f"rv_{window}"] = (
            out["return_1d"].rolling(window).std()
            * np.sqrt(252)
            * 100
        )

    out["vix_term_structure"] = out["vix3m"] - out["vix"]
    out["vix_ratio"] = out["vix"] / out["vix3m"]
    out["iv_rv_spread"] = out["vix"] - out["rv_21"]

    out["drawdown_63"] = (
        out["spx"] / out["spx"].rolling(63).max() - 1
    )

    downside_returns = out["return_1d"].where(
        out["return_1d"] < 0,
        0.0,
    )
    out["downside_rv_21"] = (
        downside_returns.rolling(21).std()
        * np.sqrt(252)
        * 100
    )

    future_variance = (
        out["return_1d"]
        .pow(2)
        .shift(-1)
        .rolling(HORIZON)
        .sum()
        .shift(-(HORIZON - 1))
    )

    out["future_rv_21"] = (
        np.sqrt(future_variance * 252 / HORIZON) * 100
    )

    out["vrp_points"] = (
        out["vix"].pow(2) - out["future_rv_21"].pow(2)
    )

    return out.replace([np.inf, -np.inf], np.nan)


def make_ridge_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]
    )


def make_boosting_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_depth=3,
                    learning_rate=0.05,
                    max_iter=250,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            ),
        ]
    )


def compute_daily_signal() -> dict:
    raw = download_market_data()
    original_last_dates = raw.attrs.get("original_last_dates", {})

    df = create_features(raw)

    train = df[FEATURES + ["vrp_points"]].dropna().copy()
    prediction_rows = df[FEATURES].dropna()

    if len(train) < 750:
        raise StrategyError(
            f"Historique insuffisant : {len(train)} observations."
        )

    if prediction_rows.empty:
        raise StrategyError("Aucune feature récente exploitable.")

    latest_features_frame = prediction_rows.iloc[[-1]]

    X_train = train[FEATURES]
    y_train = train["vrp_points"]

    ridge = make_ridge_model()
    boosting = make_boosting_model()

    ridge.fit(X_train, y_train)
    boosting.fit(X_train, y_train)

    ridge_train = ridge.predict(X_train)
    boosting_train = boosting.predict(X_train)
    ensemble_train = 0.5 * ridge_train + 0.5 * boosting_train

    threshold = float(np.quantile(ensemble_train, ENTRY_QUANTILE))

    ridge_score = float(ridge.predict(latest_features_frame)[0])
    boosting_score = float(boosting.predict(latest_features_frame)[0])
    ensemble_score = 0.5 * ridge_score + 0.5 * boosting_score

    latest_date = latest_features_frame.index[-1]
    current_spx = float(df.loc[latest_date, "spx"])
    percentile = float(np.mean(ensemble_train <= ensemble_score))

    vix3m_source_date = original_last_dates.get("vix3m")
    if vix3m_source_date is not None:
        vix3m_age_calendar_days = int(
            (latest_date.normalize() - vix3m_source_date.normalize()).days
        )
    else:
        vix3m_age_calendar_days = None

    inside_barriers = LOWER_BARRIER < current_spx < UPPER_BARRIER
    product_alive = pd.Timestamp.today().normalize() < MATURITY

    take_position = bool(
        ensemble_score >= threshold
        and inside_barriers
        and product_alive
    )

    print("\n=== SIGNAL DU JOUR ===")
    print(f"Code: {CODE_VERSION}")
    print(f"Date du signal: {latest_date.date()}")
    print(f"VIX3M réellement publié le: {vix3m_source_date}")
    print(f"Âge VIX3M en jours calendaires: {vix3m_age_calendar_days}")
    print(f"SPX: {current_spx:.2f}")
    print(f"Score Ensemble: {ensemble_score:.3f}")
    print(f"Seuil: {threshold:.3f}")
    print(f"Prendre position: {take_position}")

    return {
        "ok": True,
        "code_version": CODE_VERSION,
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
        "vix3m_source_date": vix3m_source_date,
        "vix3m_age_calendar_days": vix3m_age_calendar_days,
    }
