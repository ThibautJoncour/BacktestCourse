from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CODE_VERSION = "VIX3M_FIX_V4_WITH_JUMP_REGIME_2026-07-29"

START_DATE = "2007-01-01"
HORIZON = 21
ENTRY_QUANTILE = 0.70

# Paramètres du Jump Model Bull/Bear
JUMP_TRAINING_WINDOW = 3000
JUMP_PENALTY = 35.0
JUMP_N_INIT = 10
JUMP_FEATURES = ["dd_10", "sortino_20", "sortino_60"]

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



# ============================================================
# JUMP MODEL BULL / BEAR
# ============================================================

def _ewm_downside_deviation(returns: pd.Series, halflife: int) -> pd.Series:
    downside_squared = returns.clip(upper=0) ** 2
    downside_variance = downside_squared.ewm(
        halflife=halflife,
        adjust=False,
        min_periods=halflife,
    ).mean()
    return np.sqrt(downside_variance)


def _ewm_sortino(returns: pd.Series, halflife: int) -> pd.Series:
    ewm_return = returns.ewm(
        halflife=halflife,
        adjust=False,
        min_periods=halflife,
    ).mean()
    downside = _ewm_downside_deviation(returns, halflife)
    return ewm_return / downside.replace(0, np.nan)


def _infer_jump_states(
    X: np.ndarray,
    centroids: np.ndarray,
    jump_penalty: float,
) -> np.ndarray:
    n_obs = X.shape[0]
    n_states = centroids.shape[0]

    emission = np.zeros((n_obs, n_states))
    for state in range(n_states):
        emission[:, state] = 0.5 * np.sum(
            (X - centroids[state]) ** 2, axis=1
        )

    total = np.full((n_obs, n_states), np.inf)
    previous = np.zeros((n_obs, n_states), dtype=int)
    total[0] = emission[0]

    for t in range(1, n_obs):
        for current_state in range(n_states):
            transition = total[t - 1] + jump_penalty * (
                np.arange(n_states) != current_state
            )
            best_previous = int(np.argmin(transition))
            previous[t, current_state] = best_previous
            total[t, current_state] = (
                transition[best_previous] + emission[t, current_state]
            )

    states = np.zeros(n_obs, dtype=int)
    states[-1] = int(np.argmin(total[-1]))
    for t in range(n_obs - 1, 0, -1):
        states[t - 1] = previous[t, states[t]]

    return states


def _fit_jump_model(
    X: np.ndarray,
    jump_penalty: float,
    n_init: int,
    max_iterations: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    best_objective = np.inf
    best_centroids = None
    best_states = None
    rng = np.random.default_rng(42)

    for _ in range(n_init):
        seed = int(rng.integers(0, 1_000_000))
        kmeans = KMeans(n_clusters=2, n_init=1, random_state=seed)
        kmeans.fit(X)
        centroids = kmeans.cluster_centers_.copy()
        previous_states = None

        for _ in range(max_iterations):
            states = _infer_jump_states(X, centroids, jump_penalty)
            new_centroids = centroids.copy()

            for state in range(2):
                mask = states == state
                if mask.any():
                    new_centroids[state] = X[mask].mean(axis=0)

            converged = (
                previous_states is not None
                and np.array_equal(states, previous_states)
            )
            centroids = new_centroids
            if converged:
                break
            previous_states = states.copy()

        states = _infer_jump_states(X, centroids, jump_penalty)
        distance_cost = sum(
            0.5 * np.sum((X[t] - centroids[states[t]]) ** 2)
            for t in range(len(X))
        )
        jumps = np.sum(states[1:] != states[:-1])
        objective = distance_cost + jump_penalty * jumps

        if objective < best_objective:
            best_objective = objective
            best_centroids = centroids.copy()
            best_states = states.copy()

    if best_centroids is None or best_states is None:
        raise StrategyError("Échec de l'entraînement du Jump Model.")

    return best_centroids, best_states


def compute_jump_regime(df: pd.DataFrame) -> dict:
    """
    Calcule le régime Bull/Bear du dernier jour disponible.

    Le modèle est entraîné sur les JUMP_TRAINING_WINDOW observations
    précédant le jour du signal. Le dernier état est ensuite attribué
    sans utiliser de donnée future.
    """
    jump = df[["spx", "vix"]].copy()
    jump["return"] = np.log(jump["spx"] / jump["spx"].shift(1))
    jump["dd_10"] = _ewm_downside_deviation(jump["return"], 10)
    jump["sortino_20"] = _ewm_sortino(jump["return"], 20)
    jump["sortino_60"] = _ewm_sortino(jump["return"], 60)
    jump = jump.replace([np.inf, -np.inf], np.nan).dropna(
        subset=JUMP_FEATURES + ["return", "vix"]
    )

    if len(jump) <= JUMP_TRAINING_WINDOW:
        raise StrategyError(
            f"Historique insuffisant pour le Jump Model : {len(jump)} observations."
        )

    train = jump.iloc[-(JUMP_TRAINING_WINDOW + 1):-1].copy()
    latest = jump.iloc[[-1]].copy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[JUMP_FEATURES])
    centroids, train_states = _fit_jump_model(
        X_train,
        jump_penalty=JUMP_PENALTY,
        n_init=JUMP_N_INIT,
    )

    state_return_sums = {
        state: float(train.loc[train_states == state, "return"].sum())
        for state in (0, 1)
    }
    bull_state = max(state_return_sums, key=state_return_sums.get)
    bear_state = 1 - bull_state

    # Classification causale du jour : coût de distance + coût éventuel
    # de changement par rapport au dernier état connu du train.
    X_latest = scaler.transform(latest[JUMP_FEATURES])[0]
    previous_state = int(train_states[-1])
    costs = np.array([
        0.5 * np.sum((X_latest - centroids[state]) ** 2)
        + JUMP_PENALTY * (state != previous_state)
        for state in (0, 1)
    ])
    latest_state = int(np.argmin(costs))
    regime = "Bull" if latest_state == bull_state else "Bear"

    labelled_train = train.copy()
    labelled_train["state"] = train_states
    labelled_train["regime"] = np.where(
        labelled_train["state"] == bull_state, "Bull", "Bear"
    )
    vix_means = labelled_train.groupby("regime")["vix"].mean()

    return {
        "regime": regime,
        "regime_date": latest.index[-1],
        "regime_vix": float(latest["vix"].iloc[0]),
        "bull_vix_mean": float(vix_means.get("Bull", np.nan)),
        "bear_vix_mean": float(vix_means.get("Bear", np.nan)),
        "jump_previous_regime": (
            "Bull" if previous_state == bull_state else "Bear"
        ),
        "jump_regime_changed": bool(latest_state != previous_state),
    }


def compute_daily_signal() -> dict:
    raw = download_market_data()
    original_last_dates = raw.attrs.get("original_last_dates", {})

    df = create_features(raw)
    jump_signal = compute_jump_regime(df)

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
    print(f"Régime Jump Model: {jump_signal['regime']}")
    print(f"VIX du jour: {jump_signal['regime_vix']:.2f}")
    print(f"VIX moyen Bull: {jump_signal['bull_vix_mean']:.2f}")
    print(f"VIX moyen Bear: {jump_signal['bear_vix_mean']:.2f}")

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
        "regime": jump_signal["regime"],
        "regime_date": jump_signal["regime_date"],
        "regime_vix": jump_signal["regime_vix"],
        "bull_vix_mean": jump_signal["bull_vix_mean"],
        "bear_vix_mean": jump_signal["bear_vix_mean"],
        "jump_previous_regime": jump_signal["jump_previous_regime"],
        "jump_regime_changed": jump_signal["jump_regime_changed"],
    }
