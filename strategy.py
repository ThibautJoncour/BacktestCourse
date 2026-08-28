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


CODE_VERSION = "NO_VIX3M_WITH_HAR_VRP_GARCH_2026-08-28"

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
    "vvix",
    "skew",
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
    VIX3M n'est volontairement plus téléchargé ni utilisé.
    """

    spx = _download_close("^GSPC", "spx")

    df = pd.DataFrame(index=spx.index.copy())
    df["spx"] = spx.reindex(df.index)

    tickers = {
        "vix": "^VIX",
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
    print(df[["spx", "vix", "vvix", "skew"]].tail(5))

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


def _print_dataset_diagnostics(
    raw: pd.DataFrame,
    df: pd.DataFrame,
    train: pd.DataFrame | None = None,
    prediction_rows: pd.DataFrame | None = None,
) -> dict:
    """
    Affiche dans les logs Shiny/Posit un diagnostic compact du pipeline.
    Permet d'identifier immédiatement quelle série ou feature vide le dataset.
    """
    print("\n" + "=" * 80)
    print("DIAGNOSTIC DATASET STRATEGY")
    print("=" * 80)

    print(f"RAW rows: {len(raw)}")
    if len(raw):
        print(f"RAW dates: {raw.index.min()} -> {raw.index.max()}")

    base_cols = [c for c in ["spx", "vix", "vvix", "skew"] if c in raw.columns]
    if base_cols:
        print("\nValeurs non-NaN des séries marché :")
        for c in base_cols:
            valid = int(raw[c].notna().sum())
            first = raw[c].first_valid_index()
            last = raw[c].last_valid_index()
            print(f"  {c:8s}: {valid:5d} | {first} -> {last}")

    print(f"\nFEATURE DATA rows: {len(df)}")
    if len(df):
        print(f"FEATURE DATA dates: {df.index.min()} -> {df.index.max()}")

    diagnostic_cols = [
        "spx", "vix", "vvix", "skew",
        *FEATURES, "future_rv_21", "vrp_points",
    ]
    diagnostic_cols = list(dict.fromkeys(c for c in diagnostic_cols if c in df.columns))

    print("\nNaN / valeurs valides par colonne :")
    for c in diagnostic_cols:
        n_nan = int(df[c].isna().sum())
        n_valid = int(df[c].notna().sum())
        print(f"  {c:20s}: valid={n_valid:5d} | NaN={n_nan:5d}")

    required_train = FEATURES + ["vrp_points"]
    if all(c in df.columns for c in required_train):
        complete_mask = df[required_train].notna().all(axis=1)
        print(f"\nLignes complètes FEATURES + vrp_points : {int(complete_mask.sum())}")
        if complete_mask.any():
            complete_idx = df.index[complete_mask]
            print(f"Plage complète : {complete_idx.min()} -> {complete_idx.max()}")
        else:
            print("AUCUNE ligne complète. Colonnes bloquantes :")
            for c in required_train:
                if df[c].notna().sum() == 0:
                    print(f"  - {c}: 0 valeur valide")

    if train is not None:
        print(f"TRAIN rows après dropna: {len(train)}")
    if prediction_rows is not None:
        print(f"PREDICTION rows après dropna: {len(prediction_rows)}")

    original_last_dates = raw.attrs.get("original_last_dates", {})
    if original_last_dates:
        print("\nDernières dates réellement publiées :")
        for name, date in original_last_dates.items():
            print(f"  {name}: {date}")

    print("=" * 80 + "\n")

    return {
        "raw_rows": int(len(raw)),
        "feature_rows": int(len(df)),
        "train_rows": None if train is None else int(len(train)),
        "prediction_rows": None if prediction_rows is None else int(len(prediction_rows)),
        "valid_counts": {
            c: int(df[c].notna().sum())
            for c in diagnostic_cols
        },
    }




# ============================================================
# HAR-RV / HAR-VIX / VRP / GARCH VIX
# ============================================================

def _har_rv_forecast_5d(df: pd.DataFrame) -> dict:
    """Prévoit la volatilité réalisée annualisée des 5 prochaines séances."""
    x = pd.DataFrame(index=df.index)
    x["rv_d"] = df["rv_5"]
    x["rv_w"] = df["rv_5"].rolling(5).mean()
    x["rv_m"] = df["rv_5"].rolling(21).mean()

    future_var_5 = (
        df["return_1d"].pow(2).shift(-1).rolling(5).sum().shift(-4)
    )
    target = np.sqrt(future_var_5 * 252 / 5) * 100

    data = x.copy()
    data["target"] = target
    train = data.replace([np.inf, -np.inf], np.nan).dropna()
    latest = x.replace([np.inf, -np.inf], np.nan).dropna().iloc[[-1]]

    if len(train) < 250 or latest.empty:
        raise StrategyError("Historique insuffisant pour HAR-RV.")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    model.fit(np.log(train[["rv_d", "rv_w", "rv_m"]].clip(lower=1e-6)),
              np.log(train["target"].clip(lower=1e-6)))
    pred = float(np.exp(model.predict(np.log(latest.clip(lower=1e-6)))[0]))

    return {
        "rv_forecast_5d_pct": pred,
        "har_rv_train_rows": int(len(train)),
    }


def _har_vix_forecast_5d(df: pd.DataFrame) -> dict:
    """Prévoit le niveau du VIX à J+5 avec une structure HAR."""
    vix = df["vix"].astype(float)
    x = pd.DataFrame(index=df.index)
    x["vix_d"] = vix
    x["vix_w"] = vix.rolling(5).mean()
    x["vix_m"] = vix.rolling(21).mean()
    target = vix.shift(-5)

    data = x.copy()
    data["target"] = target
    train = data.replace([np.inf, -np.inf], np.nan).dropna()
    latest = x.replace([np.inf, -np.inf], np.nan).dropna().iloc[[-1]]

    if len(train) < 250 or latest.empty:
        raise StrategyError("Historique insuffisant pour HAR-VIX.")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    model.fit(train[["vix_d", "vix_w", "vix_m"]], train["target"])
    pred = float(model.predict(latest)[0])
    pred = max(pred, 0.01)

    return {
        "vix_forecast_5d_pct": pred,
        "har_vix_train_rows": int(len(train)),
    }


def _garch_vix_compression(df: pd.DataFrame, quantile: float = 0.25) -> dict:
    """
    GARCH(1,1) léger sur les variations logarithmiques du VIX.
    Retourne sigma conditionnel annualisé et un seuil historique de compression.
    Pas de dépendance au package `arch`.
    """
    from scipy.optimize import minimize

    vix = df["vix"].dropna().astype(float)
    r = (100.0 * np.log(vix / vix.shift(1))).replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 500:
        raise StrategyError("Historique insuffisant pour le GARCH VIX.")

    # Fenêtre bornée pour garder le calcul rapide sur Posit.
    r = r.iloc[-2500:]
    y = r.to_numpy(dtype=float)
    var0 = max(float(np.var(y)), 1e-6)

    def unpack(theta):
        # Paramétrisation garantissant omega>0, alpha>=0, beta>=0, alpha+beta<1.
        omega = np.exp(theta[0])
        ea, eb = np.exp(theta[1]), np.exp(theta[2])
        denom = 1.0 + ea + eb
        alpha = 0.999 * ea / denom
        beta = 0.999 * eb / denom
        return omega, alpha, beta

    def nll(theta):
        omega, alpha, beta = unpack(theta)
        h = np.empty_like(y)
        h[0] = var0
        for t in range(1, len(y)):
            h[t] = omega + alpha * y[t - 1] ** 2 + beta * h[t - 1]
            if not np.isfinite(h[t]) or h[t] <= 1e-12:
                return 1e20
        return 0.5 * float(np.sum(np.log(h) + y * y / h))

    init = np.array([np.log(var0 * 0.05 + 1e-8), np.log(0.08), np.log(0.90)])
    opt = minimize(nll, init, method="L-BFGS-B", options={"maxiter": 300})
    omega, alpha, beta = unpack(opt.x if opt.success else init)

    h = np.empty_like(y)
    h[0] = var0
    for t in range(1, len(y)):
        h[t] = omega + alpha * y[t - 1] ** 2 + beta * h[t - 1]

    next_var = omega + alpha * y[-1] ** 2 + beta * h[-1]
    # r est en % journalier; annualisation en % vol.
    sigma_series = np.sqrt(h) * np.sqrt(252)
    sigma_next = float(np.sqrt(max(next_var, 1e-12)) * np.sqrt(252))
    threshold = float(np.nanquantile(sigma_series, quantile))
    compression = bool(sigma_next <= threshold)

    return {
        "garch_sigma": sigma_next,
        "garch_sigma_pct": sigma_next,
        "garch_threshold": threshold,
        "garch_threshold_pct": threshold,
        "garch_compression": compression,
        "compression_garch_vix": compression,
        "garch_alpha": float(alpha),
        "garch_beta": float(beta),
        "garch_omega": float(omega),
    }


def compute_daily_signal() -> dict:
    raw = download_market_data()
    original_last_dates = raw.attrs.get("original_last_dates", {})

    df = create_features(raw)
    jump_signal = compute_jump_regime(df)

    train = df[FEATURES + ["vrp_points"]].dropna().copy()
    prediction_rows = df[FEATURES].dropna()

    diagnostics = _print_dataset_diagnostics(
        raw=raw,
        df=df,
        train=train,
        prediction_rows=prediction_rows,
    )

    if len(train) < 750:
        zero_valid = [
            name
            for name, count in diagnostics["valid_counts"].items()
            if count == 0
        ]
        blocking_text = (
            " Colonnes sans aucune valeur valide : " + ", ".join(zero_valid) + "."
            if zero_valid
            else " Consultez les logs Posit pour le détail des NaN par colonne."
        )
        raise StrategyError(
            f"Historique insuffisant : {len(train)} observations."
            f" RAW={diagnostics['raw_rows']}, "
            f"features={diagnostics['feature_rows']}, "
            f"prediction_rows={diagnostics['prediction_rows']}."
            + blocking_text
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

    # HAR / VRP / GARCH : indépendants de VIX3M.
    har_rv = _har_rv_forecast_5d(df)
    har_vix = _har_vix_forecast_5d(df)
    garch = _garch_vix_compression(df)

    iv_current_vol_pct = float(df.loc[latest_date, "vix"])
    rv_forecast_5d_pct = float(har_rv["rv_forecast_5d_pct"])
    vix_forecast_5d_pct = float(har_vix["vix_forecast_5d_pct"])
    vrp_current_pct = iv_current_vol_pct - rv_forecast_5d_pct
    vrp_forecast_5d_pct = vix_forecast_5d_pct - rv_forecast_5d_pct

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
    print(f"SPX: {current_spx:.2f}")
    print(f"Score Ensemble: {ensemble_score:.3f}")
    print(f"Seuil: {threshold:.3f}")
    print(f"Prendre position: {take_position}")
    print(f"Régime Jump Model: {jump_signal['regime']}")
    print(f"VIX du jour: {jump_signal['regime_vix']:.2f}")
    print(f"VIX moyen Bull: {jump_signal['bull_vix_mean']:.2f}")
    print(f"VIX moyen Bear: {jump_signal['bear_vix_mean']:.2f}")
    print(f"IV actuelle (VIX): {iv_current_vol_pct:.2f}%")
    print(f"RV prévue 5j (HAR-RV): {rv_forecast_5d_pct:.2f}%")
    print(f"VRP avec IV actuelle: {vrp_current_pct:.2f} pts")
    print(f"VIX prévu 5j (HAR-VIX): {vix_forecast_5d_pct:.2f}")
    print(f"VRP future prévue 5j: {vrp_forecast_5d_pct:.2f} pts")
    print(f"Sigma GARCH VIX: {garch['garch_sigma']:.2f}%")
    print(f"Seuil compression GARCH: {garch['garch_threshold']:.2f}%")
    print(f"Compression GARCH VIX: {garch['garch_compression']}")

    # Historique VRP pour le graphique Shiny.
    # On utilise des séries 5 jours cohérentes avec les cartes HAR, sans VIX3M.
    hist = pd.DataFrame(index=df.index)
    hist["Date"] = hist.index
    hist["rv_5d_realized"] = df["return_1d"].pow(2).shift(-1).rolling(5).sum().shift(-4)
    hist["rv_5d_realized"] = np.sqrt(hist["rv_5d_realized"] * 252 / 5) * 100
    # Prévision historique HAR simple/proxy basée uniquement sur l'information disponible à t.
    hist["rv_pred"] = (
        0.50 * df["rv_5"] + 0.30 * df["rv_21"] + 0.20 * df["rv_63"]
    )
    hist["vix_pred"] = (
        0.50 * df["vix"]
        + 0.30 * df["vix"].rolling(5).mean()
        + 0.20 * df["vix"].rolling(21).mean()
    )
    # L'app appelle ces colonnes "points de variance"; on conserve ici l'échelle en points de vol
    # utilisée par les cartes actuelles (VIX - RV), pour rester cohérent avec compute_daily_signal.
    hist["VRP_tradable_pred_points"] = df["vix"] - hist["rv_pred"]
    hist["VRP_future_pred_points"] = hist["vix_pred"] - hist["rv_pred"]
    hist["VRP_tradable_real_points"] = df["vix"] - hist["rv_5d_realized"]
    vrp_history = (
        hist[["Date", "VRP_tradable_pred_points", "VRP_future_pred_points", "VRP_tradable_real_points"]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .tail(500)
        .reset_index(drop=True)
    )

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
        # HAR / VRP / GARCH -- clés attendues par l'app Shiny
        "iv_current_vol_pct": iv_current_vol_pct,
        "rv_forecast_5d_pct": rv_forecast_5d_pct,
        "vrp_current_pct": vrp_current_pct,
        "vix_forecast_5d_pct": vix_forecast_5d_pct,
        "vrp_forecast_5d_pct": vrp_forecast_5d_pct,
        # Noms EXACTS attendus par app.py
        "har_date": latest_date,
        "rv_forecast_vol_pct": rv_forecast_5d_pct,
        "vrp_current_iv_forecast_points": vrp_current_pct,
        "vrp_future_forecast_points": vrp_forecast_5d_pct,
        "garch_compression_threshold": garch["garch_threshold"],
        # Alias pour compatibilité avec d'anciennes versions de l'UI
        "iv_current": iv_current_vol_pct,
        "rv_forecast_5d": rv_forecast_5d_pct,
        "vrp_current": vrp_current_pct,
        "vix_forecast_5d": vix_forecast_5d_pct,
        "vrp_future_5d": vrp_forecast_5d_pct,
        "vrp_history": vrp_history,
        **har_rv,
        **har_vix,
        **garch,
        "regime": jump_signal["regime"],
        "regime_date": jump_signal["regime_date"],
        "regime_vix": jump_signal["regime_vix"],
        "bull_vix_mean": jump_signal["bull_vix_mean"],
        "bear_vix_mean": jump_signal["bear_vix_mean"],
        "jump_previous_regime": jump_signal["jump_previous_regime"],
        "jump_regime_changed": jump_signal["jump_regime_changed"],
    }
