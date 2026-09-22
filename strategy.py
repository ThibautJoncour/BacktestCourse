from __future__ import annotations

import re

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CODE_VERSION = "NVDA_NEWS_RISK_5D_HAR_SIGMA_2026-09-18"

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

# ============================================================
# STABILITY NVIDIA - MONTE CARLO AVEC SIGMA HAR-RV
# ============================================================
NVDA_TICKER = "NVDA"
NVDA_REDEMPTION = 10.0
NVDA_MC_PATHS = 50_000
NVDA_RISK_FREE_RATE = 0.02
NVDA_DIVIDEND_YIELD = 0.0
NVDA_MC_SEED = 42

# BEGIN AUTO-UPDATED NVDA NEWS
# Faits datés et sourcés, réévalués quotidiennement par l'automatisation.
# direction : +1 = favorable à NVDA ; -1 = défavorable à NVDA.
NVDA_NEWS_LAST_UPDATE = "2026-09-22"
NVDA_NEWS_EVENTS = [
    {
        "date": "2026-09-22",
        "days_ahead": 3,
        "category": "Infrastructure IA",
        "headline": "Accelevation vise jusqu'à 720 millions de dollars lors de son introduction en Bourse, portée par la demande de systèmes d'alimentation et de refroidissement pour data centers.",
        "direction": 1,
        "risk_level": "moyen",
        "confidence": 0.85,
        "source": "Reuters",
        "url": "https://www.reuters.com/technology/accelevation-backers-aim-raise-720-million-us-ipo-2026-09-22/",
    },
    {
        "date": "2026-09-22",
        "days_ahead": 2,
        "category": "Concurrence / Chine",
        "headline": "Alibaba dévoile la puce IA Zhenwu V900, annoncée trois fois plus performante que sa devancière, et vise une production de masse début 2027.",
        "direction": -1,
        "risk_level": "élevé",
        "confidence": 0.95,
        "source": "Reuters",
        "url": "https://www.reuters.com/business/retail-consumer/alibaba-plans-ai-model-with-5-trillion-10-trillion-parameters-unveils-new-chip-2026-09-22/",
    },
    {
        "date": "2026-09-22",
        "days_ahead": 4,
        "category": "Énergie des data centers",
        "headline": "L'État australien de Victoria propose d'obliger les nouveaux data centers à sécuriser leur propre énergie renouvelable et leur stockage, et à financer les raccordements au réseau.",
        "direction": -1,
        "risk_level": "moyen",
        "confidence": 0.85,
        "source": "Reuters",
        "url": "https://www.reuters.com/business/energy/australias-victoria-proposes-new-data-centres-must-secure-their-own-renewable-2026-09-22/",
    },
    {
        "date": "2026-09-21",
        "days_ahead": 1,
        "category": "Concurrence semi-conducteurs",
        "headline": "AMD dépasse 1 000 milliards de dollars de capitalisation sur l'optimisme lié à l'IA et renforce son positionnement de concurrent direct de NVIDIA dans les GPU.",
        "direction": -1,
        "risk_level": "moyen",
        "confidence": 0.80,
        "source": "Reuters",
        "url": "https://www.reuters.com/business/amd-becomes-latest-chipmaker-reach-1-trillion-valuation-ai-demand-2026-09-21/",
    },
    {
        "date": "2026-09-21",
        "days_ahead": 2,
        "category": "Énergie des data centers",
        "headline": "Le gouverneur du Texas suspend les nouveaux permis d'État pour les data centers jusqu'à l'achèvement d'un audit de leur impact sur le réseau électrique.",
        "direction": -1,
        "risk_level": "élevé",
        "confidence": 0.95,
        "source": "Reuters",
        "url": "https://www.reuters.com/business/energy/texas-gov-abbott-halts-all-state-issued-permits-data-centers-until-grid-audit-is-2026-09-21/",
    },
    {
        "date": "2026-09-21",
        "days_ahead": 3,
        "category": "Contrôles à l'exportation / Chine",
        "headline": "Les États-Unis et la Chine conviennent de poursuivre sous deux mois leur dialogue sur la sécurité de l'IA et les protocoles de communication d'urgence.",
        "direction": 1,
        "risk_level": "moyen",
        "confidence": 0.80,
        "source": "Reuters",
        "url": "https://www.reuters.com/world/asia-pacific/us-china-meet-again-ai-safety-two-months-shenzhen-bessent-says-2026-09-21/",
    },
    {
        "date": "2026-09-22",
        "days_ahead": 1,
        "category": "Taux US / marchés",
        "headline": "Le rendement du Treasury à 10 ans reste proche de 4,9 % et le marché estime à environ 51 % la probabilité d'une nouvelle hausse de taux en octobre, tandis que NVIDIA recule légèrement.",
        "direction": -1,
        "risk_level": "élevé",
        "confidence": 0.90,
        "source": "Reuters",
        "url": "https://www.reuters.com/business/wall-st-futures-pause-after-ai-rally-focus-mideast-tensions-2026-09-22/",
    },
]
# END AUTO-UPDATED NVDA NEWS


# Source SG Bourse : la liste des Stability NVIDIA est récupérée automatiquement.
NVDA_STABILITY_URL = (
    "https://sgbourse.fr/product-search/stability/all-assettypes/"
    "nvidia/all-assets?ShowFilter=false"
)
NVDA_SG_TIMEOUT = 15

# Prix connus conservés uniquement comme secours si SG ne renvoie pas de cotation
# dans le HTML au moment du refresh. Les barrières/maturités restent récupérées du site.

# Catalogue de secours synchronisé avec la page publique SG Bourse le 22/09/2026
# (36 produits).
# Utilisé uniquement lorsque le HTML retourné à requests ne contient pas le tableau
# (le site SG charge parfois les lignes côté navigateur via JavaScript).
NVDA_FALLBACK_PRODUCTS = [
    ("7C57S", "DE000FE35392", 140, 280, "2026-10-16"),
    ("7C55S", "DE000FE35376", 100, 280, "2026-10-16"),
    ("2R74S", "DE000FE6NXC6", 160, 260, "2026-10-16"),
    ("2R79S", "DE000FE6NXH5", 140, 260, "2026-11-20"),
    ("2R83S", "DE000FE6NXM5", 160, 280, "2026-11-20"),
    ("2R85S", "DE000FE6NXP8", 160, 320, "2026-11-20"),
    ("2R80S", "DE000FE6NXJ1", 140, 280, "2026-11-20"),
    ("2R82S", "DE000FE6NXL7", 140, 320, "2026-11-20"),
    ("2R78S", "DE000FE6NXG7", 120, 320, "2026-11-20"),
    ("2R84S", "DE000FE6NXN3", 160, 300, "2026-11-20"),
    ("2R81S", "DE000FE6NXK9", 140, 300, "2026-11-20"),
    ("2R77S", "DE000FE6NXF9", 120, 300, "2026-11-20"),
    ("2R75S", "DE000FE6NXD4", 120, 260, "2026-11-20"),
    ("2R76S", "DE000FE6NXE2", 120, 280, "2026-11-20"),
    ("2R95S", "DE000FE6NXY0", 140, 340, "2026-12-18"),
    ("2R89S", "DE000FE6NXT0", 120, 320, "2026-12-18"),
    ("01M3S", "DE000FG2S7Z0", 100, 240, "2026-12-18"),
    ("2R90S", "DE000FE6NXU8", 120, 340, "2026-12-18"),
    ("2R96S", "DE000FE6NXZ7", 160, 300, "2026-12-18"),
    ("2R87S", "DE000FE6NXR4", 120, 280, "2026-12-18"),
    ("2R93S", "DE000FE6NXW4", 140, 300, "2026-12-18"),
    ("2R98S", "DE000FE6NX14", 160, 340, "2026-12-18"),
    ("3J41S", "DE000FG340X0", 200, 250, "2026-12-18"),
    ("2R86S", "DE000FE6NXQ6", 120, 260, "2026-12-18"),
    ("2R91S", "DE000FE6NXV6", 140, 280, "2026-12-18"),
    ("2R97S", "DE000FE6NX06", 160, 320, "2026-12-18"),
    ("2R94S", "DE000FE6NXX2", 140, 320, "2026-12-18"),
    ("2R88S", "DE000FE6NXS2", 120, 300, "2026-12-18"),
    ("5S06S", "DE000FG1E5H9", 140, 280, "2027-01-15"),
    ("5S04S", "DE000FG1E5F3", 100, 240, "2027-01-15"),
    ("5S07S", "DE000FG1E5J5", 160, 300, "2027-01-15"),
    ("5S05S", "DE000FG1E5G1", 120, 260, "2027-01-15"),
    ("917QS", "DE000FG4Z577", 170, 240, "2027-02-19"),
    ("1U26S", "DE000FG3T895", 120, 260, "2027-02-19"),
    ("1U28S", "DE000FG3T9B5", 160, 320, "2027-02-19"),
    ("1U27S", "DE000FG3T9A7", 140, 300, "2027-02-19"),
]

NVDA_FALLBACK_MARKET_PRICES = {
    "7C57S": 9.88, "7C55S": 9.90, "2R74S": 9.55, "2R79S": 7.37,
    "2R83S": 8.78, "2R85S": 9.47, "2R80S": 8.99, "2R82S": 9.67,
    "2R78S": 9.75, "2R84S": 9.31, "2R81S": 9.51, "2R77S": 9.59,
    "2R75S": 7.47, "2R76S": 9.07, "2R95S": 9.54, "2R89S": 9.54,
    "01M3S": 3.20, "2R90S": 9.66, "2R96S": 8.74, "2R87S": 8.37,
    "2R93S": 9.09, "2R98S": 9.18, "3J41S": 1.85, "2R86S": 6.46,
    "2R91S": 8.25, "2R97S": 9.06, "2R94S": 9.41, "2R88S": 9.22,
    "5S06S": 7.45, "5S04S": 2.77, "5S07S": 8.01, "5S05S": 5.58,
    "917QS": 1.02, "1U26S": 4.97, "1U28S": 7.79, "1U27S": 7.80,
}

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
# NVIDIA : HAR-RV 1 à 5 jours + PRICING MONTE CARLO STABILITY
# ============================================================


def compute_nvda_news_indicator(nvda_har: dict, spot: float) -> dict:
    """Agrège les nouvelles NVDA et l'écart-type HAR-RV sur cinq séances."""
    risk_weights = {"faible": 0.35, "moyen": 0.65, "élevé": 1.00}
    time_weights = {1: 1.00, 2: 0.90, 3: 0.80, 4: 0.70, 5: 0.60}
    rows: list[dict] = []
    weighted_sum = 0.0
    total_weight = 0.0
    risk_sum = 0.0

    for event in NVDA_NEWS_EVENTS:
        direction = 1 if float(event.get("direction", 0)) > 0 else -1
        risk_level = str(event.get("risk_level", "moyen")).lower()
        risk_weight = risk_weights.get(risk_level, risk_weights["moyen"])
        confidence = float(np.clip(event.get("confidence", 0.75), 0.0, 1.0))
        days_ahead = int(np.clip(event.get("days_ahead", 1), 1, 5))
        absolute_weight = risk_weight * confidence * time_weights[days_ahead]
        contribution = direction * absolute_weight
        weighted_sum += contribution
        total_weight += absolute_weight
        risk_sum += risk_weight * confidence
        rows.append({
            "Date": event.get("date", ""),
            "J+": days_ahead,
            "Sens": "+1" if direction > 0 else "-1",
            "Risque": risk_level.capitalize(),
            "Catégorie": event.get("category", ""),
            "News": event.get("headline", ""),
            "Source": event.get("source", ""),
            "URL": event.get("url", ""),
            "Contribution": contribution,
        })

    score = float(weighted_sum / total_weight) if total_weight > 0 else 0.0
    signal = 0 if not rows else (1 if score >= 0.0 else -1)
    signal_label = "HAUSSIER +1" if signal > 0 else ("BAISSIER -1" if signal < 0 else "NEUTRE 0")
    average_risk = risk_sum / len(rows) if rows else 0.0
    aggregate_risk = "élevé" if average_risk >= 0.75 else ("moyen" if average_risk >= 0.45 else "faible")

    forecasts = nvda_har.get("nvda_har_rv_forecasts", {})
    forecast_values = [
        float(forecasts[h]) for h in sorted(forecasts)
        if h <= 5 and np.isfinite(float(forecasts[h]))
    ]
    if forecast_values:
        daily_sigmas = np.asarray(forecast_values, dtype=float) / 100.0 / np.sqrt(252.0)
        sigma_5d_move_pct = float(np.sqrt(np.sum(daily_sigmas**2)) * 100.0)
        sigma_5d_ann_pct = float(
            sigma_5d_move_pct / 100.0 * np.sqrt(252.0 / len(daily_sigmas)) * 100.0
        )
    else:
        sigma_5d_ann_pct = float(nvda_har.get("nvda_har_rv_5d_pct", np.nan))
        sigma_5d_move_pct = float(sigma_5d_ann_pct * np.sqrt(5.0 / 252.0))

    sigma_5d_price = float(spot * sigma_5d_move_pct / 100.0)
    events_table = pd.DataFrame(rows)
    if not events_table.empty:
        events_table["Contribution"] = events_table["Contribution"].map(lambda x: f"{x:+.2f}")
        events_table = events_table.sort_values(["J+", "Date"]).reset_index(drop=True)

    return {
        "nvda_news_last_update": NVDA_NEWS_LAST_UPDATE,
        "nvda_news_score_continuous": score,
        "nvda_news_signal": signal,
        "nvda_news_label": signal_label,
        "nvda_news_risk_level": aggregate_risk,
        "nvda_news_event_count": len(rows),
        "nvda_news_events_table": events_table,
        "nvda_sigma_5d_ann_pct": sigma_5d_ann_pct,
        "nvda_sigma_5d_move_pct": sigma_5d_move_pct,
        "nvda_sigma_5d_price": sigma_5d_price,
    }


def _sg_number(text: str) -> float | None:
    """Convertit un nombre SG français (ex. '9,84 EUR') en float."""
    if text is None:
        return None
    cleaned = (
        str(text)
        .replace("\xa0", " ")
        .replace("€", "")
        .replace("EUR", "")
        .replace("USD", "")
        .strip()
    )
    match = re.search(r"[-+]?\d+(?:[\s.]\d{3})*(?:,\d+)?|[-+]?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    token = match.group(0).replace(" ", "")
    # Format français : point éventuel de milliers, virgule décimale.
    if "," in token:
        token = token.replace(".", "").replace(",", ".")
    try:
        return float(token)
    except ValueError:
        return None


def _parse_sg_stability_html(html: str) -> list[dict]:
    """Extrait les lignes NVIDIA Stability présentes dans une page SG Bourse."""
    soup = BeautifulSoup(html, "html.parser")
    out: list[dict] = []

    for table in soup.find_all("table"):
        headers = [
            " ".join(th.get_text(" ", strip=True).lower().split())
            for th in table.find_all("th")
        ]
        if not headers or not any("mnémo" in h or "mnemo" in h for h in headers):
            continue

        # Mappe les colonnes par libellé pour résister aux changements de mise en page.
        def idx_contains(*needles):
            for i, h in enumerate(headers):
                if any(n in h for n in needles):
                    return i
            return None

        i_code = idx_contains("mnémo", "mnemo")
        i_isin = idx_contains("isin")
        i_low = idx_contains("barrière", "barriere")
        i_high = idx_contains("borne haute")
        i_mat = idx_contains("maturité", "maturite")
        i_buy = idx_contains("achat")
        i_sell = idx_contains("vente")

        required = [i_code, i_low, i_high, i_mat]
        if any(i is None for i in required):
            continue

        for tr in table.find_all("tr"):
            cells = [" ".join(td.get_text(" ", strip=True).split()) for td in tr.find_all("td")]
            if not cells or max(i for i in required if i is not None) >= len(cells):
                continue

            code_text = cells[i_code]
            code_match = re.search(r"\b([A-Z0-9]{4,6}S)\b", code_text.upper())
            if not code_match:
                continue
            code = code_match.group(1)

            low = _sg_number(cells[i_low])
            high = _sg_number(cells[i_high])
            date_match = re.search(r"(\d{2})/(\d{2})/(\d{4})", cells[i_mat])
            if low is None or high is None or not date_match:
                continue
            dd, mm, yyyy = date_match.groups()
            maturity = f"{yyyy}-{mm}-{dd}"

            isin = ""
            if i_isin is not None and i_isin < len(cells):
                m = re.search(r"\b[A-Z]{2}[A-Z0-9]{10}\b", cells[i_isin].upper())
                isin = m.group(0) if m else cells[i_isin].replace(" KO", "").strip()

            buy = _sg_number(cells[i_buy]) if i_buy is not None and i_buy < len(cells) else None
            sell = _sg_number(cells[i_sell]) if i_sell is not None and i_sell < len(cells) else None

            # Pour acheter le warrant, on privilégie la cotation d'achat affichée par SG.
            # Si elle est absente, on prend Vente, puis le dernier prix de secours connu.
            market_price = buy if buy is not None else sell
            price_source = "SG live HTML"
            if market_price is None:
                market_price = NVDA_FALLBACK_MARKET_PRICES.get(code)
                price_source = "fallback ancien prix" if market_price is not None else "indisponible"

            out.append({
                "Code": code,
                "ISIN": isin,
                "Barriere_basse": float(low),
                "Barriere_haute": float(high),
                "Maturite": maturity,
                "Prix_marche": None if market_price is None else float(market_price),
                "Source_prix": price_source,
            })

    return out


def fetch_nvda_stability_products() -> pd.DataFrame:
    """
    Récupère automatiquement les Stability NVIDIA depuis SG Bourse.

    Plusieurs variantes de pagination sont essayées car le site peut changer son
    paramétrage front. Les doublons sont éliminés par mnémo. Si le site est
    indisponible, une StrategyError explicite est levée plutôt que d'utiliser
    silencieusement une vieille liste de barrières.
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
        ),
        "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
    })

    base = NVDA_STABILITY_URL
    candidate_urls = [
        base,
        base + "&PageSize=100",
        base + "&pageSize=100",
        base + "&itemsPerPage=100",
        base + "&limit=100",
        base + "&page=2",
        base + "&Page=2",
        base + "&pageNumber=2",
    ]

    rows: dict[str, dict] = {}
    errors = []
    expected_total = None

    for url in candidate_urls:
        try:
            response = session.get(url, timeout=NVDA_SG_TIMEOUT)
            response.raise_for_status()
            html = response.text
            total_match = re.search(r"Stability\s*\(?\s*(\d+)\s*produits", html, flags=re.I)
            if total_match:
                expected_total = int(total_match.group(1))
            for row in _parse_sg_stability_html(html):
                rows[row["Code"]] = row
            if expected_total is not None and len(rows) >= expected_total:
                break
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    if not rows:
        # SG renvoie parfois uniquement le shell HTML à `requests`, puis remplit le
        # tableau côté navigateur en JavaScript. Dans ce cas on ne fait plus tomber
        # toute l'application : on reprend le dernier catalogue SG embarqué.
        fallback_rows = []
        for code, isin, low, high, maturity in NVDA_FALLBACK_PRODUCTS:
            px = NVDA_FALLBACK_MARKET_PRICES.get(code)
            fallback_rows.append({
                "Code": code,
                "ISIN": isin,
                "Barriere_basse": float(low),
                "Barriere_haute": float(high),
                "Maturite": maturity,
                "Prix_marche": np.nan if px is None else float(px),
                "Source_prix": "fallback SG 21/09/2026" if px is not None else "prix indisponible",
            })
        products = pd.DataFrame(fallback_rows)
        products.attrs["expected_total"] = 36
        products.attrs["scraped_total"] = 0
        products.attrs["complete"] = False
        products.attrs["source_mode"] = "fallback_catalogue"
        products.attrs["scrape_error"] = errors[-1] if errors else "aucune ligne HTML reconnue"
        return products

    products = pd.DataFrame(rows.values())
    products["Maturite"] = pd.to_datetime(products["Maturite"], errors="coerce")
    products = products.dropna(subset=["Maturite", "Barriere_basse", "Barriere_haute"])
    products = products.sort_values(["Maturite", "Barriere_basse", "Barriere_haute", "Code"])
    products["Maturite"] = products["Maturite"].dt.strftime("%Y-%m-%d")
    products = products.reset_index(drop=True)

    products.attrs["expected_total"] = expected_total
    products.attrs["scraped_total"] = len(products)
    products.attrs["complete"] = expected_total is None or len(products) >= expected_total
    return products

def _download_nvidia_history() -> pd.DataFrame:
    """Télécharge NVDA et construit les composantes HAR sur variance quotidienne."""
    nvda = _download_close(NVDA_TICKER, "nvda")
    out = pd.DataFrame({"nvda": nvda})
    out["return_1d"] = np.log(out["nvda"] / out["nvda"].shift(1))

    # Avec des closes quotidiens (pas d'intraday), la variance réalisée 1j est
    # approchée par r_t² annualisé. Les composantes HAR hebdo/mensuelle sont des
    # moyennes de cette variance, ce qui permet une vraie récursion J+1 -> J+5.
    out["rv_var_d"] = out["return_1d"].pow(2) * 252.0
    out["rv_var_w"] = out["rv_var_d"].rolling(5).mean()
    out["rv_var_m"] = out["rv_var_d"].rolling(21).mean()

    # Colonnes en volatilité annualisée (%) conservées pour diagnostic/affichage.
    out["rv_d"] = np.sqrt(out["rv_var_d"].clip(lower=0.0)) * 100.0
    out["rv_w"] = np.sqrt(out["rv_var_w"].clip(lower=0.0)) * 100.0
    out["rv_m"] = np.sqrt(out["rv_var_m"].clip(lower=0.0)) * 100.0
    return out.replace([np.inf, -np.inf], np.nan)


def _fit_har_one_step_nvda(train: pd.DataFrame) -> Pipeline:
    """HAR one-step sur log-variance annualisée."""
    data = train[["rv_var_d", "rv_var_w", "rv_var_m"]].copy()
    data["target"] = train["rv_var_d"].shift(-1)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 250:
        raise StrategyError(f"Historique insuffisant pour HAR-RV NVDA : {len(data)} lignes.")

    X = np.log(data[["rv_var_d", "rv_var_w", "rv_var_m"]].clip(lower=1e-12))
    y = np.log(data["target"].clip(lower=1e-12))
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    model.fit(X, y)
    return model


def _recursive_har_variance_path_nvda(
    history: pd.DataFrame,
    model: Pipeline,
    horizon: int = 5,
) -> list[float]:
    """Prévoit récursivement la variance annualisée des prochains jours."""
    realized = history["rv_var_d"].dropna().astype(float).tolist()
    if len(realized) < 21:
        raise StrategyError("Historique NVDA insuffisant pour initialiser le HAR récursif.")

    path = []
    eps = 1e-12
    for _ in range(horizon):
        x = pd.DataFrame([{
            "rv_var_d": max(realized[-1], eps),
            "rv_var_w": max(float(np.mean(realized[-5:])), eps),
            "rv_var_m": max(float(np.mean(realized[-21:])), eps),
        }])
        pred_log = float(model.predict(np.log(x.clip(lower=eps)))[0])
        pred_var = float(np.exp(np.clip(pred_log, -20, 10)))
        pred_var = max(pred_var, eps)
        path.append(pred_var)
        realized.append(pred_var)
    return path


def _qlike(actual_var: np.ndarray, forecast_var: np.ndarray) -> float:
    """QLIKE = E[log(forecast) + actual/forecast], constante omise."""
    eps = 1e-12
    a = np.clip(np.asarray(actual_var, dtype=float), eps, None)
    f = np.clip(np.asarray(forecast_var, dtype=float), eps, None)
    return float(np.mean(np.log(f) + a / f))


def _walk_forward_har_nvda(
    nvda: pd.DataFrame,
    max_horizon: int = 5,
    eval_days: int = 756,
    refit_every: int = 21,
) -> pd.DataFrame:
    """
    Backtest expanding-window strictement out-of-sample.

    À chaque date d'origine t, on ne voit que les données <= t, on estime le HAR
    (ré-estimation toutes les `refit_every` séances), puis on génère récursivement
    J+1...J+5. Les erreurs sont comparées à la variance quotidienne effectivement
    réalisée à t+h, donc J+5 n'est pas artificiellement lissé par une moyenne 5j.
    """
    clean = nvda[["rv_var_d", "rv_var_w", "rv_var_m"]].copy()
    valid_pos = np.flatnonzero(clean["rv_var_m"].notna().to_numpy())
    if len(valid_pos) < 500:
        raise StrategyError("Historique NVDA insuffisant pour le backtest walk-forward HAR.")

    first_origin = max(int(valid_pos[0]) + 260, len(clean) - int(eval_days) - max_horizon)
    last_origin = len(clean) - max_horizon - 1
    records = []
    model = None
    last_refit_origin = None

    for origin in range(first_origin, last_origin + 1):
        if model is None or last_refit_origin is None or (origin - last_refit_origin) >= refit_every:
            model = _fit_har_one_step_nvda(clean.iloc[: origin + 1])
            last_refit_origin = origin

        pred_path = _recursive_har_variance_path_nvda(clean.iloc[: origin + 1], model, max_horizon)
        for h, pred_var in enumerate(pred_path, start=1):
            actual_var = clean["rv_var_d"].iloc[origin + h]
            if pd.notna(actual_var):
                records.append({
                    "Horizon_jours": h,
                    "pred_var": float(pred_var),
                    "actual_var": float(actual_var),
                })

    bt = pd.DataFrame(records)
    if bt.empty:
        raise StrategyError("Backtest HAR-RV NVDA vide.")

    rows = []
    for h in range(1, max_horizon + 1):
        d = bt[bt["Horizon_jours"] == h].copy()
        pred_vol = np.sqrt(d["pred_var"].clip(lower=0.0)) * 100.0
        act_vol = np.sqrt(d["actual_var"].clip(lower=0.0)) * 100.0
        err = pred_vol - act_vol
        rows.append({
            "Horizon_jours": h,
            "OOS_n": int(len(d)),
            "MAE_vol_pct": float(np.mean(np.abs(err))),
            "RMSE_vol_pct": float(np.sqrt(np.mean(err ** 2))),
            "QLIKE": _qlike(d["actual_var"].to_numpy(), d["pred_var"].to_numpy()),
            "Bias_vol_pct": float(np.mean(err)),
        })
    return pd.DataFrame(rows)


def _har_rv_forecast_horizons_nvda(nvda: pd.DataFrame, max_horizon: int = 5) -> dict:
    """
    HAR-RV NVDA one-step puis prévisions récursives J+1 à J+5.

    Un seul modèle J+1 est estimé. J+2...J+5 sont obtenus en réinjectant la
    variance prévue dans les composantes quotidienne/hebdomadaire/mensuelle.
    La précision est mesurée par un backtest walk-forward out-of-sample avec
    MAE, RMSE et QLIKE, et non par un R² in-sample.
    """
    train = nvda.dropna(subset=["rv_var_d", "rv_var_w", "rv_var_m"])
    if len(train) < 250:
        raise StrategyError("Historique insuffisant pour HAR-RV NVDA.")

    model = _fit_har_one_step_nvda(nvda)
    var_path = _recursive_har_variance_path_nvda(nvda, model, max_horizon)
    forecasts = {h: float(np.sqrt(v) * 100.0) for h, v in enumerate(var_path, start=1)}

    metrics = _walk_forward_har_nvda(nvda, max_horizon=max_horizon)
    table = pd.DataFrame({
        "Horizon_jours": list(forecasts.keys()),
        "HAR_RV_pred_pct_ann": list(forecasts.values()),
    }).merge(metrics, on="Horizon_jours", how="left")

    # Niveau de long terme robuste utilisé après J+5 dans le Monte Carlo.
    long_run_var = float(nvda["rv_var_m"].dropna().tail(1260).median())
    long_run_vol_pct = float(np.sqrt(max(long_run_var, 1e-12)) * 100.0)

    return {
        "nvda_har_rv_forecasts": forecasts,
        "nvda_har_rv_table": table,
        "nvda_har_rv_5d_pct": float(forecasts[max_horizon]),
        "nvda_har_long_run_vol_pct": long_run_vol_pct,
        "nvda_har_model": model,
    }


def _business_days_to_maturity(today: pd.Timestamp, maturity: pd.Timestamp) -> int:
    start = np.datetime64(today.date(), "D")
    end = np.datetime64(maturity.date(), "D")
    return max(1, int(np.busday_count(start, end)))


def _build_mc_sigma_path(
    t_days: int,
    har_forecasts: dict[int, float],
    long_run_vol_pct: float,
    mean_reversion_days: float = 21.0,
) -> np.ndarray:
    """
    Term structure de sigma pour le MC.

    J+1..J+5 = prévisions HAR récursives. Après J+5, la vol converge
    exponentiellement vers le niveau long terme au lieu de rester figée à J+5.
    """
    sigmas = []
    sigma5 = float(har_forecasts[max(har_forecasts)])
    for day in range(1, t_days + 1):
        if day <= 5:
            pct = float(har_forecasts[day])
        else:
            decay = np.exp(-(day - 5) / float(mean_reversion_days))
            pct = float(long_run_vol_pct + (sigma5 - long_run_vol_pct) * decay)
        sigmas.append(max(pct / 100.0, 1e-6))
    return np.asarray(sigmas, dtype=float)


def _price_nvda_stabilities_mc(
    spot: float,
    today: pd.Timestamp,
    har_forecasts: dict[int, float],
    products: pd.DataFrame,
    long_run_vol_pct: float,
    nb_paths: int = NVDA_MC_PATHS,
    risk_free_rate: float = NVDA_RISK_FREE_RATE,
    dividend_yield: float = NVDA_DIVIDEND_YIELD,
    seed: int = NVDA_MC_SEED,
) -> pd.DataFrame:
    """
    Prix MC des Stability NVDA avec term structure HAR-RV.

    Chaque pas utilise son sigma propre : J+1...J+5 viennent du HAR récursif,
    puis sigma converge vers la composante long terme. Le drift est risk-neutral
    (r-q) et les barrières sont observées à chaque séance simulée.
    """
    if spot <= 0:
        raise StrategyError("Spot NVDA invalide pour le Monte Carlo.")

    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    today = pd.Timestamp(today).normalize()
    rows = []

    if products is None or products.empty:
        raise StrategyError("Aucun Stability NVIDIA à valoriser.")

    grouped = {}
    for row in products.to_dict("records"):
        maturity = pd.Timestamp(row["Maturite"]).normalize()
        grouped.setdefault(maturity, []).append(row)

    for maturity, group in sorted(grouped.items()):
        t_days = _business_days_to_maturity(today, maturity)
        sigma_path = _build_mc_sigma_path(
            t_days=t_days,
            har_forecasts=har_forecasts,
            long_run_vol_pct=long_run_vol_pct,
        )

        s = np.full(nb_paths, float(spot), dtype=float)
        path_min = s.copy()
        path_max = s.copy()

        for sigma in sigma_path:
            z = rng.standard_normal(nb_paths)
            drift = (risk_free_rate - dividend_yield - 0.5 * sigma**2) * dt
            diffusion = sigma * np.sqrt(dt)
            s *= np.exp(drift + diffusion * z)
            path_min = np.minimum(path_min, s)
            path_max = np.maximum(path_max, s)

        T = t_days / 252.0
        discount = np.exp(-risk_free_rate * T)
        avg_sigma_pct = float(np.mean(sigma_path) * 100.0)
        rms_sigma_pct = float(np.sqrt(np.mean(sigma_path**2)) * 100.0)

        for product in group:
            code = product["Code"]
            low = float(product["Barriere_basse"])
            high = float(product["Barriere_haute"])
            market_price = product.get("Prix_marche")
            isin = product.get("ISIN", "")
            price_source = product.get("Source_prix", "")

            if not (low < spot < high):
                survival = 0.0
                theo_price = 0.0
                mc_se = 0.0
            else:
                alive = (path_min > low) & (path_max < high)
                survival = float(alive.mean())
                theo_price = float(NVDA_REDEMPTION * discount * survival)
                mc_se = float(
                    NVDA_REDEMPTION * discount
                    * np.sqrt(max(survival * (1.0 - survival), 0.0) / nb_paths)
                )

            market_price_float = (
                float(market_price) if market_price is not None and pd.notna(market_price) else np.nan
            )
            edge = theo_price - market_price_float if np.isfinite(market_price_float) else np.nan

            # RR maison utilisé dans l'analyse Stability : distance en dollars
            # jusqu'à la barrière la plus proche, divisée par sqrt(nombre de jours de bourse).
            # Pour un double KO, la barrière la plus proche est celle qui pilote le risque immédiat.
            distance_low = float(spot - low)
            distance_high = float(high - spot)
            distance_nearest = float(min(distance_low, distance_high))
            rr = (
                distance_nearest / np.sqrt(float(t_days))
                if distance_nearest > 0 and t_days > 0
                else 0.0
            )

            if 4.0 <= rr <= 5.0:
                rr_zone = "Vega optimal"
            elif 6.0 <= rr <= 8.0:
                rr_zone = "Thêta optimal"
            else:
                rr_zone = ""

            rows.append({
                "Code": code,
                "ISIN": isin,
                "Spot": float(spot),
                "Barriere_basse": float(low),
                "Barriere_haute": float(high),
                "Maturite": maturity.date().isoformat(),
                "Jours_bourse": int(t_days),
                "Distance_barriere_proche": distance_nearest,
                "RR": float(rr),
                "Zone_RR": rr_zone,
                "Sigma_HAR_horizon_j": int(min(t_days, 5)),
                "Sigma_HAR_pct_ann": avg_sigma_pct,
                "Sigma_RMS_pct_ann": rms_sigma_pct,
                "Sigma_J1_pct_ann": float(har_forecasts[1]),
                "Sigma_J5_pct_ann": float(har_forecasts[5]),
                "Sigma_long_run_pct_ann": float(long_run_vol_pct),
                "Prob_survie_MC": survival,
                "Prix_theorique_MC": theo_price,
                "Erreur_std_MC": mc_se,
                "Prix_marche": market_price_float,
                "Source_prix": price_source,
                "Edge_theo_moins_marche": edge,
            })

    return pd.DataFrame(rows).sort_values(
        ["Maturite", "Barriere_basse", "Barriere_haute"]
    ).reset_index(drop=True)


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

    # NVIDIA : HAR-RV 1..5j puis valorisation Monte Carlo des Stability.
    nvda = _download_nvidia_history()
    nvda_har = _har_rv_forecast_horizons_nvda(nvda, max_horizon=5)
    nvda_latest_date = nvda["nvda"].dropna().index[-1]
    nvda_spot = float(nvda.loc[nvda_latest_date, "nvda"])
    nvda_news = compute_nvda_news_indicator(nvda_har, nvda_spot)
    nvda_products = fetch_nvda_stability_products()
    nvda_stability_mc = _price_nvda_stabilities_mc(
        spot=nvda_spot,
        today=pd.Timestamp.today().normalize(),
        har_forecasts=nvda_har["nvda_har_rv_forecasts"],
        products=nvda_products,
        long_run_vol_pct=nvda_har["nvda_har_long_run_vol_pct"],
    )

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
    print(f"NVDA spot: {nvda_spot:.2f}")
    print(
        "Stability SG récupérés: "
        f"{nvda_products.attrs.get('scraped_total', len(nvda_products))}"
        + (
            f"/{nvda_products.attrs.get('expected_total')}"
            if nvda_products.attrs.get('expected_total') is not None else ""
        )
    )
    print("HAR-RV NVDA annualisée 1-5j:")
    for h, sigma in nvda_har["nvda_har_rv_forecasts"].items():
        print(f"  J+{h}: {sigma:.2f}%")
    print("\nPrix Monte Carlo Stability NVDA:")
    print(nvda_stability_mc[[
        "Code", "Maturite", "Barriere_basse", "Barriere_haute",
        "Sigma_HAR_pct_ann", "Prob_survie_MC", "Prix_theorique_MC",
        "Prix_marche", "Edge_theo_moins_marche"
    ]].to_string(index=False))

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
        # NVIDIA HAR-RV + Monte Carlo Stability
        "nvda_date": nvda_latest_date,
        "nvda_spot": nvda_spot,
        "nvda_har_rv_forecasts": nvda_har["nvda_har_rv_forecasts"],
        "nvda_har_rv_table": nvda_har["nvda_har_rv_table"],
        "nvda_har_rv_5d_pct": nvda_har["nvda_har_rv_5d_pct"],
        **nvda_news,
        "nvda_stability_mc": nvda_stability_mc,
        "nvda_stability_count": int(len(nvda_products)),
        "nvda_stability_expected_count": nvda_products.attrs.get("expected_total"),
        "nvda_stability_complete": bool(nvda_products.attrs.get("complete", True)),
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
