"""Archive la RV intraday NVDA issue des bougies Yahoo 15 minutes."""

from strategy import refresh_nvda_rv_cache


if __name__ == "__main__":
    history = refresh_nvda_rv_cache()
    print(f"{len(history)} séances RV 15 min, dernière séance : {history.index[-1].date()}")
