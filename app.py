from __future__ import annotations

from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
from shiny import App, reactive, render, ui

from strategy import compute_daily_signal


APP_TITLE = "Signal Stability SPX"


app_ui = ui.page_fluid(
    ui.tags.style(
        """
        body {background:#f5f7fb;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}
        .app-shell {max-width:980px;margin:0 auto;padding:28px 18px 42px;}
        .hero,.panel,.metric {background:white;border:1px solid #e7eaf0;box-shadow:0 8px 28px rgba(25,35,55,.06);}
        .hero {border-radius:18px;padding:24px;margin-bottom:18px;}
        .signal-card,.regime-card {border-radius:18px;padding:28px;margin-bottom:18px;text-align:center;box-shadow:0 8px 28px rgba(25,35,55,.08);}
        .signal-buy {background:#eaf8ef;border:1px solid #b9e6c7;}
        .signal-wait {background:#f1f3f6;border:1px solid #d9dee7;}
        .signal-error {background:#fff0f0;border:1px solid #f0bcbc;}
        .regime-bull {background:#eaf8ef;border:1px solid #b9e6c7;}
        .regime-bear {background:#fff0f0;border:1px solid #f0bcbc;}
        .regime-unknown {background:#f1f3f6;border:1px solid #d9dee7;}
        .signal-title,.regime-title {font-size:34px;font-weight:800;margin-bottom:8px;}
        .signal-subtitle,.regime-subtitle {font-size:16px;color:#4d5665;}
        .metric-grid {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin-bottom:18px;}
        .metric {border-radius:14px;padding:18px;}
        .metric-label {color:#6b7280;font-size:13px;margin-bottom:6px;}
        .metric-value {font-size:23px;font-weight:750;}
        .panel {border-radius:16px;padding:20px;margin-bottom:18px;}
        .small-muted {color:#6b7280;font-size:13px;}
        .refresh-row {display:flex;align-items:center;gap:12px;flex-wrap:wrap;}
        @media (max-width:760px){
            .metric-grid{grid-template-columns:repeat(2,minmax(0,1fr));}
            .signal-title,.regime-title{font-size:28px;}
        }
        """
    ),
    ui.div(
        {"class": "app-shell"},
        ui.div(
            {"class": "hero"},
            ui.h1(APP_TITLE),
            ui.p(
                "Le modèle Ensemble calcule le signal du Stability, le Jump Model indique le régime Bull/Bear, et les modèles HAR prévoient la RV, le VIX et la VRP à 5 jours."
            ),
            ui.div(
                {"class": "refresh-row"},
                ui.input_action_button(
                    "refresh",
                    "Actualiser le signal",
                    class_="btn btn-primary",
                ),
                ui.span(
                    "Les données sont téléchargées à chaque actualisation.",
                    class_="small-muted",
                ),
            ),
        ),
        ui.output_ui("regime_panel"),
        ui.output_ui("signal_panel"),
        ui.div(
            {"class": "metric-grid"},
            ui.div(
                {"class": "metric"},
                ui.div("SPX", class_="metric-label"),
                ui.div(ui.output_text("spx"), class_="metric-value"),
            ),
            ui.div(
                {"class": "metric"},
                ui.div("VIX actuel", class_="metric-label"),
                ui.div(ui.output_text("regime_vix"), class_="metric-value"),
            ),
            ui.div(
                {"class": "metric"},
                ui.div("Score Ensemble", class_="metric-label"),
                ui.div(ui.output_text("score"), class_="metric-value"),
            ),
            ui.div(
                {"class": "metric"},
                ui.div("Percentile", class_="metric-label"),
                ui.div(ui.output_text("percentile"), class_="metric-value"),
            ),
        ),
        ui.div(
            {"class": "metric-grid"},
            ui.div(
                {"class": "metric"},
                ui.div("VIX moyen Bull", class_="metric-label"),
                ui.div(ui.output_text("bull_vix_mean"), class_="metric-value"),
            ),
            ui.div(
                {"class": "metric"},
                ui.div("VIX moyen Bear", class_="metric-label"),
                ui.div(ui.output_text("bear_vix_mean"), class_="metric-value"),
            ),
            ui.div(
                {"class": "metric"},
                ui.div("Régime précédent", class_="metric-label"),
                ui.div(ui.output_text("previous_regime"), class_="metric-value"),
            ),
            ui.div(
                {"class": "metric"},
                ui.div("Changement de régime", class_="metric-label"),
                ui.div(ui.output_text("regime_changed"), class_="metric-value"),
            ),
        ),
        ui.div(
            {"class": "metric-grid"},
            ui.div(
                {"class": "metric"},
                ui.div("IV actuelle", class_="metric-label"),
                ui.div(ui.output_text("iv_current"), class_="metric-value"),
            ),
            ui.div(
                {"class": "metric"},
                ui.div("RV prévue 5j", class_="metric-label"),
                ui.div(ui.output_text("rv_forecast"), class_="metric-value"),
            ),
            ui.div(
                {"class": "metric"},
                ui.div("VRP avec IV actuelle", class_="metric-label"),
                ui.div(ui.output_text("vrp_current"), class_="metric-value"),
            ),
            ui.div(
                {"class": "metric"},
                ui.div("VRP future prévue 5j", class_="metric-label"),
                ui.div(ui.output_text("vrp_future"), class_="metric-value"),
            ),
        ),
        ui.div(
            {"class": "metric-grid"},
            ui.div(
                {"class": "metric"},
                ui.div("VIX prévu 5j", class_="metric-label"),
                ui.div(ui.output_text("vix_forecast"), class_="metric-value"),
            ),
            ui.div(
                {"class": "metric"},
                ui.div("Compression GARCH VIX", class_="metric-label"),
                ui.div(ui.output_text("garch_compression"), class_="metric-value"),
            ),
            ui.div(
                {"class": "metric"},
                ui.div("Sigma GARCH", class_="metric-label"),
                ui.div(ui.output_text("garch_sigma"), class_="metric-value"),
            ),
            ui.div(
                {"class": "metric"},
                ui.div("Seuil compression", class_="metric-label"),
                ui.div(ui.output_text("garch_threshold"), class_="metric-value"),
            ),
        ),
        ui.div(
            {"class": "panel"},
            ui.h3("Prévisions VRP à 5 jours"),
            ui.output_plot("vrp_plot", height="430px"),
            ui.p(
                "VRP avec IV actuelle = IV observée aujourd'hui − RV prévue. "
                "VRP future prévue = IV future HAR-VIX − RV future HAR-RV.",
                class_="small-muted",
            ),
        ),
        ui.div(
            {"class": "panel"},
            ui.h3("Détail HAR / VRP / GARCH"),
            ui.output_table("vrp_table"),
        ),
        ui.div(
            {"class": "panel"},
            ui.h3("Produit analysé"),
            ui.output_table("product_table"),
        ),
        ui.div(
            {"class": "panel"},
            ui.h3("Détail du calcul"),
            ui.output_table("model_table"),
        ),
        ui.div(
            {"class": "panel"},
            ui.h3("Détail du Jump Model"),
            ui.output_table("regime_table"),
        ),
        ui.p(
            "Ce tableau de bord fournit un signal quantitatif, pas un ordre d'achat automatique.",
            class_="small-muted",
        ),
    ),
)


def server(input, output, session):
    refresh_counter = reactive.value(0)

    @reactive.effect
    @reactive.event(input.refresh)
    def _refresh():
        refresh_counter.set(refresh_counter.get() + 1)

    @reactive.calc
    def result():
        refresh_counter.get()
        try:
            return compute_daily_signal()
        except Exception as exc:
            return {
                "ok": False,
                "error": str(exc),
                "generated_at": datetime.now(),
            }

    @render.ui
    def regime_panel():
        r = result()

        if not r.get("ok"):
            return ui.div(
                {"class": "regime-card signal-error"},
                ui.div("Régime indisponible", class_="regime-title"),
                ui.div(
                    r.get("error", "Erreur inconnue"),
                    class_="regime-subtitle",
                ),
            )

        regime = str(r.get("regime", "Indéterminé"))
        regime_date = r.get("regime_date", r.get("date"))
        date_text = (
            regime_date.date().isoformat()
            if hasattr(regime_date, "date")
            else str(regime_date)
        )

        if regime == "Bull":
            css_class = "regime-card regime-bull"
            title = "🟢 RÉGIME BULL"
            description = "Marché haussier et risque relativement faible selon le Jump Model."
        elif regime == "Bear":
            css_class = "regime-card regime-bear"
            title = "🔴 RÉGIME BEAR"
            description = "Marché baissier ou turbulent et risque élevé selon le Jump Model."
        else:
            css_class = "regime-card regime-unknown"
            title = "⚪ RÉGIME INDÉTERMINÉ"
            description = "Le Jump Model n'a pas produit de régime exploitable."

        changed = bool(r.get("jump_regime_changed", False))
        change_text = " Changement de régime détecté." if changed else ""

        return ui.div(
            {"class": css_class},
            ui.div(title, class_="regime-title"),
            ui.div(
                f"Signal daté du {date_text}. {description}{change_text}",
                class_="regime-subtitle",
            ),
        )

    @render.ui
    def signal_panel():
        r = result()

        if not r.get("ok"):
            return ui.div(
                {"class": "signal-card signal-error"},
                ui.div("Erreur de calcul", class_="signal-title"),
                ui.div(
                    r.get("error", "Erreur inconnue"),
                    class_="signal-subtitle",
                ),
            )

        if r["take_position"]:
            return ui.div(
                {"class": "signal-card signal-buy"},
                ui.div("🟢 PRENDRE POSITION", class_="signal-title"),
                ui.div(
                    f"Signal observé sur les données du {r['date'].date()}. "
                    "Vérifie le prix, le spread et les caractéristiques du produit avant l'ordre.",
                    class_="signal-subtitle",
                ),
            )

        return ui.div(
            {"class": "signal-card signal-wait"},
            ui.div("⚪ NE PAS PRENDRE POSITION", class_="signal-title"),
            ui.div(
                f"Le score du {r['date'].date()} reste sous le seuil d'entrée.",
                class_="signal-subtitle",
            ),
        )

    @render.text
    def spx():
        r = result()
        return "—" if not r.get("ok") else f"{r['spx']:,.2f}"

    @render.text
    def regime_vix():
        r = result()
        value = r.get("regime_vix") if r.get("ok") else None
        return "—" if value is None or pd.isna(value) else f"{value:.2f}"

    @render.text
    def score():
        r = result()
        return "—" if not r.get("ok") else f"{r['ensemble_score']:.2f}"

    @render.text
    def percentile():
        r = result()
        return "—" if not r.get("ok") else f"{100 * r['percentile']:.1f} %"

    @render.text
    def bull_vix_mean():
        r = result()
        value = r.get("bull_vix_mean") if r.get("ok") else None
        return "—" if value is None or pd.isna(value) else f"{value:.2f}"

    @render.text
    def bear_vix_mean():
        r = result()
        value = r.get("bear_vix_mean") if r.get("ok") else None
        return "—" if value is None or pd.isna(value) else f"{value:.2f}"

    @render.text
    def previous_regime():
        r = result()
        if not r.get("ok"):
            return "—"
        previous = r.get("jump_previous_regime")
        return "—" if previous in (None, "None", "nan") else str(previous)

    @render.text
    def regime_changed():
        r = result()
        if not r.get("ok"):
            return "—"
        return "Oui" if r.get("jump_regime_changed", False) else "Non"

    @render.text
    def iv_current():
        r = result()
        value = r.get("iv_current_vol_pct") if r.get("ok") else None
        return "—" if value is None or pd.isna(value) else f"{value:.2f} %"

    @render.text
    def rv_forecast():
        r = result()
        value = r.get("rv_forecast_vol_pct") if r.get("ok") else None
        return "—" if value is None or pd.isna(value) else f"{value:.2f} %"

    @render.text
    def vrp_current():
        r = result()
        value = r.get("vrp_current_iv_forecast_points") if r.get("ok") else None
        return "—" if value is None or pd.isna(value) else f"{value:+.1f} pts"

    @render.text
    def vrp_future():
        r = result()
        value = r.get("vrp_future_forecast_points") if r.get("ok") else None
        return "—" if value is None or pd.isna(value) else f"{value:+.1f} pts"

    @render.text
    def vix_forecast():
        r = result()
        value = r.get("vix_forecast_5d") if r.get("ok") else None
        return "—" if value is None or pd.isna(value) else f"{value:.2f}"

    @render.text
    def garch_compression():
        r = result()
        if not r.get("ok"):
            return "—"
        return "Oui" if r.get("garch_compression", False) else "Non"

    @render.text
    def garch_sigma():
        r = result()
        value = r.get("garch_sigma") if r.get("ok") else None
        return "—" if value is None or pd.isna(value) else f"{value:.5f}"

    @render.text
    def garch_threshold():
        r = result()
        value = r.get("garch_compression_threshold") if r.get("ok") else None
        return "—" if value is None or pd.isna(value) else f"{value:.5f}"

    @render.plot
    def vrp_plot():
        r = result()
        fig, ax = plt.subplots(figsize=(11, 4.5))
        if not r.get("ok"):
            ax.text(0.5, 0.5, r.get("error", "Erreur"), ha="center", va="center")
            ax.axis("off")
            return fig

        history = r.get("vrp_history")
        if history is None or len(history) == 0:
            ax.text(0.5, 0.5, "Historique VRP indisponible", ha="center", va="center")
            ax.axis("off")
            return fig

        h = history.copy()
        h["Date"] = pd.to_datetime(h["Date"])
        ax.plot(h["Date"], h["VRP_tradable_pred_points"], label="VRP prédite avec IV actuelle", linewidth=1.5)
        ax.plot(h["Date"], h["VRP_future_pred_points"], label="VRP future prédite 5j", linewidth=1.5)
        ax.plot(h["Date"], h["VRP_tradable_real_points"], label="VRP réalisée avec IV d'entrée", linewidth=1.0, alpha=0.7)
        ax.axhline(0, linewidth=1)
        ax.set_title("Prévisions walk-forward de VRP — horizon 5 jours")
        ax.set_xlabel("Date")
        ax.set_ylabel("Points de variance annualisés")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        fig.autofmt_xdate()
        fig.tight_layout()
        return fig

    @render.table
    def vrp_table():
        r = result()
        if not r.get("ok"):
            return pd.DataFrame({"Mesure": ["Erreur"], "Valeur": [r["error"]]})
        date = r.get("har_date")
        date_text = date.date().isoformat() if hasattr(date, "date") else str(date)
        return pd.DataFrame({
            "Mesure": [
                "Date HAR",
                "IV actuelle (VIX)",
                "RV prévue sur 5 jours",
                "VIX moyen prévu sur 5 jours",
                "VRP avec IV actuelle",
                "VRP future prévue sur 5 jours",
                "Compression GARCH VIX",
                "Sigma GARCH",
                "Seuil de compression",
            ],
            "Valeur": [
                date_text,
                f"{r['iv_current_vol_pct']:.2f} %",
                f"{r['rv_forecast_vol_pct']:.2f} %",
                f"{r['vix_forecast_5d']:.2f}",
                f"{r['vrp_current_iv_forecast_points']:+.2f} points de variance",
                f"{r['vrp_future_forecast_points']:+.2f} points de variance",
                "Oui" if r.get("garch_compression", False) else "Non",
                "—" if pd.isna(r.get("garch_sigma")) else f"{r['garch_sigma']:.6f}",
                "—" if pd.isna(r.get("garch_compression_threshold")) else f"{r['garch_compression_threshold']:.6f}",
            ],
        })

    @render.table
    def product_table():
        r = result()

        if not r.get("ok"):
            return pd.DataFrame(
                {"Information": ["Erreur"], "Valeur": [r["error"]]}
            )

        return pd.DataFrame(
            {
                "Information": [
                    "Nom",
                    "ISIN",
                    "Barrière basse",
                    "Barrière haute",
                    "Distance barrière basse",
                    "Distance barrière haute",
                    "Maturité",
                ],
                "Valeur": [
                    r["product_name"],
                    r["product_isin"],
                    f"{r['lower_barrier']:.0f}",
                    f"{r['upper_barrier']:.0f}",
                    f"{100 * r['lower_distance']:.2f} %",
                    f"+{100 * r['upper_distance']:.2f} %",
                    r["maturity"].date().isoformat(),
                ],
            }
        )

    @render.table
    def model_table():
        r = result()

        if not r.get("ok"):
            return pd.DataFrame(
                {"Mesure": ["Erreur"], "Valeur": [r["error"]]}
            )

        return pd.DataFrame(
            {
                "Mesure": [
                    "Date des données",
                    "Dernière cible connue",
                    "Score Ridge",
                    "Score Boosting",
                    "Score Ensemble",
                    "Seuil quantile",
                    "Décision",
                ],
                "Valeur": [
                    r["date"].date().isoformat(),
                    r["last_target_date"].date().isoformat(),
                    f"{r['ridge_score']:.3f}",
                    f"{r['boosting_score']:.3f}",
                    f"{r['ensemble_score']:.3f}",
                    f"{r['threshold']:.3f}",
                    "Prendre position" if r["take_position"] else "Attendre",
                ],
            }
        )

    @render.table
    def regime_table():
        r = result()

        if not r.get("ok"):
            return pd.DataFrame(
                {"Mesure": ["Erreur"], "Valeur": [r["error"]]}
            )

        regime_date = r.get("regime_date")
        regime_date_text = (
            regime_date.date().isoformat()
            if hasattr(regime_date, "date")
            else str(regime_date)
        )

        bull_mean = r.get("bull_vix_mean")
        bear_mean = r.get("bear_vix_mean")
        current_vix = r.get("regime_vix")

        return pd.DataFrame(
            {
                "Mesure": [
                    "Date du régime",
                    "Régime actuel",
                    "Régime précédent",
                    "Changement détecté",
                    "VIX actuel",
                    "VIX moyen en Bull",
                    "VIX moyen en Bear",
                ],
                "Valeur": [
                    regime_date_text,
                    r.get("regime", "—"),
                    r.get("jump_previous_regime") or "—",
                    "Oui" if r.get("jump_regime_changed", False) else "Non",
                    "—" if current_vix is None or pd.isna(current_vix) else f"{current_vix:.2f}",
                    "—" if bull_mean is None or pd.isna(bull_mean) else f"{bull_mean:.2f}",
                    "—" if bear_mean is None or pd.isna(bear_mean) else f"{bear_mean:.2f}",
                ],
            }
        )


app = App(app_ui, server)
