from __future__ import annotations

import os

from flask import Flask, render_template, request

from config import AppConfig, DEFAULT_SYMBOL
from predict import run_analysis


app = Flask(__name__)


def parse_decimal(value: str | None, default: float) -> float:
    raw = (value or "").strip()
    if not raw:
        return default
    normalized = raw.replace(",", ".")
    return float(normalized)


def build_config_from_form(form) -> AppConfig:
    return AppConfig(
        symbol=(form.get("symbol") or DEFAULT_SYMBOL).upper().strip() or DEFAULT_SYMBOL,
        api_key=(form.get("api_key") or "").strip(),
        api_secret=(form.get("api_secret") or "").strip(),
        initial_capital=parse_decimal(form.get("initial_capital"), 10000.0),
        risk_per_trade_pct=parse_decimal(form.get("risk_per_trade_pct"), 10.0),
        news_api_key=(form.get("news_api_key") or "").strip(),
        news_provider=(form.get("news_provider") or "none").strip(),
    )


def default_form_data() -> dict:
    return {
        "symbol": DEFAULT_SYMBOL,
        "api_key": "",
        "api_secret": "",
        "initial_capital": 10000.0,
        "risk_per_trade_pct": 10.0,
        "news_provider": "none",
        "news_api_key": os.getenv("CRYPTOPANIC_API_KEY", ""),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    form_data = default_form_data()
    context = {
        "results": None,
        "error_message": None,
        "news_notice": "CryptoPanic anuncio que el plan gratis DEVELOPER se elimina el 1 de abril de 2026. Si deja de responder, necesitarias un plan pago o cambiar de proveedor.",
        "form_data": form_data,
    }

    if request.method == "POST":
        form_data = {
            "symbol": request.form.get("symbol", DEFAULT_SYMBOL),
            "api_key": request.form.get("api_key", ""),
            "api_secret": request.form.get("api_secret", ""),
            "initial_capital": request.form.get("initial_capital", "10000"),
            "risk_per_trade_pct": request.form.get("risk_per_trade_pct", "10"),
            "news_provider": request.form.get("news_provider", "none"),
            "news_api_key": request.form.get("news_api_key", ""),
        }
        context["form_data"] = form_data
        try:
            config = build_config_from_form(request.form)
            results = run_analysis(config)

            summary = results["backtest_summary"]
            forecast = results["forecast_summary"]
            news_df = results["news_df"]
            trades_df = results["trades_df"]
            equity_df = results["equity_df"]
            price_history = results["price_history"]
            last = price_history.iloc[-1]

            context["results"] = {
                "latest_price": f"{results['latest_price']:.2f}",
                "summary": summary,
                "forecast": forecast,
                "technical": {
                    "Fecha": last["date"].strftime("%Y-%m-%d"),
                    "SMA20": round(float(last["SMA20"]), 2),
                    "SMA50": round(float(last["SMA50"]), 2),
                    "RSI14": round(float(last["RSI14"]), 2),
                    "MACD": round(float(last["MACD"]), 4),
                    "MACD Signal": round(float(last["MACD_SIGNAL"]), 4),
                    "ATR14": round(float(last["ATR14"]), 2),
                    "Soporte 90d": round(float(last["support_90d"]), 2),
                    "Resistencia 90d": round(float(last["resistance_90d"]), 2),
                },
                "chart_html": results["chart"].to_html(full_html=False, include_plotlyjs="cdn"),
                "trades_table": trades_df.sort_values("Fecha", ascending=False).to_html(classes="data-table", index=False) if not trades_df.empty else None,
                "equity_table": equity_df.tail(30).to_html(classes="data-table", index=False),
                "news_table": news_df[[column for column in ["date", "title_es", "title", "source", "sentiment", "impact", "url"] if column in news_df.columns]].to_html(classes="data-table", index=False) if not news_df.empty else None,
                "news_error": results["news_error"],
            }
        except Exception as exc:
            context["error_message"] = str(exc)

    return render_template("index.html", **context)


if __name__ == "__main__":
    app.run(debug=True)
