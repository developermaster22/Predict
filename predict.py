"""
Punto de entrada del bot BTC/EUR.
"""

from __future__ import annotations

import pandas as pd

from backtest import run_backtest
from charting import create_chart
from config import (
    AppConfig,
    BACKTEST_LOOKBACK_DAYS,
    CHART_HTML,
    EQUITY_CSV,
    NEWS_CSV,
    OPERATIONS_CSV,
    PRICE_LOOKBACK_YEARS,
    SUPPORT_RESISTANCE_WINDOW,
    load_config,
)
from data_loader import build_binance_client, fetch_latest_price, fetch_price_history
from indicators import calculate_fibonacci_levels, enrich_indicators, extract_forecast_turning_points, monte_carlo_forecast
from news import load_news
from signals import prepare_backtest_dataset


def print_summary(price_data: pd.DataFrame, forecast_summary: dict, backtest_summary: dict, news_df: pd.DataFrame) -> None:
    last = price_data.iloc[-1]
    print("")
    print("=" * 80)
    print("RESUMEN")
    print("=" * 80)
    print(f"Fecha ultimo dato:           {last['date']:%Y-%m-%d}")
    print(f"Precio actual BTC/EUR:       {last['Close']:.2f} EUR")
    print(f"SMA20 / SMA50:               {last['SMA20']:.2f} / {last['SMA50']:.2f}")
    print(f"RSI(14):                     {last['RSI14']:.2f}")
    print(f"MACD / Signal / Hist:        {last['MACD']:.4f} / {last['MACD_SIGNAL']:.4f} / {last['MACD_HIST']:.4f}")
    print(f"ATR(14):                     {last['ATR14']:.2f}")
    print(f"Soporte 90d / Resistencia:   {last['support_90d']:.2f} / {last['resistance_90d']:.2f}")
    print("")
    print("Monte Carlo 30 dias")
    print(f"Drift anualizado:            {forecast_summary['annual_drift']:.4f}")
    print(f"Volatilidad anualizada:      {forecast_summary['annual_volatility']:.4f}")
    print(f"Min esperado IC 95%:         {forecast_summary['expected_min_95']:.2f} EUR")
    print(f"Max esperado IC 95%:         {forecast_summary['expected_max_95']:.2f} EUR")
    print(f"Precio medio final esperado: {forecast_summary['expected_final_mean']:.2f} EUR")
    print(f"Prob. cierre sobre actual:   {forecast_summary['prob_final_above_last'] * 100:.2f}%")
    print("")
    print("Backtesting 1 ano")
    print(f"Capital final:               {backtest_summary['capital_final']:.2f} EUR")
    print(f"Ganancia total:              {backtest_summary['ganancia_total_eur']:+.2f} EUR")
    print(f"Rentabilidad total:          {backtest_summary['ganancia_total_pct']:+.2f}%")
    print(f"Operaciones registradas:     {int(backtest_summary['operaciones_totales'])}")
    print(f"Ventas cerradas:             {int(backtest_summary['ventas_cerradas'])}")
    print(f"Win rate:                    {backtest_summary['win_rate']:.2f}%")
    print(f"Profit factor:               {backtest_summary['profit_factor']:.2f}")
    print(f"Expectancy:                  {backtest_summary['expectancy_eur']:+.2f} EUR")
    print(f"Sharpe aprox.:               {backtest_summary['sharpe']:.2f}")
    print(f"Max drawdown:                {backtest_summary['max_drawdown_pct']:.2f}%")
    print(f"Fees pagadas:                {backtest_summary['fees_paid_eur']:.2f} EUR")
    print("")
    print(f"Noticias relevantes cargadas: {len(news_df)}")
    if news_df.empty:
        print("Noticias: no se cargaron. Configura CryptoPanic o NewsAPI para activarlas.")


def run_analysis(config: AppConfig) -> dict:
    client = build_binance_client(config.api_key, config.api_secret)
    price_history = fetch_price_history(client, config.symbol, years=PRICE_LOOKBACK_YEARS)
    latest_price = fetch_latest_price(client, config.symbol)
    price_history.loc[price_history.index[-1], "Close"] = latest_price
    price_history.loc[price_history.index[-1], "High"] = max(price_history["High"].iloc[-1], latest_price)
    price_history.loc[price_history.index[-1], "Low"] = min(price_history["Low"].iloc[-1], latest_price)
    price_history = enrich_indicators(price_history)

    news_df, news_error = load_news(config.news_provider, config.news_api_key)
    if not news_df.empty:
        news_df.to_csv(NEWS_CSV, index=False)

    forecast_df, _, forecast_summary = monte_carlo_forecast(
        price_history,
        forecast_days=config.forecast_days,
        paths=config.monte_carlo_paths,
    )
    forecast_points = extract_forecast_turning_points(forecast_df)

    backtest_data = prepare_backtest_dataset(price_history, news_df)
    trades_df, equity_df, backtest_summary = run_backtest(backtest_data, config)

    if not trades_df.empty:
        trades_df.to_csv(OPERATIONS_CSV, index=False)
    equity_df.to_csv(EQUITY_CSV, index=False)

    fibonacci_levels = calculate_fibonacci_levels(price_history, lookback=SUPPORT_RESISTANCE_WINDOW)
    chart = create_chart(
        data=price_history.tail(BACKTEST_LOOKBACK_DAYS).copy(),
        forecast=forecast_df,
        forecast_points=forecast_points,
        trades=trades_df,
        equity_df=equity_df,
        fibonacci_levels=fibonacci_levels,
        news_df=news_df,
        latest_price=latest_price,
        summary=backtest_summary,
    )
    chart.write_html(CHART_HTML)
    return {
        "config": config,
        "price_history": price_history,
        "latest_price": latest_price,
        "news_df": news_df,
        "news_error": news_error,
        "forecast_df": forecast_df,
        "forecast_summary": forecast_summary,
        "forecast_points": forecast_points,
        "backtest_data": backtest_data,
        "trades_df": trades_df,
        "equity_df": equity_df,
        "backtest_summary": backtest_summary,
        "chart": chart,
    }


def main() -> None:
    config = load_config()

    print("")
    print("Descargando historicos de Binance...")
    print("Consultando noticias...")
    print("Generando simulacion Monte Carlo...")
    print("Preparando backtesting de 1 ano...")
    results = run_analysis(config)

    chart = results["chart"]
    try:
        chart.show()
    except Exception:
        pass

    trades_df = results["trades_df"]
    news_df = results["news_df"]

    if not trades_df.empty:
        print("")
        print("Ultimas operaciones simuladas:")
        printable = trades_df.copy()
        printable["Fecha"] = pd.to_datetime(printable["Fecha"]).dt.strftime("%Y-%m-%d")
        print(printable.tail(12).to_string(index=False))
    else:
        print("")
        print("No se generaron operaciones con las reglas actuales del backtest.")

    print_summary(results["price_history"], results["forecast_summary"], results["backtest_summary"], news_df)
    print("")
    print(f"HTML exportado: {CHART_HTML.resolve()}")
    print(f"CSV operaciones: {OPERATIONS_CSV.resolve()}")
    print(f"CSV equity: {EQUITY_CSV.resolve()}")
    if not news_df.empty:
        print(f"CSV noticias: {NEWS_CSV.resolve()}")


if __name__ == "__main__":
    main()
