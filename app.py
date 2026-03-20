from __future__ import annotations

import streamlit as st

from config import AppConfig, CHART_HTML, DEFAULT_SYMBOL, EQUITY_CSV, NEWS_CSV, OPERATIONS_CSV
from predict import run_analysis


st.set_page_config(page_title="BTC/EUR Quant Bot", layout="wide")


def main() -> None:
    st.title("BTC/EUR Quant Bot")
    st.caption("Carga parametros desde esta interfaz y ejecuta el analisis sin usar la terminal.")
    st.warning("CryptoPanic anuncio que el plan gratis DEVELOPER se elimina el 1 de abril de 2026. Si deja de responder, necesitarias un plan pago o cambiar de proveedor.")

    with st.sidebar:
        st.header("Parametros")
        with st.form("bot_form"):
            symbol = st.text_input("Par a analizar", value=DEFAULT_SYMBOL)
            api_key = st.text_input("Binance API Key", value="", type="password")
            api_secret = st.text_input("Binance API Secret", value="", type="password")
            initial_capital = st.number_input("Capital inicial (EUR)", min_value=1.0, value=10000.0, step=100.0)
            risk_per_trade_pct = st.number_input("Porcentaje de capital por operacion", min_value=0.1, max_value=100.0, value=10.0, step=0.5)
            news_provider = st.selectbox("Proveedor de noticias", options=["none", "cryptopanic", "newsapi"], index=0)
            news_api_key = st.text_input("API Key noticias", value="", type="password")
            submitted = st.form_submit_button("Ejecutar analisis", use_container_width=True)

    if not submitted:
        st.info("Completa el formulario lateral y pulsa 'Ejecutar analisis'.")
        return

    config = AppConfig(
        symbol=symbol.upper().strip() or DEFAULT_SYMBOL,
        api_key=api_key.strip(),
        api_secret=api_secret.strip(),
        initial_capital=float(initial_capital),
        risk_per_trade_pct=float(risk_per_trade_pct),
        news_api_key=news_api_key.strip(),
        news_provider=news_provider,
    )

    with st.spinner("Descargando datos, calculando indicadores y generando resultados..."):
        results = run_analysis(config)

    summary = results["backtest_summary"]
    forecast = results["forecast_summary"]
    news_df = results["news_df"]
    news_error = results["news_error"]
    trades_df = results["trades_df"]
    equity_df = results["equity_df"]
    price_history = results["price_history"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precio actual", f"{results['latest_price']:.2f} EUR")
    col2.metric("Capital final", f"{summary['capital_final']:.2f} EUR", f"{summary['ganancia_total_pct']:.2f}%")
    col3.metric("Operaciones", int(summary["operaciones_totales"]))
    col4.metric("Win rate", f"{summary['win_rate']:.2f}%")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Profit factor", f"{summary['profit_factor']:.2f}")
    col2.metric("Max drawdown", f"{summary['max_drawdown_pct']:.2f}%")
    col3.metric("Monte Carlo media final", f"{forecast['expected_final_mean']:.2f} EUR")
    col4.metric("Prob. cierre > actual", f"{forecast['prob_final_above_last'] * 100:.2f}%")

    st.subheader("Grafico")
    st.plotly_chart(results["chart"], use_container_width=True)

    st.subheader("Resumen tecnico")
    last = price_history.iloc[-1]
    st.write(
        {
            "Fecha": last["date"].strftime("%Y-%m-%d"),
            "SMA20": round(float(last["SMA20"]), 2),
            "SMA50": round(float(last["SMA50"]), 2),
            "RSI14": round(float(last["RSI14"]), 2),
            "MACD": round(float(last["MACD"]), 4),
            "MACD Signal": round(float(last["MACD_SIGNAL"]), 4),
            "ATR14": round(float(last["ATR14"]), 2),
            "Soporte 90d": round(float(last["support_90d"]), 2),
            "Resistencia 90d": round(float(last["resistance_90d"]), 2),
        }
    )

    st.subheader("Operaciones simuladas")
    if trades_df.empty:
        st.warning("No se generaron operaciones con los parametros actuales.")
    else:
        st.dataframe(trades_df.sort_values("Fecha", ascending=False), use_container_width=True)

    st.subheader("Equity curve")
    st.dataframe(equity_df.tail(30), use_container_width=True)

    st.subheader("Noticias")
    if news_df.empty:
        if news_error:
            st.error(news_error)
        else:
            st.info("No se cargaron noticias. Usa un proveedor y una API Key para activarlas.")
    else:
        if news_error:
            st.warning(news_error)
        columns = [column for column in ["date", "title_es", "title", "source", "sentiment", "impact", "url"] if column in news_df.columns]
        st.dataframe(news_df[columns], use_container_width=True)

    st.subheader("Archivos exportados")
    st.write(str(CHART_HTML.resolve()))
    st.write(str(OPERATIONS_CSV.resolve()))
    st.write(str(EQUITY_CSV.resolve()))
    if not news_df.empty:
        st.write(str(NEWS_CSV.resolve()))


if __name__ == "__main__":
    main()
