from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


pio.renderers.default = "browser"


def build_news_markers(news_df: pd.DataFrame, price_data: pd.DataFrame) -> pd.DataFrame:
    if news_df.empty:
        return news_df
    markers = news_df.copy()
    markers["date"] = pd.to_datetime(markers["date"], errors="coerce")
    if getattr(markers["date"].dt, "tz", None) is not None:
        markers["date"] = markers["date"].dt.tz_localize(None)
    markers = markers.dropna(subset=["date"])
    merged = pd.merge_asof(
        markers.sort_values("date"),
        price_data[["date", "High"]].sort_values("date"),
        on="date",
        direction="backward",
    )
    merged["plot_price"] = merged["High"].fillna(price_data["High"].iloc[-1]) * 1.02
    return merged


def create_chart(
    data: pd.DataFrame,
    forecast: pd.DataFrame,
    forecast_points: pd.DataFrame,
    trades: pd.DataFrame,
    equity_df: pd.DataFrame,
    fibonacci_levels: Dict[str, float],
    news_df: pd.DataFrame,
    latest_price: float,
    summary: Dict[str, float],
) -> go.Figure:
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.52, 0.14, 0.16, 0.18],
        subplot_titles=("Precio, prediccion y eventos", "RSI (14)", "MACD", "Equity Curve"),
    )

    fig.add_trace(go.Candlestick(x=data["date"], open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"], name="BTC/EUR"), row=1, col=1)
    fig.add_trace(go.Scatter(x=data["date"], y=data["SMA20"], name="SMA20", line=dict(color="#ff9f1c", width=1.6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data["date"], y=data["SMA50"], name="SMA50", line=dict(color="#2ec4b6", width=1.6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data["date"], y=data["support_90d"], name="Soporte 90d", line=dict(color="#198754", dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=data["date"], y=data["resistance_90d"], name="Resistencia 90d", line=dict(color="#d00000", dash="dash")), row=1, col=1)

    fig.add_trace(go.Scatter(x=forecast["date"], y=forecast["mean"], mode="lines", name="Monte Carlo media", line=dict(color="#3a86ff", width=2.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=forecast["date"], y=forecast["p25"], mode="lines", name="P25", line=dict(color="rgba(58,134,255,0.35)", dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=forecast["date"], y=forecast["p75"], mode="lines", name="P75", line=dict(color="rgba(58,134,255,0.35)", dash="dot")), row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=pd.concat([forecast["date"], forecast["date"][::-1]]),
            y=pd.concat([forecast["p95"], forecast["p05"][::-1]]),
            fill="toself",
            fillcolor="rgba(58,134,255,0.12)",
            line=dict(color="rgba(58,134,255,0)"),
            hoverinfo="skip",
            name="IC 95%",
        ),
        row=1,
        col=1,
    )

    for label, level in fibonacci_levels.items():
        fig.add_hline(y=level, line_dash="dot", line_color="rgba(131,56,236,0.35)", annotation_text=f"Fib {label}", annotation_font_size=10, row=1, col=1)

    if not forecast_points.empty:
        buy_points = forecast_points[forecast_points["type"] == "COMPRA_PRED"]
        sell_points = forecast_points[forecast_points["type"] == "VENTA_PRED"]
        if not buy_points.empty:
            fig.add_trace(go.Scatter(x=buy_points["date"], y=buy_points["price"], mode="markers", name="Minimos prediccion", marker=dict(symbol="triangle-up", size=11, color="#00b050")), row=1, col=1)
        if not sell_points.empty:
            fig.add_trace(go.Scatter(x=sell_points["date"], y=sell_points["price"], mode="markers", name="Maximos prediccion", marker=dict(symbol="triangle-down", size=11, color="#ef233c")), row=1, col=1)

    if not trades.empty:
        buys = trades[trades["Tipo"] == "COMPRA"]
        sells = trades[trades["Tipo"] == "VENTA"]
        if not buys.empty:
            fig.add_trace(go.Scatter(x=buys["Fecha"], y=buys["Precio"], mode="markers", name="Compras", marker=dict(symbol="circle", size=8, color="#06d6a0"), hovertemplate="Compra<br>%{x|%Y-%m-%d}<br>Precio: %{y:.2f}<extra></extra>"), row=1, col=1)
        if not sells.empty:
            fig.add_trace(go.Scatter(x=sells["Fecha"], y=sells["Precio"], mode="markers", name="Ventas", marker=dict(symbol="x", size=9, color="#e63946"), hovertemplate="Venta<br>%{x|%Y-%m-%d}<br>Precio: %{y:.2f}<extra></extra>"), row=1, col=1)

    news_markers = build_news_markers(news_df, data)
    if not news_markers.empty:
        color_map = {"bullish": "#38b000", "bearish": "#d90429", "neutral": "#ffbe0b"}
        size_map = {"low": 11, "medium": 15, "high": 19}
        fig.add_trace(
            go.Scatter(
                x=news_markers["date"],
                y=news_markers["plot_price"],
                mode="markers",
                name="Noticias",
                marker=dict(
                    symbol="star",
                    size=[size_map.get(value, 11) for value in news_markers["impact"]],
                    color=[color_map.get(value, "#ffbe0b") for value in news_markers["sentiment"]],
                    line=dict(color="white", width=0.8),
                ),
                customdata=np.stack(
                    [
                        news_markers["title_es"].fillna(news_markers["title"].fillna("")),
                        news_markers["title"].fillna(""),
                        news_markers["sentiment"].fillna(""),
                        news_markers["impact"].fillna(""),
                        news_markers["source"].fillna(""),
                    ],
                    axis=1,
                ),
                hovertemplate="%{x|%Y-%m-%d}<br>%{customdata[0]}<br>Original: %{customdata[1]}<br>Sentimiento: %{customdata[2]}<br>Impacto: %{customdata[3]}<br>Fuente: %{customdata[4]}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    fig.add_trace(go.Scatter(x=data["date"], y=data["RSI14"], name="RSI14", line=dict(color="#8338ec")), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#d90429", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#38b000", row=2, col=1)

    fig.add_trace(go.Scatter(x=data["date"], y=data["MACD"], name="MACD", line=dict(color="#4361ee")), row=3, col=1)
    fig.add_trace(go.Scatter(x=data["date"], y=data["MACD_SIGNAL"], name="Signal", line=dict(color="#fb8500")), row=3, col=1)
    fig.add_trace(go.Bar(x=data["date"], y=data["MACD_HIST"], name="Histograma", marker_color=np.where(data["MACD_HIST"] >= 0, "#2a9d8f", "#e76f51")), row=3, col=1)

    fig.add_trace(go.Scatter(x=equity_df["Fecha"], y=equity_df["Equity"], name="Equity", line=dict(color="#264653", width=2.5)), row=4, col=1)

    fig.update_layout(
        title=(
            f"BTC/EUR Quant Bot | Precio actual: {latest_price:.2f} EUR | "
            f"Capital final: {summary['capital_final']:.2f} EUR | "
            f"MDD: {summary['max_drawdown_pct']:.2f}%"
        ),
        template="plotly_white",
        hovermode="x unified",
        height=1200,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    fig.update_xaxes(title_text="Fecha", row=4, col=1)
    fig.update_yaxes(title_text="Precio EUR", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="Equity EUR", row=4, col=1)
    return fig
