import math
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(page_title="FX Alert Dashboard", layout="wide")


def parse_pair(pair: str) -> tuple[str, str]:
    base, quote = pair.split("/")
    return base.strip().upper(), quote.strip().upper()


def pip_size(quote: str) -> float:
    return 0.01 if quote == "JPY" else 0.0001


@st.cache_data(ttl=90)
def fetch_fx_intraday(api_key: str, symbol: str, interval: str = "5min") -> pd.DataFrame:
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": 500,
        "apikey": api_key,
    }
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()

    if payload.get("status") == "error":
        raise RuntimeError(payload.get("message", "Twelve Data devolvio un error."))
    if "code" in payload and "message" in payload:
        raise RuntimeError(f"[{payload['code']}] {payload['message']}")

    values = payload.get("values")
    if not values:
        raise ValueError(
            f"No se encontraron velas en la respuesta de Twelve Data. Claves recibidas: {list(payload.keys())}"
        )

    raw = pd.DataFrame(values)
    if "datetime" not in raw.columns:
        raise ValueError("Respuesta invalida de Twelve Data: falta columna 'datetime'.")

    raw["datetime"] = pd.to_datetime(raw["datetime"], utc=True, errors="coerce")
    raw = raw.set_index("datetime").sort_index()

    for col in ["open", "high", "low", "close"]:
        if col not in raw.columns:
            raise ValueError(f"Respuesta invalida de Twelve Data: falta columna '{col}'.")
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw = raw.dropna().copy()
    return raw


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    delta = out["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)

    out["rsi"] = 100 - (100 / (1 + rs))

    out["bb_mid"] = out["close"].rolling(20).mean()
    rolling_std = out["close"].rolling(20).std(ddof=0)
    out["bb_upper"] = out["bb_mid"] + (2 * rolling_std)
    out["bb_lower"] = out["bb_mid"] - (2 * rolling_std)

    return out


def fibonacci_levels_24h(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {}

    end_time = df.index.max()
    start_time = end_time - pd.Timedelta(hours=24)
    window = df.loc[df.index >= start_time]

    if len(window) < 2:
        window = df.tail(min(288, len(df)))

    high_24h = float(window["high"].max())
    low_24h = float(window["low"].min())
    span = high_24h - low_24h

    if math.isclose(span, 0.0):
        return {
            "0.0": low_24h,
            "0.236": low_24h,
            "0.382": low_24h,
            "0.5": low_24h,
            "0.618": low_24h,
            "0.786": low_24h,
            "1.0": high_24h,
        }

    return {
        "0.0": low_24h,
        "0.236": low_24h + span * 0.236,
        "0.382": low_24h + span * 0.382,
        "0.5": low_24h + span * 0.5,
        "0.618": low_24h + span * 0.618,
        "0.786": low_24h + span * 0.786,
        "1.0": high_24h,
    }


def evaluate_signal(row: pd.Series, fib_levels: dict[str, float], fib_tolerance: float) -> dict[str, bool]:
    price = row.get("close", np.nan)
    rsi = row.get("rsi", np.nan)
    upper = row.get("bb_upper", np.nan)
    lower = row.get("bb_lower", np.nan)

    if np.isnan(price) or np.isnan(rsi) or np.isnan(upper) or np.isnan(lower):
        return {"buy": False, "sell": False, "touch_upper": False, "touch_lower": False}

    touch_upper = price >= upper
    touch_lower = price <= lower

    near_0786 = abs(price - fib_levels.get("0.786", price)) <= fib_tolerance
    near_10 = abs(price - fib_levels.get("1.0", price)) <= fib_tolerance
    near_0236 = abs(price - fib_levels.get("0.236", price)) <= fib_tolerance
    near_00 = abs(price - fib_levels.get("0.0", price)) <= fib_tolerance

    sell = (rsi > 75) and touch_upper and (near_0786 or near_10)
    buy = (rsi < 25) and touch_lower and (near_0236 or near_00)

    return {
        "buy": bool(buy),
        "sell": bool(sell),
        "touch_upper": bool(touch_upper),
        "touch_lower": bool(touch_lower),
        "near_0786": bool(near_0786),
        "near_10": bool(near_10),
        "near_0236": bool(near_0236),
        "near_00": bool(near_00),
    }


def make_chart(df: pd.DataFrame, fib_levels: dict[str, float], title: str, max_bars: int = 260) -> go.Figure:
    view = df.tail(max_bars).copy()

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=view.index,
            open=view["open"],
            high=view["high"],
            low=view["low"],
            close=view["close"],
            name=title,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=view.index,
            y=view["bb_upper"],
            name="Bollinger Upper",
            line=dict(color="#ff6b6b", width=1.3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=view.index,
            y=view["bb_mid"],
            name="Bollinger Mid",
            line=dict(color="#ffd166", width=1.0, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=view.index,
            y=view["bb_lower"],
            name="Bollinger Lower",
            line=dict(color="#06d6a0", width=1.3),
        )
    )

    for lvl, price in fib_levels.items():
        color = "#9aa0a6"
        width = 1
        if lvl in {"0.0", "1.0"}:
            color = "#ef476f"
            width = 1.4
        elif lvl in {"0.236", "0.786"}:
            color = "#118ab2"
            width = 1.2
        fig.add_hline(
            y=price,
            line_dash="dot",
            line_color=color,
            line_width=width,
            annotation_text=f"Fib {lvl}",
            annotation_position="right",
        )

    fig.update_layout(
        title=f"{title} - Velas 5m + Bollinger + Fibonacci 24h",
        xaxis_title="Fecha/Hora (UTC)",
        yaxis_title="Precio",
        template="plotly_dark",
        height=650,
        xaxis_rangeslider_visible=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    return fig


def render_alert_box(signal_state: str, message: str) -> None:
    st.markdown(
        """
        <style>
        .alert-box {
            border-radius: 10px;
            padding: 16px;
            margin: 8px 0 16px 0;
            text-align: center;
            font-size: 22px;
            font-weight: 700;
            color: #ffffff;
            border: 2px solid rgba(255,255,255,0.25);
        }
        .buy {
            background: linear-gradient(90deg, #0f9b0f, #00b09b);
            animation: pulse 1s infinite;
        }
        .sell {
            background: linear-gradient(90deg, #b31217, #e52d27);
            animation: pulse 1s infinite;
        }
        .neutral {
            background: #2f3542;
            color: #dfe4ea;
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(255,255,255,0.35); }
            70% { box-shadow: 0 0 0 14px rgba(255,255,255,0); }
            100% { box-shadow: 0 0 0 0 rgba(255,255,255,0); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div class='alert-box {signal_state}'>{message}</div>",
        unsafe_allow_html=True,
    )


st.title("Dashboard de Alertas FX en Tiempo Real")
st.caption("Reglas: RSI(14) + Bandas de Bollinger + Niveles Fibonacci 24h")

with st.sidebar:
    st.header("Configuracion")
    api_key = st.text_input("Twelve Data API Key", type="password", help="No se guarda fuera de esta sesion")
    pair = st.selectbox("Par FX", ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD"], index=0)
    max_bars = st.slider("Velas visibles en grafico", min_value=120, max_value=500, value=260, step=20)
    refresh_sec = st.slider("Auto refresh (seg)", min_value=0, max_value=300, value=120, step=5)
    fib_tolerance_pips = st.slider("Tolerancia Fib (pips)", min_value=1, max_value=30, value=7, step=1)
    st.button("Actualizar ahora", use_container_width=True)
    st.caption("Nota: en plan free de Twelve Data conviene usar refresh >= 60-120s.")

if not api_key:
    st.info("Ingresa tu API key en el sidebar para iniciar.")
    st.stop()

_, to_symbol = parse_pair(pair)
this_pip = pip_size(to_symbol)
fib_tolerance = fib_tolerance_pips * this_pip

if refresh_sec > 0:
    components.html(
        f"""
        <script>
            setTimeout(function() {{
                window.parent.location.reload();
            }}, {refresh_sec * 1000});
        </script>
        """,
        height=0,
        width=0,
    )

try:
    prices = fetch_fx_intraday(api_key=api_key, symbol=pair)
except Exception as exc:
    st.error(f"Error obteniendo datos: {exc}")
    st.stop()

prices = add_indicators(prices)
fib_levels = fibonacci_levels_24h(prices)

if len(prices) < 30:
    st.warning("No hay suficientes velas para calcular indicadores aun.")
    st.stop()

latest = prices.iloc[-1]
prev_close = prices.iloc[-2]["close"] if len(prices) >= 2 else latest["close"]
change_pct = ((latest["close"] - prev_close) / prev_close) * 100 if prev_close else 0.0

signal = evaluate_signal(latest, fib_levels, fib_tolerance)

if "alert_log" not in st.session_state:
    st.session_state.alert_log = []
if "alert_seen" not in st.session_state:
    st.session_state.alert_seen = set()

latest_ts = prices.index[-1]
if signal["sell"]:
    signal_kind = "SELL"
    signal_key = f"{pair}_{latest_ts.isoformat()}_SELL"
    alert_text = "ALERTA VENTA: RSI alto + toque de banda superior + zona Fib 0.786/1.0"
    alert_class = "sell"
elif signal["buy"]:
    signal_kind = "BUY"
    signal_key = f"{pair}_{latest_ts.isoformat()}_BUY"
    alert_text = "ALERTA COMPRA: RSI bajo + toque de banda inferior + zona Fib 0.236/0.0"
    alert_class = "buy"
else:
    signal_kind = "NONE"
    signal_key = ""
    alert_text = "Sin senal activa"
    alert_class = "neutral"

if signal_kind != "NONE" and signal_key not in st.session_state.alert_seen:
    st.session_state.alert_seen.add(signal_key)
    st.session_state.alert_log.append(
        {
            "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "pair": pair,
            "candle_time": latest_ts.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "signal": signal_kind,
            "price": round(float(latest["close"]), 6),
            "rsi": round(float(latest["rsi"]), 2),
            "touch_band": "upper" if signal_kind == "SELL" else "lower",
            "fib_zone": "0.786/1.0" if signal_kind == "SELL" else "0.236/0.0",
        }
    )

c1, c2, c3 = st.columns(3)
c1.metric("Precio actual", f"{latest['close']:.5f}", delta=f"{change_pct:+.3f}%")
c2.metric("Cambio 5m %", f"{change_pct:+.3f}%")
c3.metric("RSI (14)", f"{latest['rsi']:.2f}")

render_alert_box(alert_class, alert_text)

chart = make_chart(prices, fib_levels, f"{pair}", max_bars=max_bars)
st.plotly_chart(chart, use_container_width=True)

fib_df = pd.DataFrame(
    [{"Nivel": k, "Precio": round(v, 6)} for k, v in fib_levels.items()]
).sort_values("Nivel")

left, right = st.columns([2, 3])
with left:
    st.subheader("Niveles Fibonacci (ultimas 24h)")
    st.dataframe(fib_df, use_container_width=True, hide_index=True)
with right:
    st.subheader("Estado de reglas actuales")
    state_df = pd.DataFrame(
        [
            {"Regla": "RSI > 75", "Cumple": bool(latest["rsi"] > 75)},
            {"Regla": "RSI < 25", "Cumple": bool(latest["rsi"] < 25)},
            {"Regla": "Toque Banda Superior", "Cumple": signal["touch_upper"]},
            {"Regla": "Toque Banda Inferior", "Cumple": signal["touch_lower"]},
            {"Regla": "Cerca Fib 0.786/1.0", "Cumple": bool(signal.get("near_0786") or signal.get("near_10"))},
            {"Regla": "Cerca Fib 0.236/0.0", "Cumple": bool(signal.get("near_0236") or signal.get("near_00"))},
        ]
    )
    st.dataframe(state_df, use_container_width=True, hide_index=True)

st.subheader("Historial de alertas (sesion actual)")
if st.session_state.alert_log:
    logs = pd.DataFrame(st.session_state.alert_log)
    st.dataframe(logs.iloc[::-1], use_container_width=True, hide_index=True)
else:
    st.caption("Aun no se generaron alertas en esta sesion.")
