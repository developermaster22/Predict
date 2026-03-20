from __future__ import annotations

from datetime import timedelta
from typing import Dict, Tuple
import math

import numpy as np
import pandas as pd

from config import SUPPORT_RESISTANCE_WINDOW, TRADING_DAYS_PER_YEAR


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def calculate_macd(close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = data["Close"].shift(1)
    tr = pd.concat(
        [
            data["High"] - data["Low"],
            (data["High"] - prev_close).abs(),
            (data["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def calculate_bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower


def enrich_indicators(data: pd.DataFrame) -> pd.DataFrame:
    enriched = data.copy()
    enriched["support_90d"] = enriched["Low"].rolling(SUPPORT_RESISTANCE_WINDOW).min()
    enriched["resistance_90d"] = enriched["High"].rolling(SUPPORT_RESISTANCE_WINDOW).max()
    enriched["SMA20"] = enriched["Close"].rolling(20).mean()
    enriched["SMA50"] = enriched["Close"].rolling(50).mean()
    enriched["RSI14"] = calculate_rsi(enriched["Close"], 14)
    macd, signal, hist = calculate_macd(enriched["Close"])
    enriched["MACD"] = macd
    enriched["MACD_SIGNAL"] = signal
    enriched["MACD_HIST"] = hist
    enriched["ATR14"] = calculate_atr(enriched, 14)
    bb_mid, bb_upper, bb_lower = calculate_bollinger(enriched["Close"], 20, 2.0)
    enriched["BB_MID"] = bb_mid
    enriched["BB_UPPER"] = bb_upper
    enriched["BB_LOWER"] = bb_lower
    enriched["daily_return"] = enriched["Close"].pct_change()
    enriched["log_return"] = np.log(enriched["Close"] / enriched["Close"].shift(1))
    return enriched


def calculate_fibonacci_levels(data: pd.DataFrame, lookback: int = SUPPORT_RESISTANCE_WINDOW) -> Dict[str, float]:
    recent = data.tail(lookback)
    swing_high = recent["High"].max()
    swing_low = recent["Low"].min()
    price_range = swing_high - swing_low
    return {
        "0.0%": swing_high,
        "23.6%": swing_high - price_range * 0.236,
        "38.2%": swing_high - price_range * 0.382,
        "50.0%": swing_high - price_range * 0.5,
        "61.8%": swing_high - price_range * 0.618,
        "78.6%": swing_high - price_range * 0.786,
        "100.0%": swing_low,
    }


def monte_carlo_forecast(data: pd.DataFrame, forecast_days: int, paths: int, seed: int = 42):
    log_returns = data["log_return"].dropna()
    if log_returns.empty:
        raise RuntimeError("No hay suficientes retornos historicos para la simulacion Monte Carlo.")

    daily_mean = log_returns.mean()
    daily_vol = log_returns.std()
    annual_drift = daily_mean * TRADING_DAYS_PER_YEAR
    annual_vol = daily_vol * math.sqrt(TRADING_DAYS_PER_YEAR)
    last_price = float(data["Close"].iloc[-1])

    rng = np.random.default_rng(seed)
    shocks = rng.normal(loc=(daily_mean - 0.5 * daily_vol ** 2), scale=daily_vol, size=(forecast_days, paths))
    simulated_paths = last_price * np.exp(np.cumsum(shocks, axis=0))

    future_dates = pd.date_range(start=data["date"].iloc[-1] + timedelta(days=1), periods=forecast_days, freq="D")
    simulations = pd.DataFrame(simulated_paths, index=future_dates)

    forecast = pd.DataFrame(index=future_dates)
    forecast["mean"] = simulations.mean(axis=1)
    forecast["p05"] = simulations.quantile(0.05, axis=1)
    forecast["p25"] = simulations.quantile(0.25, axis=1)
    forecast["p75"] = simulations.quantile(0.75, axis=1)
    forecast["p95"] = simulations.quantile(0.95, axis=1)
    forecast["prob_above_last"] = (simulations.gt(last_price)).mean(axis=1)

    summary = {
        "last_price": last_price,
        "annual_drift": float(annual_drift),
        "annual_volatility": float(annual_vol),
        "expected_min_95": float(forecast["p05"].min()),
        "expected_max_95": float(forecast["p95"].max()),
        "expected_final_mean": float(forecast["mean"].iloc[-1]),
        "prob_final_above_last": float(forecast["prob_above_last"].iloc[-1]),
    }
    return forecast.reset_index(names="date"), simulations, summary


def extract_forecast_turning_points(forecast: pd.DataFrame) -> pd.DataFrame:
    points = []
    prices = forecast["mean"].to_numpy()
    dates = forecast["date"].to_list()

    for idx in range(1, len(prices) - 1):
        if prices[idx] < prices[idx - 1] and prices[idx] < prices[idx + 1]:
            points.append({"date": dates[idx], "price": prices[idx], "type": "COMPRA_PRED"})
        if prices[idx] > prices[idx - 1] and prices[idx] > prices[idx + 1]:
            points.append({"date": dates[idx], "price": prices[idx], "type": "VENTA_PRED"})

    return pd.DataFrame(points)
