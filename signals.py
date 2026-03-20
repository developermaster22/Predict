from __future__ import annotations

import pandas as pd

from config import BACKTEST_LOOKBACK_DAYS
from news import build_daily_sentiment_map


def attach_sentiment(data: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
    enriched = data.copy()
    sentiment = build_daily_sentiment_map(news_df)
    enriched["date_key"] = pd.to_datetime(enriched["date"]).dt.floor("D")
    if sentiment.empty:
        enriched["news_sentiment_score"] = 0.0
        enriched["news_impact_weight"] = 0.0
        enriched["headline"] = ""
        return enriched

    merged = enriched.merge(sentiment, how="left", left_on="date_key", right_on="date")
    merged["news_sentiment_score"] = merged["news_sentiment_score"].fillna(0.0)
    merged["news_impact_weight"] = merged["news_impact_weight"].fillna(0.0)
    merged["headline"] = merged["headline"].fillna("")
    return merged.drop(columns=["date_y"]).rename(columns={"date_x": "date"})


def compute_signal_scores(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    frame["distance_support"] = (frame["Close"] - frame["support_90d"]) / frame["Close"]
    frame["distance_resistance"] = (frame["resistance_90d"] - frame["Close"]) / frame["Close"]

    long_score = pd.Series(0.0, index=frame.index)
    short_score = pd.Series(0.0, index=frame.index)
    long_reason = pd.Series("", index=frame.index)
    short_reason = pd.Series("", index=frame.index)

    conditions = [
        (frame["SMA20"] > frame["SMA50"], 1.3, "tendencia alcista"),
        (frame["MACD"] > frame["MACD_SIGNAL"], 1.2, "MACD positivo"),
        (frame["RSI14"] < 38, 1.1, "RSI bajo"),
        (frame["Close"] < frame["BB_LOWER"], 1.4, "por debajo de Bollinger inferior"),
        (frame["distance_support"] < 0.02, 1.2, "cerca de soporte"),
        (frame["news_sentiment_score"] > 0.5, 0.8, "sentimiento favorable"),
    ]
    for condition, weight, reason in conditions:
        long_score += condition.astype(float) * weight
        long_reason = long_reason.mask(condition, long_reason.where(~condition, long_reason + f"{reason}; "))

    conditions = [
        (frame["SMA20"] < frame["SMA50"], 1.3, "tendencia bajista"),
        (frame["MACD"] < frame["MACD_SIGNAL"], 1.2, "MACD negativo"),
        (frame["RSI14"] > 62, 1.1, "RSI alto"),
        (frame["Close"] > frame["BB_UPPER"], 1.4, "por encima de Bollinger superior"),
        (frame["distance_resistance"] < 0.02, 1.2, "cerca de resistencia"),
        (frame["news_sentiment_score"] < -0.5, 0.8, "sentimiento negativo"),
    ]
    for condition, weight, reason in conditions:
        short_score += condition.astype(float) * weight
        short_reason = short_reason.mask(condition, short_reason.where(~condition, short_reason + f"{reason}; "))

    frame["long_score"] = long_score.round(2)
    frame["short_score"] = short_score.round(2)
    frame["long_reason"] = long_reason.str.strip("; ")
    frame["short_reason"] = short_reason.str.strip("; ")
    frame["signal"] = "HOLD"
    frame.loc[frame["long_score"] >= 2.6, "signal"] = "BUY"
    frame.loc[frame["short_score"] >= 2.6, "signal"] = "SELL"
    frame.loc[(frame["long_score"] >= 2.6) & (frame["short_score"] >= 2.6), "signal"] = "HOLD"

    frame["position_size_multiplier"] = 1.0
    frame.loc[frame["signal"] == "BUY", "position_size_multiplier"] += frame["news_sentiment_score"].clip(-1.5, 1.5) * 0.15
    frame.loc[frame["signal"] == "SELL", "position_size_multiplier"] += (-frame["news_sentiment_score"]).clip(-1.5, 1.5) * 0.15
    frame["position_size_multiplier"] = frame["position_size_multiplier"].clip(0.35, 1.35)
    return frame


def prepare_backtest_dataset(data: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
    frame = attach_sentiment(data, news_df)
    frame = compute_signal_scores(frame)
    return frame.tail(BACKTEST_LOOKBACK_DAYS).reset_index(drop=True)
