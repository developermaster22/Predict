from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable, Optional, Tuple

import pandas as pd
import requests
from deep_translator import GoogleTranslator

from config import NEWS_LIMIT


def keyword_sentiment_score(text: str) -> Tuple[str, str, float]:
    bullish_words = {"approval", "buy", "inflow", "adoption", "record", "bull", "surge", "upgrade", "etf", "institutional", "accumulation"}
    bearish_words = {"hack", "ban", "crash", "selloff", "lawsuit", "outflow", "bear", "liquidation", "fraud", "conflict", "war", "attack"}
    impact_words = {"federal", "sec", "etf", "blackrock", "binance", "fed", "government", "institutional", "macro", "regulation"}

    lowered = text.lower()
    bull_hits = sum(word in lowered for word in bullish_words)
    bear_hits = sum(word in lowered for word in bearish_words)
    impact_hits = sum(word in lowered for word in impact_words)

    if bull_hits > bear_hits:
        sentiment = "bullish"
    elif bear_hits > bull_hits:
        sentiment = "bearish"
    else:
        sentiment = "neutral"

    if impact_hits >= 2 or abs(bull_hits - bear_hits) >= 2:
        impact = "high"
    elif impact_hits == 1 or bull_hits != bear_hits:
        impact = "medium"
    else:
        impact = "low"

    return sentiment, impact, float(bull_hits - bear_hits)


def rank_news_relevance(title: str, published_at: datetime, query_terms: Iterable[str]) -> float:
    lowered = title.lower()
    keyword_score = sum(term.lower() in lowered for term in query_terms)
    days_old = max((datetime.utcnow() - published_at).days, 0)
    freshness_score = max(0, 365 * 5 - days_old) / (365 * 5)
    return keyword_score * 3 + freshness_score


def _empty_news_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "title", "title_es", "url", "source", "sentiment", "impact", "score", "relevance"])


def translate_titles_to_spanish(news_df: pd.DataFrame) -> pd.DataFrame:
    if news_df.empty or "title" not in news_df.columns:
        return news_df

    translated = news_df.copy()
    translator = GoogleTranslator(source="auto", target="es")
    title_es: list[str] = []

    for title in translated["title"].fillna(""):
        if not title:
            title_es.append("")
            continue
        try:
            title_es.append(translator.translate(title))
        except Exception:
            title_es.append(title)

    translated["title_es"] = title_es
    return translated


def fetch_news_from_cryptopanic(api_key: str, limit: int, lookback_days: int = 365 * 5) -> pd.DataFrame:
    response = requests.get(
        "https://cryptopanic.com/api/developer/v2/posts/",
        params={
            "auth_token": api_key,
            "currencies": "BTC",
            "kind": "news",
            "limit": limit,
        },
        headers={"Accept": "application/json"},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    items = payload.get("results") or payload.get("data") or payload.get("posts") or []
    if not items:
        return _empty_news_frame()

    cutoff = datetime.utcnow() - timedelta(days=lookback_days)
    records = []
    for item in items:
        published_at = pd.to_datetime(
            item.get("published_at") or item.get("created_at") or item.get("published"),
            utc=True,
            errors="coerce",
        )
        if pd.isna(published_at):
            continue
        published_at = published_at.tz_convert(None)
        if published_at < cutoff:
            continue
        title = item.get("title") or item.get("headline") or ""
        source_obj = item.get("source") or {}
        if isinstance(source_obj, dict):
            source_name = source_obj.get("title") or source_obj.get("name") or "CryptoPanic"
        else:
            source_name = str(source_obj) if source_obj else "CryptoPanic"
        sentiment, impact, score = keyword_sentiment_score(title)
        records.append(
            {
                "date": published_at,
                "title": title,
                "url": item.get("url") or item.get("domain") or "",
                "source": source_name,
                "sentiment": sentiment,
                "impact": impact,
                "score": score,
                "relevance": rank_news_relevance(title, published_at, ["btc", "bitcoin", "crypto", "etf", "binance", "macro"]),
            }
        )
    if not records:
        return _empty_news_frame()
    frame = pd.DataFrame(records).sort_values(["relevance", "date"], ascending=[False, False]).head(limit).reset_index(drop=True)
    return translate_titles_to_spanish(frame)


def fetch_news_from_newsapi(api_key: str, limit: int, lookback_days: int = 30) -> pd.DataFrame:
    from_date = (datetime.utcnow() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    response = requests.get(
        "https://newsapi.org/v2/everything",
        params={
            "q": "Bitcoin OR BTC OR Binance OR crypto",
            "language": "en",
            "sortBy": "publishedAt",
            "from": from_date,
            "pageSize": limit,
            "apiKey": api_key,
        },
        timeout=20,
    )
    response.raise_for_status()
    articles = response.json().get("articles", [])
    records = []
    for article in articles:
        published_at = pd.to_datetime(article.get("publishedAt"), utc=True, errors="coerce")
        if pd.isna(published_at):
            continue
        published_at = published_at.tz_convert(None)
        title = article.get("title", "")
        sentiment, impact, score = keyword_sentiment_score(title)
        records.append(
            {
                "date": published_at,
                "title": title,
                "url": article.get("url", ""),
                "source": (article.get("source") or {}).get("name", "NewsAPI"),
                "sentiment": sentiment,
                "impact": impact,
                "score": score,
                "relevance": rank_news_relevance(title, published_at, ["btc", "bitcoin", "crypto", "etf", "binance", "macro"]),
            }
        )
    if not records:
        return _empty_news_frame()
    frame = pd.DataFrame(records).sort_values(["relevance", "date"], ascending=[False, False]).head(limit).reset_index(drop=True)
    return translate_titles_to_spanish(frame)


def load_news(provider: str, api_key: str, limit: int = NEWS_LIMIT) -> tuple[pd.DataFrame, Optional[str]]:
    if provider != "none" and not api_key:
        return _empty_news_frame(), "No ingresaste API key para el proveedor de noticias."
    try:
        if provider == "cryptopanic" and api_key:
            news_df = fetch_news_from_cryptopanic(api_key, limit=limit)
            if news_df.empty:
                return news_df, "CryptoPanic respondio sin noticias. Recuerda que el plan gratis DEVELOPER se elimina el 1 de abril de 2026 y los planes pagos indican historial de 1 mes."
            return news_df, None
        if provider == "newsapi" and api_key:
            news_df = fetch_news_from_newsapi(api_key, limit=limit)
            if news_df.empty:
                return news_df, "NewsAPI respondio sin noticias para los parametros actuales."
            return news_df, None
    except requests.RequestException as exc:
        response = getattr(exc, "response", None)
        detail = ""
        if response is not None:
            try:
                detail = f" | detalle: {response.text[:300]}"
            except Exception:
                detail = ""
        return _empty_news_frame(), f"No se pudieron actualizar las noticias: {exc}{detail}"
    return _empty_news_frame(), None


def build_daily_sentiment_map(news_df: pd.DataFrame) -> pd.DataFrame:
    if news_df.empty:
        return pd.DataFrame(columns=["date", "news_sentiment_score", "news_impact_weight", "headline"])

    impact_weight = {"low": 0.5, "medium": 1.0, "high": 1.5}
    sentiment_weight = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}

    mapped = news_df.copy()
    mapped["date"] = pd.to_datetime(mapped["date"]).dt.floor("D")
    mapped["news_impact_weight"] = mapped["impact"].map(impact_weight).fillna(0.5)
    mapped["sentiment_value"] = mapped["sentiment"].map(sentiment_weight).fillna(0.0)
    mapped["news_sentiment_score"] = mapped["news_impact_weight"] * mapped["sentiment_value"]
    return (
        mapped.groupby("date", as_index=False)
        .agg(
            news_sentiment_score=("news_sentiment_score", "sum"),
            news_impact_weight=("news_impact_weight", "max"),
            headline=("title", "first"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
