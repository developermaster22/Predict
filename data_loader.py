from __future__ import annotations

from datetime import datetime, timedelta, timezone

from binance.client import Client
import pandas as pd


def build_binance_client(api_key: str, api_secret: str) -> Client:
    if api_key and api_secret:
        return Client(api_key, api_secret)
    return Client()


def fetch_price_history(client: Client, symbol: str, years: int, interval: str = Client.KLINE_INTERVAL_1DAY) -> pd.DataFrame:
    start_dt = datetime.now(timezone.utc) - timedelta(days=years * 365)
    start_str = start_dt.strftime("%d %b %Y")
    klines = client.get_historical_klines(symbol, interval, start_str)
    if not klines:
        raise RuntimeError(f"No se pudieron descargar datos historicos de {symbol}.")

    frame = pd.DataFrame(
        klines,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["date"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True).dt.tz_localize(None)
    frame = frame.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    return frame[["date", "Open", "High", "Low", "Close", "Volume"]].dropna().reset_index(drop=True)


def fetch_latest_price(client: Client, symbol: str) -> float:
    return float(client.get_symbol_ticker(symbol=symbol)["price"])
