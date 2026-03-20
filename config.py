from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os


DEFAULT_SYMBOL = "BTCEUR"
PRICE_LOOKBACK_YEARS = 5
BACKTEST_LOOKBACK_DAYS = 365
SUPPORT_RESISTANCE_WINDOW = 90
FORECAST_DAYS = 30
MONTE_CARLO_PATHS = 1000
NEWS_LIMIT = 20
TRADING_DAYS_PER_YEAR = 365

FEE_RATE = 0.001
SLIPPAGE_RATE = 0.0005
STOP_LOSS_PCT = 0.07
TAKE_PROFIT_PCT = 0.15
TRAILING_STOP_PCT = 0.08
COOLDOWN_DAYS = 3

OUTPUT_DIR = Path("/tmp") if os.getenv("VERCEL") else Path(".")
OPERATIONS_CSV = OUTPUT_DIR / "btc_eur_operaciones.csv"
NEWS_CSV = OUTPUT_DIR / "btc_eur_noticias.csv"
EQUITY_CSV = OUTPUT_DIR / "btc_eur_equity_curve.csv"
CHART_HTML = OUTPUT_DIR / "btc_eur_trading_system.html"


@dataclass
class AppConfig:
    symbol: str
    api_key: str
    api_secret: str
    initial_capital: float
    risk_per_trade_pct: float
    news_api_key: str
    news_provider: str
    forecast_days: int = FORECAST_DAYS
    monte_carlo_paths: int = MONTE_CARLO_PATHS
    fee_rate: float = FEE_RATE
    slippage_rate: float = SLIPPAGE_RATE
    stop_loss_pct: float = STOP_LOSS_PCT
    take_profit_pct: float = TAKE_PROFIT_PCT
    trailing_stop_pct: float = TRAILING_STOP_PCT
    cooldown_days: int = COOLDOWN_DAYS


def prompt_with_default(label: str, default: str) -> str:
    value = input(f"{label} [{default}]: ").strip()
    return value or default


def prompt_float(label: str, default: float, minimum: float = 0.0, maximum: Optional[float] = None) -> float:
    raw = input(f"{label} [{default}]: ").strip()
    try:
        value = float(raw) if raw else float(default)
    except ValueError:
        value = float(default)
    if value < minimum:
        value = minimum
    if maximum is not None and value > maximum:
        value = maximum
    return value


def load_config() -> AppConfig:
    print("=" * 80)
    print("BTC/EUR ANALYTICS BOT")
    print("=" * 80)
    print("Modo publico de Binance disponible sin API Key.")
    print("Las claves quedan listas para trading real y endpoints privados.")
    print("")

    symbol = prompt_with_default("Par a analizar", DEFAULT_SYMBOL)
    api_key = input("Binance API Key [opcional]: ").strip() or os.getenv("BINANCE_API_KEY", "")
    api_secret = input("Binance API Secret [opcional]: ").strip() or os.getenv("BINANCE_API_SECRET", "")
    initial_capital = prompt_float("Capital inicial en EUR", 10000.0, minimum=1.0)
    risk_per_trade_pct = prompt_float("Porcentaje de capital por operacion", 10.0, minimum=0.1, maximum=100.0)
    news_provider = prompt_with_default("Proveedor noticias (none/cryptopanic/newsapi)", "none").lower()
    default_news_key = os.getenv("CRYPTOPANIC_API_KEY", "") if news_provider == "cryptopanic" else os.getenv("NEWSAPI_API_KEY", "")
    news_api_key = input("API Key noticias [opcional]: ").strip() or default_news_key

    return AppConfig(
        symbol=symbol.upper(),
        api_key=api_key,
        api_secret=api_secret,
        initial_capital=initial_capital,
        risk_per_trade_pct=risk_per_trade_pct,
        news_api_key=news_api_key,
        news_provider=news_provider,
    )
