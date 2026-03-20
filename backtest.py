from __future__ import annotations

from typing import Dict, List, Tuple
import math

import numpy as np
import pandas as pd

from config import AppConfig


def _calculate_max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = (equity_curve / running_max) - 1
    return float(drawdown.min() * 100)


def run_backtest(data: pd.DataFrame, config: AppConfig) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    cash = config.initial_capital
    btc_position = 0.0
    position_entry_price = 0.0
    position_entry_cost = 0.0
    trailing_stop = None
    last_trade_index = -10_000

    operations: List[Dict[str, object]] = []
    equity_rows: List[Dict[str, float]] = []

    for idx, row in data.iterrows():
        market_price = float(row["Close"])
        high_price = float(row["High"])
        reason = ""
        action = "HOLD"

        if btc_position > 0:
            trailing_stop = max(trailing_stop or 0.0, high_price * (1 - config.trailing_stop_pct))
            stop_price = position_entry_price * (1 - config.stop_loss_pct)
            take_profit_price = position_entry_price * (1 + config.take_profit_pct)

            if market_price <= stop_price:
                action = "SELL"
                reason = "stop_loss"
            elif market_price <= trailing_stop:
                action = "SELL"
                reason = "trailing_stop"
            elif market_price >= take_profit_price:
                action = "SELL"
                reason = "take_profit"
            elif row["signal"] == "SELL":
                action = "SELL"
                reason = row["short_reason"] or "senal de salida"
        else:
            if idx - last_trade_index >= config.cooldown_days and row["signal"] == "BUY":
                action = "BUY"
                reason = row["long_reason"] or "senal de entrada"

        if action == "BUY" and cash > 25:
            invest_fraction = (config.risk_per_trade_pct / 100.0) * float(row["position_size_multiplier"])
            eur_to_invest = cash * min(invest_fraction, 1.0)
            execution_price = market_price * (1 + config.slippage_rate)
            fee = eur_to_invest * config.fee_rate
            net_invested = eur_to_invest - fee
            btc_bought = net_invested / execution_price if execution_price else 0.0
            if btc_bought > 0:
                cash -= eur_to_invest
                btc_position = btc_bought
                position_entry_price = execution_price
                position_entry_cost = eur_to_invest
                trailing_stop = execution_price * (1 - config.trailing_stop_pct)
                last_trade_index = idx
                operations.append(
                    {
                        "Tipo": "COMPRA",
                        "Fecha": row["date"],
                        "Precio": execution_price,
                        "Cantidad BTC": btc_bought,
                        "Monto EUR": eur_to_invest,
                        "Fee EUR": fee,
                        "PnL EUR": 0.0,
                        "PnL %": 0.0,
                        "Score Long": row["long_score"],
                        "Score Short": row["short_score"],
                        "Sentimiento Diario": row["news_sentiment_score"],
                        "Motivo": reason,
                    }
                )

        elif action == "SELL" and btc_position > 0:
            execution_price = market_price * (1 - config.slippage_rate)
            gross_value = btc_position * execution_price
            fee = gross_value * config.fee_rate
            net_value = gross_value - fee
            pnl_eur = net_value - position_entry_cost
            pnl_pct = (pnl_eur / position_entry_cost * 100) if position_entry_cost else 0.0
            cash += net_value
            operations.append(
                {
                    "Tipo": "VENTA",
                    "Fecha": row["date"],
                    "Precio": execution_price,
                    "Cantidad BTC": btc_position,
                    "Monto EUR": net_value,
                    "Fee EUR": fee,
                    "PnL EUR": pnl_eur,
                    "PnL %": pnl_pct,
                    "Score Long": row["long_score"],
                    "Score Short": row["short_score"],
                    "Sentimiento Diario": row["news_sentiment_score"],
                    "Motivo": reason,
                }
            )
            btc_position = 0.0
            position_entry_price = 0.0
            position_entry_cost = 0.0
            trailing_stop = None
            last_trade_index = idx

        equity = cash + (btc_position * market_price)
        equity_rows.append(
            {
                "Fecha": row["date"],
                "Cash": cash,
                "BTC Posicion": btc_position,
                "Equity": equity,
                "Precio": market_price,
            }
        )

    if btc_position > 0:
        final_price = float(data["Close"].iloc[-1]) * (1 - config.slippage_rate)
        gross_value = btc_position * final_price
        fee = gross_value * config.fee_rate
        net_value = gross_value - fee
        pnl_eur = net_value - position_entry_cost
        pnl_pct = (pnl_eur / position_entry_cost * 100) if position_entry_cost else 0.0
        cash += net_value
        operations.append(
            {
                "Tipo": "VENTA",
                "Fecha": data["date"].iloc[-1],
                "Precio": final_price,
                "Cantidad BTC": btc_position,
                "Monto EUR": net_value,
                "Fee EUR": fee,
                "PnL EUR": pnl_eur,
                "PnL %": pnl_pct,
                "Score Long": data["long_score"].iloc[-1],
                "Score Short": data["short_score"].iloc[-1],
                "Sentimiento Diario": data["news_sentiment_score"].iloc[-1],
                "Motivo": "cierre_fin_backtest",
            }
        )
        equity_rows[-1]["Cash"] = cash
        equity_rows[-1]["BTC Posicion"] = 0.0
        equity_rows[-1]["Equity"] = cash

    trades = pd.DataFrame(operations)
    equity_df = pd.DataFrame(equity_rows)
    equity_returns = equity_df["Equity"].pct_change().fillna(0.0)
    closed = trades[trades["Tipo"] == "VENTA"].copy() if not trades.empty else pd.DataFrame()

    gross_profit = float(closed.loc[closed["PnL EUR"] > 0, "PnL EUR"].sum()) if not closed.empty else 0.0
    gross_loss = float(-closed.loc[closed["PnL EUR"] < 0, "PnL EUR"].sum()) if not closed.empty else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
    expectancy = float(closed["PnL EUR"].mean()) if not closed.empty else 0.0
    sharpe = 0.0
    if equity_returns.std() > 0:
        sharpe = float((equity_returns.mean() / equity_returns.std()) * math.sqrt(365))

    summary = {
        "capital_inicial": config.initial_capital,
        "capital_final": float(equity_df["Equity"].iloc[-1]),
        "ganancia_total_eur": float(equity_df["Equity"].iloc[-1] - config.initial_capital),
        "ganancia_total_pct": float((equity_df["Equity"].iloc[-1] / config.initial_capital - 1) * 100),
        "operaciones_totales": float(len(trades)),
        "ventas_cerradas": float(len(closed)),
        "win_rate": float((closed["PnL EUR"] > 0).mean() * 100) if not closed.empty else 0.0,
        "mejor_operacion_pct": float(closed["PnL %"].max()) if not closed.empty else 0.0,
        "peor_operacion_pct": float(closed["PnL %"].min()) if not closed.empty else 0.0,
        "profit_factor": float(profit_factor),
        "expectancy_eur": float(expectancy),
        "max_drawdown_pct": _calculate_max_drawdown(equity_df["Equity"]),
        "sharpe": sharpe,
        "fees_paid_eur": float(trades["Fee EUR"].sum()) if not trades.empty else 0.0,
    }
    return trades, equity_df, summary
