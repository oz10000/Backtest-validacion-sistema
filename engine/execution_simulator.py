"""
DAPS Certification Lab — Simulador de ejecución de órdenes.

Simula el ciclo completo de un trade: entrada → gestión → salida.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ExitReason(str, Enum):
    TP = 'TP'
    SL = 'SL'
    BE = 'BE'
    TRAILING = 'TRAILING'
    TIME = 'TIME'
    MANUAL = 'MANUAL'


@dataclass
class TradeRecord:
    """Registro completo de un trade."""
    trade_id: int
    symbol: str
    direction: str
    tier: str
    omega_score: float
    regime: str
    entry_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit: float
    break_even_price: float
    be_trigger_pct: float
    trailing_distance: float
    trailing_activation_pct: float
    leverage: int
    size: float

    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    bars_held: int = 0
    duration_minutes: float = 0.0
    pnl_pct: float = 0.0
    pnl_abs: float = 0.0
    mae_pct: float = 0.0    # Maximum Adverse Excursion
    mfe_pct: float = 0.0    # Maximum Favorable Excursion

    be_activated: bool = False
    trailing_activated: bool = False


class ExecutionSimulator:
    """
    Simula la ejecución de un trade bar-by-bar.
    Modela TP, SL, Break Even, Trailing y Time Exit.
    """

    def __init__(self, config: dict):
        self.fee = config['backtest']['fee_per_side']
        self.slippage = config['backtest']['slippage']
        self.spread = config['backtest']['spread']
        self.latency_bars = config['backtest']['latency_bars']
        self.total_cost_per_side = self.fee + self.slippage / 2 + self.spread / 2

    def simulate_trade(
        self,
        signal,
        price_data: pd.DataFrame,
        trade_id: int,
        max_hold_bars: int = 60,
    ) -> Optional[TradeRecord]:
        """
        Simula un trade desde la señal hasta el cierre.

        Args:
            signal: SignalSnapshot con entrada, SL, TP, etc.
            price_data: DataFrame OHLCV del timeframe de entrada
            trade_id: ID único del trade
            max_hold_bars: Máximo de barras a mantener

        Returns:
            TradeRecord o None si no se pudo ejecutar
        """
        # Buscar índice de entrada con latencia
        try:
            entry_idx = price_data.index.get_indexer(
                [signal.timestamp], method='nearest'
            )[0]
        except Exception:
            return None

        entry_idx += self.latency_bars
        if entry_idx < 0 or entry_idx >= len(price_data) - 1:
            return None

        # Precio de entrada con slippage y spread
        if signal.direction == 'LONG':
            entry_price = price_data['open'].iloc[entry_idx] * (1 + self.total_cost_per_side)
        else:
            entry_price = price_data['open'].iloc[entry_idx] * (1 - self.total_cost_per_side)

        # Ajustar SL/TP/BE al precio real de entrada
        sl = signal.stop_loss
        tp = signal.take_profit
        be = signal.break_even_price

        # Crear registro
        trade = TradeRecord(
            trade_id=trade_id,
            symbol=signal.symbol,
            direction=signal.direction,
            tier=signal.tier,
            omega_score=signal.omega_score,
            regime=signal.regime,
            entry_time=price_data.index[entry_idx],
            entry_price=entry_price,
            stop_loss=sl,
            take_profit=tp,
            break_even_price=be,
            be_trigger_pct=signal.be_trigger_pct,
            trailing_distance=signal.trailing_distance,
            trailing_activation_pct=signal.trailing_activation_pct,
            leverage=signal.leverage_max,
            size=0.0,  # se asigna en portfolio
        )

        # Estado de gestión
        be_triggered = False
        trailing_active = False
        best_price = entry_price
        max_favorable = 0.0
        max_adverse = 0.0

        end_idx = min(entry_idx + max_hold_bars, len(price_data) - 1)
        future = price_data.iloc[entry_idx + 1: end_idx + 1]

        for bar_i, (ts, row) in enumerate(future.iterrows(), start=1):
            high = float(row['high'])
            low = float(row['low'])
            close = float(row['close'])

            # Actualizar MAE/MFE
            if signal.direction == 'LONG':
                mfe = (high - entry_price) / entry_price * 100
                mae = (low - entry_price) / entry_price * 100
            else:
                mfe = (entry_price - low) / entry_price * 100
                mae = (entry_price - high) / entry_price * 100
            max_favorable = max(max_favorable, mfe)
            max_adverse = min(max_adverse, mae)

            # 1. SL (más conservador)
            if signal.direction == 'LONG' and low <= sl:
                exit_price = sl * (1 - self.total_cost_per_side)
                return self._close_trade(
                    trade, ts, exit_price, ExitReason.SL,
                    bar_i, max_favorable, max_adverse, be_triggered, trailing_active
                )
            if signal.direction == 'SHORT' and high >= sl:
                exit_price = sl * (1 + self.total_cost_per_side)
                return self._close_trade(
                    trade, ts, exit_price, ExitReason.SL,
                    bar_i, max_favorable, max_adverse, be_triggered, trailing_active
                )

            # 2. TP
            if signal.direction == 'LONG' and high >= tp:
                exit_price = tp * (1 - self.total_cost_per_side)
                return self._close_trade(
                    trade, ts, exit_price, ExitReason.TP,
                    bar_i, max_favorable, max_adverse, be_triggered, trailing_active
                )
            if signal.direction == 'SHORT' and low <= tp:
                exit_price = tp * (1 + self.total_cost_per_side)
                return self._close_trade(
                    trade, ts, exit_price, ExitReason.TP,
                    bar_i, max_favorable, max_adverse, be_triggered, trailing_active
                )

            # 3. Break Even
            if not be_triggered:
                be_trigger_price = (
                    entry_price * (1 + signal.be_trigger_pct)
                    if signal.direction == 'LONG'
                    else entry_price * (1 - signal.be_trigger_pct)
                )
                if (signal.direction == 'LONG' and high >= be_trigger_price) or \
                   (signal.direction == 'SHORT' and low <= be_trigger_price):
                    be_triggered = True
                    sl = be

            # 4. Trailing
            if be_triggered:
                activation_price = (
                    entry_price * (1 + signal.trailing_activation_pct)
                    if signal.direction == 'LONG'
                    else entry_price * (1 - signal.trailing_activation_pct)
                )
                if signal.direction == 'LONG' and close >= activation_price:
                    trailing_active = True
                    best_price = max(best_price, high)
                    new_sl = best_price * (1 - signal.trailing_distance)
                    sl = max(sl, new_sl)
                elif signal.direction == 'SHORT' and close <= activation_price:
                    trailing_active = True
                    best_price = min(best_price, low)
                    new_sl = best_price * (1 + signal.trailing_distance)
                    sl = min(sl, new_sl)

        # Time exit
        if not future.empty:
            last = future.iloc[-1]
            exit_price = float(last['close'])
            exit_price *= (1 - self.total_cost_per_side) if signal.direction == 'LONG' \
                else (1 + self.total_cost_per_side)
            return self._close_trade(
                trade, future.index[-1], exit_price, ExitReason.TIME,
                len(future), max_favorable, max_adverse, be_triggered, trailing_active
            )

        return trade  # trade abierto (no debería pasar)

    def _close_trade(self, trade: TradeRecord, exit_time: pd.Timestamp,
                     exit_price: float, reason: ExitReason, bars_held: int,
                     mfe: float, mae: float, be_activated: bool,
                     trailing_activated: bool) -> TradeRecord:
        """Cierra el trade y calcula PnL."""
        if trade.direction == 'LONG':
            gross = (exit_price - trade.entry_price) / trade.entry_price
        else:
            gross = (trade.entry_price - exit_price) / trade.entry_price

        net = gross - 2 * self.fee  # fee de ambos lados

        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_reason = reason.value
        trade.bars_held = bars_held
        trade.duration_minutes = (exit_time - trade.entry_time).total_seconds() / 60
        trade.pnl_pct = net * 100
        trade.pnl_abs = net * trade.entry_price * trade.size
        trade.mfe_pct = mfe
        trade.mae_pct = mae
        trade.be_activated = be_activated
        trade.trailing_activated = trailing_activated
        return trade
