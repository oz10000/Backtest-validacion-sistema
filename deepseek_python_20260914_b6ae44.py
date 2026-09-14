"""
DAPS Certification Lab — Gestión de portafolio.
Maneja capital, posiciones concurrentes y sizing.
"""
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PortfolioState:
    """Estado del portafolio en un momento dado."""
    timestamp: pd.Timestamp
    equity: float
    cash: float
    open_positions: int
    open_symbols: List[str] = field(default_factory=list)
    daily_pnl: float = 0.0
    peak_equity: float = 0.0
    drawdown_pct: float = 0.0


class Portfolio:
    """Gestión de portafolio con límites de riesgo."""

    def __init__(self, config: dict):
        self.initial_capital = config['backtest']['initial_capital']
        self.risk_per_trade = config['backtest']['risk_per_trade']
        self.max_concurrent = config['backtest']['max_concurrent']
        self.max_daily_dd = config['backtest']['max_daily_drawdown']

        self.equity = self.initial_capital
        self.cash = self.initial_capital
        self.peak_equity = self.initial_capital
        self.open_positions: List[dict] = []
        self.equity_curve: List[PortfolioState] = []
        self.trades: List = []

    def can_open(self, symbol: str, timestamp: pd.Timestamp) -> bool:
        """Verifica si se puede abrir una nueva posición."""
        # Máximo de posiciones concurrentes
        if len(self.open_positions) >= self.max_concurrent:
            return False

        # No duplicar símbolo
        if any(p['symbol'] == symbol for p in self.open_positions):
            return False

        # Drawdown diario
        current_dd = (self.equity - self.peak_equity) / self.peak_equity
        if current_dd < -self.max_daily_dd:
            logger.warning(f"DD diario excedido: {current_dd:.2%}")
            return False

        return True

    def compute_position_size(self, entry_price: float, stop_loss: float,
                               leverage: int = 1) -> float:
        """Calcula el tamaño de posición basado en riesgo fijo."""
        risk_amount = self.equity * self.risk_per_trade
        per_unit_risk = abs(entry_price - stop_loss)
        if per_unit_risk <= 0:
            return 0.0
        qty = risk_amount / per_unit_risk
        # Limitar por leverage
        max_notional = self.equity * leverage
        qty = min(qty, max_notional / entry_price)
        return qty

    def register_open(self, trade):
        """Registra la apertura de una posición."""
        self.open_positions.append({
            'symbol': trade.symbol,
            'trade': trade,
            'entry_time': trade.entry_time,
        })

    def register_close(self, trade):
        """Registra el cierre y actualiza equity."""
        self.trades.append(trade)

        # Remover de posiciones abiertas
        self.open_positions = [
            p for p in self.open_positions
            if p['symbol'] != trade.symbol or p['entry_time'] != trade.entry_time
        ]

        # Actualizar equity
        self.equity += trade.pnl_abs
        self.cash = self.equity

        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        # Registrar estado
        dd = (self.equity - self.peak_equity) / self.peak_equity
        self.equity_curve.append(PortfolioState(
            timestamp=trade.exit_time,
            equity=self.equity,
            cash=self.cash,
            open_positions=len(self.open_positions),
            peak_equity=self.peak_equity,
            drawdown_pct=dd * 100,
        ))

    def get_equity_series(self) -> pd.Series:
        """Retorna la serie temporal de equity."""
        if not self.equity_curve:
            return pd.Series([self.initial_capital])
        df = pd.DataFrame([
            {'timestamp': s.timestamp, 'equity': s.equity}
            for s in self.equity_curve
        ])
        return df.set_index('timestamp')['equity'].sort_index()