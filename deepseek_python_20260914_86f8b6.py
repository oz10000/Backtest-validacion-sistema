"""
DAPS Certification Lab — Motor de backtesting principal.
Coordina señal → ejecución → portafolio → métricas.
"""
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np

from daps_adapter.signal_adapter import DAPSSignalAdapter, SignalSnapshot
from engine.execution_simulator import ExecutionSimulator, TradeRecord
from engine.portfolio import Portfolio

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Resultado completo de un backtest."""
    trades: List[TradeRecord]
    equity_curve: pd.Series
    initial_capital: float
    final_equity: float
    config: dict


class Backtester:
    """Motor de backtesting bar-by-bar."""

    def __init__(self, config: dict):
        self.config = config
        self.adapter = DAPSSignalAdapter(config)
        self.executor = ExecutionSimulator(config)

    def run(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> BacktestResult:
        """
        Ejecuta el backtest completo.

        Args:
            data: {symbol: {timeframe: df}}

        Returns:
            BacktestResult con trades y equity curve
        """
        portfolio = Portfolio(self.config)
        entry_tf = self.config['data']['timeframes']['entry']
        max_hold = 60

        logger.info(f"🚀 Iniciando backtest sobre {len(data)} símbolos")

        # 1. Generar todas las señales
        all_signals: List[SignalSnapshot] = []
        for symbol, tf_data in data.items():
            logger.info(f"  Generando señales para {symbol}...")
            sigs = self.adapter.generate_signals_bar_by_bar(symbol, tf_data)
            all_signals.extend(sigs)
            logger.info(f"    → {len(sigs)} señales")

        # 2. Ordenar señales cronológicamente
        all_signals.sort(key=lambda s: s.timestamp)
        logger.info(f"📊 Total señales: {len(all_signals)}")

        # 3. Ejecutar trades en orden cronológico
        trade_id = 0
        for sig in all_signals:
            if not portfolio.can_open(sig.symbol, sig.timestamp):
                continue

            # Calcular tamaño
            size = portfolio.compute_position_size(
                sig.entry_price, sig.stop_loss, sig.leverage_max
            )
            if size <= 0:
                continue

            # Obtener datos de precio del símbolo
            symbol_data = data[sig.symbol].get(entry_tf)
            if symbol_data is None:
                continue

            # Simular trade
            trade = self.executor.simulate_trade(
                sig, symbol_data, trade_id, max_hold
            )
            if trade is None:
                continue

            trade.size = size
            trade_id += 1

            # Registrar en portfolio
            portfolio.register_open(trade)
            portfolio.register_close(trade)

        # 4. Construir resultado
        equity = portfolio.get_equity_series()

        logger.info(f"✅ Backtest completo: {len(portfolio.trades)} trades, "
                    f"equity final: ${portfolio.equity:,.2f}")

        return BacktestResult(
            trades=portfolio.trades,
            equity_curve=equity,
            initial_capital=portfolio.initial_capital,
            final_equity=portfolio.equity,
            config=self.config,
        )

    @staticmethod
    def trades_to_dataframe(trades: List[TradeRecord]) -> pd.DataFrame:
        """Convierte lista de TradeRecord a DataFrame."""
        if not trades:
            return pd.DataFrame()
        return pd.DataFrame([t.__dict__ for t in trades])