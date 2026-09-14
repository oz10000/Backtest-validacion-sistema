"""
DAPS Certification Lab — Walk-Forward Validation.
"""
import logging
from datetime import timedelta
from typing import Dict, List

import pandas as pd
import numpy as np

from engine.backtester import Backtester
from metrics.performance import compute_metrics

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """Validación walk-forward con ventanas rodantes."""

    def __init__(self, config: dict):
        self.config = config

    def run(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict:
        """Ejecuta walk-forward validation."""
        entry_tf = self.config['data']['timeframes']['entry']
        wf_config = self.config['optimization']['walk_forward']

        train_days = wf_config['train_days']
        val_days = wf_config['validation_days']
        test_days = wf_config['test_days']
        n_windows = wf_config['n_windows']

        # Encontrar rango temporal global
        all_times = []
        for sym_data in data.values():
            if entry_tf in sym_data:
                all_times.append(sym_data[entry_tf].index.min())
                all_times.append(sym_data[entry_tf].index.max())

        if not all_times:
            return {'error': 'Sin datos'}

        global_start = max(all_times[0:2])  # último start
        global_end = min(all_times[2:4])    # primer end

        total_days = (global_end - global_start).days
        window_span = train_days + val_days + test_days
        max_windows = max(1, (total_days - window_span) // test_days)
        n_windows = min(n_windows, max_windows)

        logger.info(f"🔄 Walk-Forward: {n_windows} ventanas")
        logger.info(f"   Rango: {global_start.date()} → {global_end.date()}")

        windows_results = []
        for i in range(n_windows):
            offset_days = i * test_days
            train_start = global_start + timedelta(days=offset_days)
            train_end = train_start + timedelta(days=train_days)
            val_end = train_end + timedelta(days=val_days)
            test_end = val_end + timedelta(days=test_days)

            logger.info(f"  Ventana {i+1}: train {train_start.date()}-{train_end.date()}, "
                        f"test {val_end.date()}-{test_end.date()}")

            # Split data
            train_data = self._slice_data(data, train_start, train_end, entry_tf)
            test_data = self._slice_data(data, val_end, test_end, entry_tf)

            if not train_data or not test_data:
                continue

            # Backtest en test
            try:
                bt = Backtester(self.config)
                result = bt.run(test_data)

                if not result.trades:
                    continue

                trades_df = bt.trades_to_dataframe(result.trades)
                metrics = compute_metrics(
                    trades_df, result.equity_curve, result.initial_capital
                )

                windows_results.append({
                    'window': i + 1,
                    'test_start': val_end,
                    'test_end': test_end,
                    'metrics': metrics,
                })
            except Exception as e:
                logger.error(f"Error ventana {i+1}: {e}")

        return self._summarize(windows_results)

    @staticmethod
    def _slice_data(data: Dict, start, end, tf: str) -> Dict:
        """Recorta el diccionario de datos a un rango temporal."""
        out = {}
        for sym, tf_data in data.items():
            if tf not in tf_data:
                continue
            df = tf_data[tf]
            sliced = df[(df.index >= start) & (df.index < end)]
            if len(sliced) > 50:
                out[sym] = {tf: sliced}
        return out

    @staticmethod
    def _summarize(windows: List[dict]) -> Dict:
        """Resume los resultados de las ventanas."""
        if not windows:
            return {'error': 'Sin ventanas válidas'}

        returns = [w['metrics']['total_return_pct'] for w in windows]
        sharpes = [w['metrics']['sharpe'] for w in windows]
        wrs = [w['metrics']['win_rate'] for w in windows]

        positive = sum(1 for r in returns if r > 0)

        return {
            'n_windows': len(windows),
            'positive_windows': positive,
            'positive_pct': positive / len(windows) * 100,
            'mean_return_pct': float(np.mean(returns)),
            'std_return_pct': float(np.std(returns)),
            'mean_sharpe': float(np.mean(sharpes)),
            'mean_win_rate': float(np.mean(wrs)),
            'windows': windows,
        }
