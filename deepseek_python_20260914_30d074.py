"""
DAPS Certification Lab — Grid Search.
"""
import logging
from itertools import product
from typing import Dict, List, Callable

import pandas as pd

from engine.backtester import Backtester
from metrics.performance import compute_metrics

logger = logging.getLogger(__name__)


class GridSearchOptimizer:
    """Búsqueda sistemática en grid de parámetros."""

    def __init__(self, config: dict, param_grid: Dict[str, List]):
        self.config = config
        self.param_grid = param_grid
        self.results: List[dict] = []

    def run(self, data: Dict) -> pd.DataFrame:
        """Ejecuta grid search sobre todos los parámetros."""
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combos = list(product(*values))
        logger.info(f"🔍 Grid search: {len(combos)} combinaciones")

        for i, combo in enumerate(combos):
            params = dict(zip(keys, combo))
            logger.info(f"  [{i+1}/{len(combos)}] {params}")

            # Aplicar params al config
            config = self._apply_params(params)

            try:
                backtester = Backtester(config)
                result = backtester.run(data)

                if not result.trades:
                    continue

                trades_df = backtester.trades_to_dataframe(result.trades)
                metrics = compute_metrics(
                    trades_df, result.equity_curve, result.initial_capital
                )

                row = {**params, **metrics}
                self.results.append(row)
            except Exception as e:
                logger.error(f"Error con {params}: {e}")
                continue

        return pd.DataFrame(self.results)

    def _apply_params(self, params: dict) -> dict:
        """Aplica parámetros al config."""
        import copy
        config = copy.deepcopy(self.config)

        # Mapear params a config
        if 'sl_atr_mult' in params:
            config['backtest']['sl_atr_mult'] = params['sl_atr_mult']
        if 'tp_atr_mult' in params:
            config['backtest']['tp_atr_mult'] = params['tp_atr_mult']
        if 'leverage' in params:
            config['backtest']['fixed_leverage'] = params['leverage']

        return config