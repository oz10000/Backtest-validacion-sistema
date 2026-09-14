"""
DAPS Certification Lab — Monte Carlo simulation.
"""
import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MonteCarloValidator:
    """Validador Monte Carlo."""

    def __init__(self, config: dict):
        self.config = config
        self.n_iter = config['validation']['monte_carlo']['n_iterations']

    def run(self, trades_df: pd.DataFrame,
            initial_capital: float = 10000.0,
            method: str = 'trade_shuffle') -> Dict:
        """
        Ejecuta simulación Monte Carlo.

        Args:
            trades_df: DataFrame de trades
            initial_capital: Capital inicial
            method: 'trade_shuffle' | 'bootstrap' | 'slippage_var'
        """
        if trades_df is None or trades_df.empty:
            return {'error': 'Sin trades'}

        logger.info(f"🎲 Monte Carlo: {self.n_iter} iteraciones ({method})")

        pnls = trades_df['pnl_pct'].values / 100
        n = len(pnls)

        final_returns = []
        max_dds = []
        sharpes = []
        win_rates = []

        rng = np.random.default_rng(42)

        for i in range(self.n_iter):
            if method == 'trade_shuffle':
                sample = rng.permutation(pnls)
            elif method == 'bootstrap':
                sample = rng.choice(pnls, size=n, replace=True)
            elif method == 'slippage_var':
                noise = rng.normal(0, 0.001, n)
                sample = pnls - abs(noise)
            else:
                sample = pnls

            equity = initial_capital * np.cumprod(1 + sample)
            final_ret = equity[-1] / initial_capital - 1
            peak = np.maximum.accumulate(equity)
            dd = ((equity - peak) / peak).min()
            sharpe = sample.mean() / sample.std() * np.sqrt(252) if sample.std() > 0 else 0
            wr = (sample > 0).mean()

            final_returns.append(final_ret)
            max_dds.append(dd)
            sharpes.append(sharpe)
            win_rates.append(wr)

        final_returns = np.array(final_returns)
        max_dds = np.array(max_dds)

        return {
            'n_iterations': self.n_iter,
            'method': method,
            'final_return_mean': float(final_returns.mean()),
            'final_return_median': float(np.median(final_returns)),
            'final_return_ci95': (
                float(np.percentile(final_returns, 2.5)),
                float(np.percentile(final_returns, 97.5)),
            ),
            'max_dd_mean': float(max_dds.mean()),
            'max_dd_ci95': (
                float(np.percentile(max_dds, 2.5)),
                float(np.percentile(max_dds, 97.5)),
            ),
            'sharpe_mean': float(np.mean(sharpes)),
            'win_rate_mean': float(np.mean(win_rates)),
            'prob_positive': float((final_returns > 0).mean()),
            'prob_dd_gt_20': float((max_dds < -0.20).mean()),
            'prob_dd_gt_30': float((max_dds < -0.30).mean()),
            'worst_case_return': float(final_returns.min()),
            'best_case_return': float(final_returns.max()),
        }