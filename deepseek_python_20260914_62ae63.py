"""
DAPS Certification Lab — Bootstrap resampling.
"""
import numpy as np
import pandas as pd
from typing import Dict, List


def bootstrap_confidence_intervals(
    trades_df: pd.DataFrame,
    n_samples: int = 10000,
    confidence_levels: List[float] = None,
) -> Dict:
    """Calcula intervalos de confianza por bootstrap."""
    if confidence_levels is None:
        confidence_levels = [0.90, 0.95, 0.99]

    if trades_df is None or trades_df.empty:
        return {}

    pnls = trades_df['pnl_pct'].values
    n = len(pnls)
    rng = np.random.default_rng(42)

    boot_means = np.zeros(n_samples)
    boot_wrs = np.zeros(n_samples)
    boot_pfs = np.zeros(n_samples)

    for i in range(n_samples):
        sample = rng.choice(pnls, size=n, replace=True)
        boot_means[i] = sample.mean()
        boot_wrs[i] = (sample > 0).mean()

        wins = sample[sample > 0].sum()
        losses = abs(sample[sample <= 0].sum())
        boot_pfs[i] = wins / losses if losses > 0 else 0

    result = {'n_samples': n_samples}
    for cl in confidence_levels:
        alpha = (1 - cl) / 2
        result[f'mean_pnl_ci{int(cl*100)}'] = (
            float(np.percentile(boot_means, alpha * 100)),
            float(np.percentile(boot_means, (1 - alpha) * 100)),
        )
        result[f'win_rate_ci{int(cl*100)}'] = (
            float(np.percentile(boot_wrs, alpha * 100) * 100),
            float(np.percentile(boot_wrs, (1 - alpha) * 100) * 100),
        )
        result[f'profit_factor_ci{int(cl*100)}'] = (
            float(np.percentile(boot_pfs, alpha * 100)),
            float(np.percentile(boot_pfs, (1 - alpha) * 100)),
        )

    return result