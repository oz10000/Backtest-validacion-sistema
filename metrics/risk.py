"""
DAPS Certification Lab — Métricas de riesgo.
"""
import numpy as np
import pandas as pd
from typing import Dict


def compute_risk_metrics(trades_df: pd.DataFrame,
                         equity_curve: pd.Series,
                         initial_capital: float) -> Dict:
    """Calcula métricas de riesgo avanzadas."""
    if trades_df is None or trades_df.empty:
        return {'error': 'Sin trades'}

    # VaR y CVaR (95%, 99%)
    pnls = trades_df['pnl_pct'].values
    var_95 = float(np.percentile(pnls, 5))
    var_99 = float(np.percentile(pnls, 1))
    cvar_95 = float(pnls[pnls <= var_95].mean()) if (pnls <= var_95).any() else var_95
    cvar_99 = float(pnls[pnls <= var_99].mean()) if (pnls <= var_99).any() else var_99

    # Drawdown stats
    peak = equity_curve.cummax()
    dd = (equity_curve - peak) / peak
    max_dd = float(dd.min())
    avg_dd = float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0

    # Recovery time
    recovery_times = []
    underwater = dd < 0
    current_underwater = 0
    for uw in underwater:
        if uw:
            current_underwater += 1
        else:
            if current_underwater > 0:
                recovery_times.append(current_underwater)
            current_underwater = 0
    max_recovery_time = max(recovery_times) if recovery_times else 0
    avg_recovery_time = np.mean(recovery_times) if recovery_times else 0

    # Kelly criterion
    wins = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].mean() / 100
    losses = abs(trades_df[trades_df['pnl_pct'] <= 0]['pnl_pct'].mean()) / 100
    wr = (trades_df['pnl_pct'] > 0).mean()
    if losses > 0:
        kelly = wr - (1 - wr) / (wins / losses) if (wins / losses) > 0 else 0
    else:
        kelly = 0

    # Ulcer Index
    ulcer = float(np.sqrt((dd**2).mean()) * 100)

    return {
        'var_95': var_95,
        'var_99': var_99,
        'cvar_95': cvar_95,
        'cvar_99': cvar_99,
        'max_drawdown_pct': max_dd * 100,
        'avg_drawdown_pct': avg_dd * 100,
        'max_recovery_bars': max_recovery_time,
        'avg_recovery_bars': float(avg_recovery_time),
        'kelly_criterion': float(kelly),
        'ulcer_index': ulcer,
    }
