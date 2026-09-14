"""
DAPS Certification Lab — Métricas de rendimiento.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional


def compute_metrics(trades_df: pd.DataFrame,
                    equity_curve: pd.Series,
                    initial_capital: float) -> Dict:
    """Calcula todas las métricas de rendimiento."""
    if trades_df is None or trades_df.empty:
        return {'error': 'Sin trades'}

    n = len(trades_df)
    wins = trades_df[trades_df['pnl_pct'] > 0]
    losses = trades_df[trades_df['pnl_pct'] <= 0]

    wr = len(wins) / n * 100 if n > 0 else 0
    avg_win = wins['pnl_pct'].mean() if len(wins) > 0 else 0
    avg_loss = losses['pnl_pct'].mean() if len(losses) > 0 else 0
    gp = wins['pnl_pct'].sum() if len(wins) > 0 else 0
    gl = abs(losses['pnl_pct'].sum()) if len(losses) > 0 else 1e-9
    pf = gp / gl if gl > 0 else 0

    expectancy = (wr/100) * avg_win + (1 - wr/100) * avg_loss

    # Equity metrics
    final_equity = equity_curve.iloc[-1] if len(equity_curve) > 0 else initial_capital
    total_return = (final_equity / initial_capital - 1) * 100

    peak = equity_curve.cummax()
    dd = (equity_curve - peak) / peak * 100
    max_dd = dd.min()

    # Sharpe / Sortino (sobre retornos por trade)
    rets = trades_df['pnl_pct'] / 100
    sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
    downside = rets[rets < 0].std()
    sortino = (rets.mean() / downside * np.sqrt(252)) if downside and downside > 0 else 0
    calmar = total_return / abs(max_dd) if max_dd != 0 else 0

    # Recovery Factor
    recovery = total_return / abs(max_dd) if max_dd != 0 else 0

    # CAGR (aproximado)
    try:
        if 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
            t0 = pd.Timestamp(trades_df['entry_time'].min())
            t1 = pd.Timestamp(trades_df['exit_time'].max())
            years = max((t1 - t0).days / 365.25, 1e-6)
            cagr = ((final_equity / initial_capital) ** (1/years) - 1) * 100
        else:
            cagr = 0
    except Exception:
        cagr = 0

    # Riesgo de ruina (empírico)
    risk_of_ruin = float((equity_curve < initial_capital * 0.5).mean() * 100)

    return {
        'total_trades': n,
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': wr,
        'profit_factor': pf,
        'expectancy_pct': expectancy,
        'total_return_pct': total_return,
        'final_equity': float(final_equity),
        'max_drawdown_pct': float(max_dd),
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'recovery_factor': recovery,
        'cagr_pct': cagr,
        'risk_of_ruin_pct': risk_of_ruin,
        'avg_win_pct': float(avg_win),
        'avg_loss_pct': float(avg_loss),
        'avg_duration_min': float(trades_df['duration_minutes'].mean()) if 'duration_minutes' in trades_df else 0,
        'max_consecutive_wins': _max_streak(trades_df, True),
        'max_consecutive_losses': _max_streak(trades_df, False),
    }


def compute_metrics_by_tier(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Métricas desglosadas por tier."""
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()

    tiers = ['Ω-TIER', 'S-TIER', 'A-TIER', 'B-TIER', 'NO-TIER']
    rows = []
    for tier in tiers:
        sub = trades_df[trades_df['tier'] == tier]
        if sub.empty:
            rows.append({
                'tier': tier, 'trades': 0, 'win_rate': 0, 'profit_factor': 0,
                'expectancy': 0, 'avg_pnl': 0, 'max_win': 0, 'max_loss': 0,
            })
            continue

        wins = sub[sub['pnl_pct'] > 0]
        losses = sub[sub['pnl_pct'] <= 0]
        wr = len(wins) / len(sub) * 100
        gp = wins['pnl_pct'].sum() if len(wins) > 0 else 0
        gl = abs(losses['pnl_pct'].sum()) if len(losses) > 0 else 1e-9
        pf = gp / gl if gl > 0 else 0
        exp = (wr/100) * (wins['pnl_pct'].mean() if len(wins) > 0 else 0) + \
              (1 - wr/100) * (losses['pnl_pct'].mean() if len(losses) > 0 else 0)

        rows.append({
            'tier': tier,
            'trades': len(sub),
            'win_rate': wr,
            'profit_factor': pf,
            'expectancy': exp,
            'avg_pnl': sub['pnl_pct'].mean(),
            'max_win': sub['pnl_pct'].max(),
            'max_loss': sub['pnl_pct'].min(),
        })
    return pd.DataFrame(rows)


def compute_metrics_by_hour(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Métricas por hora del día (UTC)."""
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()
    df = trades_df.copy()
    df['entry_time'] = pd.to_datetime(df['entry_time'], utc=True)
    df['hour'] = df['entry_time'].dt.hour

    agg = df.groupby('hour').agg(
        trades=('pnl_pct', 'count'),
        pnl_sum=('pnl_pct', 'sum'),
        pnl_mean=('pnl_pct', 'mean'),
        wins=('pnl_pct', lambda s: (s > 0).sum()),
    ).reset_index()
    agg['win_rate'] = agg['wins'] / agg['trades'] * 100
    return agg


def compute_metrics_by_symbol(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Métricas por activo."""
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()
    agg = trades_df.groupby('symbol').agg(
        trades=('pnl_pct', 'count'),
        pnl_sum=('pnl_pct', 'sum'),
        pnl_mean=('pnl_pct', 'mean'),
        wins=('pnl_pct', lambda s: (s > 0).sum()),
    ).reset_index()
    agg['win_rate'] = agg['wins'] / agg['trades'] * 100
    return agg.sort_values('pnl_sum', ascending=False)


def _max_streak(df: pd.DataFrame, positive: bool) -> int:
    if df.empty:
        return 0
    mask = (df['pnl_pct'] > 0) if positive else (df['pnl_pct'] <= 0)
    max_streak = 0
    current = 0
    for m in mask:
        if m:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak