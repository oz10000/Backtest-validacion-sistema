"""
DAPS Certification Lab — Orquestador principal.
"""
import argparse
import logging
import yaml
from pathlib import Path

import pandas as pd

from data.downloader import HistoricalDownloader
from engine.backtester import Backtester
from metrics.performance import (
    compute_metrics, compute_metrics_by_tier,
    compute_metrics_by_hour, compute_metrics_by_symbol,
)
from metrics.risk import compute_risk_metrics
from validation.monte_carlo import MonteCarloValidator
from validation.bootstrap import bootstrap_confidence_intervals
from validation.markov_analysis import MarkovRegimeAnalyzer
from optimization.grid_search import GridSearchOptimizer
from optimization.walk_forward import WalkForwardValidator
from reports.generator import ReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def load_config(path: str = 'config.yaml') -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_backtest(config: dict):
    """Fase 1: Descarga + Backtest."""
    logger.info("=" * 60)
    logger.info("FASE 1 — DESCARGA DE DATOS Y BACKTEST")
    logger.info("=" * 60)

    downloader = HistoricalDownloader(config)
    data = downloader.download_all()
    logger.info(f"✅ Datos: {len(data)} símbolos")

    backtester = Backtester(config)
    result = backtester.run(data)

    trades_df = backtester.trades_to_dataframe(result.trades)
    metrics = compute_metrics(trades_df, result.equity_curve, result.initial_capital)
    tier_metrics = compute_metrics_by_tier(trades_df)
    hourly_metrics = compute_metrics_by_hour(trades_df)
    symbol_metrics = compute_metrics_by_symbol(trades_df)
    risk_metrics = compute_risk_metrics(trades_df, result.equity_curve, result.initial_capital)

    logger.info(f"📊 Trades: {metrics['total_trades']}")
    logger.info(f"📊 Win Rate: {metrics['win_rate']:.1f}%")
    logger.info(f"📊 Profit Factor: {metrics['profit_factor']:.2f}")
    logger.info(f"📊 Sharpe: {metrics['sharpe']:.2f}")
    logger.info(f"📊 Max DD: {metrics['max_drawdown_pct']:.2f}%")

    return result, trades_df, metrics, tier_metrics, hourly_metrics, symbol_metrics, risk_metrics


def run_validation(config: dict, trades_df: pd.DataFrame, initial_capital: float):
    """Fase 2: Validación (Monte Carlo + Bootstrap)."""
    logger.info("=" * 60)
    logger.info("FASE 2 — VALIDACIÓN ESTADÍSTICA")
    logger.info("=" * 60)

    mc_validator = MonteCarloValidator(config)
    mc_results = mc_validator.run(trades_df, initial_capital)

    logger.info(f"🎲 MC Prob positiva: {mc_results.get('prob_positive', 0) * 100:.1f}%")

    bootstrap_results = bootstrap_confidence_intervals(trades_df)
    return mc_results, bootstrap_results


def run_walk_forward(config: dict, data: dict):
    """Fase 3: Walk-Forward."""
    logger.info("=" * 60)
    logger.info("FASE 3 — WALK-FORWARD VALIDATION")
    logger.info("=" * 60)

    wf = WalkForwardValidator(config)
    results = wf.run(data)

    if 'error' not in results:
        logger.info(f"🔄 Ventanas positivas: {results['positive_pct']:.1f}%")
    return results


def run_optimization(config: dict, data: dict):
    """Fase 4: Grid Search."""
    logger.info("=" * 60)
    logger.info("FASE 4 — GRID SEARCH OPTIMIZATION")
    logger.info("=" * 60)

    grid = config['optimization']['grid_search']
    optimizer = GridSearchOptimizer(config, grid)
    results = optimizer.run(data)

    if not results.empty:
        best = results.loc[results['sharpe'].idxmax()]
        logger.info(f"🎯 Mejor Sharpe: {best['sharpe']:.2f}")
        logger.info(f"🎯 Mejores params: "
                    f"SL={best.get('sl_atr_mult')}, "
                    f"TP={best.get('tp_atr_mult')}, "
                    f"Lev={best.get('leverage')}")
    return results


def main():
    parser = argparse.ArgumentParser(description='DAPS Certification Lab')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--phase', default='all',
                        choices=['all', 'backtest', 'validate', 'wf', 'optimize'])
    args = parser.parse_args()

    config = load_config(args.config)

    if args.phase in ('all', 'backtest'):
        result, trades_df, metrics, tier_m, hourly_m, symbol_m, risk_m = run_backtest(config)
    else:
        return

    if args.phase in ('all', 'validate'):
        mc_results, bootstrap_results = run_validation(
            config, trades_df, result.initial_capital
        )
    else:
        mc_results, bootstrap_results = {}, {}

    if args.phase in ('all', 'wf'):
        downloader = HistoricalDownloader(config)
        data = downloader.download_all()
        wf_results = run_walk_forward(config, data)
    else:
        wf_results = {}

    if args.phase in ('all', 'optimize'):
        downloader = HistoricalDownloader(config)
        data = downloader.download_all()
        opt_results = run_optimization(config, data)
    else:
        opt_results = None

    # Generar reportes
    reporter = ReportGenerator(config)
    outputs = reporter.generate_full_report(
        backtest_result=result,
        metrics=metrics,
        tier_metrics=tier_m,
        hourly_metrics=hourly_m,
        symbol_metrics=symbol_m,
        mc_results=mc_results,
        wf_results=wf_results,
        risk_metrics=risk_m,
        optimization_results=opt_results,
    )

    logger.info("=" * 60)
    logger.info("✅ CERTIFICATION LAB COMPLETADO")
    logger.info("=" * 60)
    for k, v in outputs.items():
        logger.info(f"  {k}: {v}")


if __name__ == '__main__':
    main()
