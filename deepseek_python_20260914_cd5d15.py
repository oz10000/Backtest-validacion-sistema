"""
DAPS Certification Lab — Bayesian Optimization.
"""
import logging
import numpy as np
from typing import Dict, Callable

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    logging.warning("scikit-optimize no instalado. Bayesian optimization no disponible.")

logger = logging.getLogger(__name__)


class BayesianOptimizer:
    """Optimización bayesiana de parámetros."""

    def __init__(self, config: dict):
        self.config = config

    def run(self, objective_fn: Callable, space: List, n_calls: int = 100) -> Dict:
        """
        Ejecuta optimización bayesiana.

        Args:
            objective_fn: Función que recibe params y retorna score (a minimizar)
            space: Espacio de búsqueda (skopt space)
            n_calls: Número de evaluaciones
        """
        if not SKOPT_AVAILABLE:
            return {'error': 'scikit-optimize no instalado'}

        logger.info(f"🎯 Bayesian Optimization: {n_calls} evaluaciones")
        result = gp_minimize(
            objective_fn,
            space,
            n_calls=n_calls,
            random_state=42,
            verbose=True,
        )

        return {
            'best_params': dict(zip([s.name for s in space], result.x)),
            'best_score': -result.fun,
            'n_iterations': len(result.func_vals),
            'history': [-v for v in result.func_vals],
        }

    @staticmethod
    def default_space():
        """Espacio de búsqueda por defecto."""
        if not SKOPT_AVAILABLE:
            return None
        return [
            Real(0.2, 0.8, name='sl_atr_mult'),
            Real(1.0, 3.0, name='tp_atr_mult'),
            Integer(1, 10, name='leverage'),
            Real(0.002, 0.008, name='trailing_pct'),
        ]

    @staticmethod
    def default_objective(backtest_fn: Callable) -> Callable:
        """
        Objetivo por defecto: maximizar Sharpe penalizado por DD.
        Score = Sharpe - 0.5 * |MaxDD|
        """
        def _objective(params):
            try:
                metrics = backtest_fn(params)
                sharpe = metrics.get('sharpe', 0)
                max_dd = abs(metrics.get('max_drawdown_pct', 0)) / 100
                score = sharpe - 0.5 * max_dd
                return -score  # gp_minimize minimiza
            except Exception:
                return 0.0  # peor score
        return _objective