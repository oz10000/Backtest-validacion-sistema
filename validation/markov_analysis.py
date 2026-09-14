"""
DAPS Certification Lab — Análisis de régimen con Cadenas de Markov.
"""
import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarkovRegimeAnalyzer:
    """Análisis de transiciones entre regímenes."""

    def __init__(self, states: List[str] = None):
        self.states = states or ['Bull', 'Bear', 'Chop', 'HighVol']
        self.n_states = len(self.states)
        self.transition_matrix = None

    def fit(self, regime_series: pd.Series) -> np.ndarray:
        """
        Ajusta la matriz de transición a partir de una serie de regímenes.

        Args:
            regime_series: Serie temporal con etiquetas de régimen

        Returns:
            Matriz de transición (n_states × n_states)
        """
        # Inicializar matriz con suavizado de Laplace
        trans = np.ones((self.n_states, self.n_states)) * 0.01

        labels = regime_series.values
        for i in range(len(labels) - 1):
            if labels[i] in self.states and labels[i + 1] in self.states:
                idx_from = self.states.index(labels[i])
                idx_to = self.states.index(labels[i + 1])
                trans[idx_from, idx_to] += 1

        # Normalizar filas
        row_sums = trans.sum(axis=1, keepdims=True)
        self.transition_matrix = trans / row_sums

        return self.transition_matrix

    def steady_state(self) -> Dict[str, float]:
        """Calcula el estado estacionario."""
        if self.transition_matrix is None:
            return {}
        eigvals, eigvecs = np.linalg.eig(self.transition_matrix.T)
        idx = np.argmin(np.abs(eigvals - 1.0))
        stationary = np.real(eigvecs[:, idx])
        stationary = np.abs(stationary) / np.abs(stationary).sum()
        return {state: float(p) for state, p in zip(self.states, stationary)}

    def expected_duration(self, state: str) -> float:
        """Duración esperada en un estado (en barras)."""
        if self.transition_matrix is None:
            return 0.0
        idx = self.states.index(state)
        p_stay = self.transition_matrix[idx, idx]
        if p_stay >= 1:
            return float('inf')
        return 1 / (1 - p_stay)

    def probability_of_transition(self, from_state: str, to_state: str) -> float:
        """Probabilidad de transición entre dos estados."""
        if self.transition_matrix is None:
            return 0.0
        i = self.states.index(from_state)
        j = self.states.index(to_state)
        return float(self.transition_matrix[i, j])

    def summarize(self) -> Dict:
        """Resumen completo del análisis."""
        return {
            'states': self.states,
            'transition_matrix': self.transition_matrix.tolist() if self.transition_matrix is not None else [],
            'steady_state': self.steady_state(),
            'expected_durations': {
                s: self.expected_duration(s) for s in self.states
            },
        }
