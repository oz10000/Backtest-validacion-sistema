"""
Adapter para el motor DAPS-SIGNALS Ω original.
Importa los módulos de señal y los ejecuta bar-by-bar sin look-ahead.
"""
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SignalSnapshot:
    """Una señal en un timestamp específico."""
    timestamp: pd.Timestamp
    symbol: str
    direction: str          # 'LONG' | 'SHORT'
    omega_score: float
    tier: str
    consensus_score: float
    mtf_confirmed: bool
    regime: str
    adx: float
    ker: float
    atr_pct: float
    entry_price: float
    stop_loss: float
    take_profit: float
    break_even_price: float
    be_trigger_pct: float
    trailing_distance: float
    trailing_activation_pct: float
    leverage_max: int
    is_valid: bool
    reason: str


class DAPSSignalAdapter:
    """
    Adaptador que conecta el laboratorio con el motor DAPS real.

    Sin look-ahead: para generar señal en tiempo t, usa solo datos hasta t-1.
    """

    def __init__(self, config: dict):
        self.config = config
        daps_path = Path(config['signal_engine']['daps_repo_path']).resolve()
        if daps_path.exists():
            sys.path.insert(0, str(daps_path.parent))
            sys.path.insert(0, str(daps_path))
            logger.info(f"✅ DAPS repo inyectado: {daps_path}")
        else:
            logger.warning(f"⚠️ DAPS repo no encontrado en {daps_path}")
            logger.warning("   Se usará un adaptador simulado.")

        self.min_score = config['signal_engine']['min_omega_score']
        self.omega_threshold = config['signal_engine']['omega_tier_threshold']
        self.stier_threshold = config['signal_engine']['stier_threshold']

        self._load_daps_modules()

    def _load_daps_modules(self):
        """Carga los módulos DAPS si están disponibles."""
        try:
            from signal_engine import SignalEngine
            from omega_ranker import OmegaRanker
            from omega_regime_detector import OmegaRegimeDetector

            self.signal_engine = SignalEngine()
            self.ranker = OmegaRanker()
            self.regime_detector = OmegaRegimeDetector()
            self.available = True
            logger.info("✅ Módulos DAPS cargados correctamente")
        except ImportError as e:
            logger.warning(f"⚠️ No se pudieron cargar módulos DAPS: {e}")
            self.available = False

    def generate_signals_bar_by_bar(
        self,
        symbol: str,
        data: Dict[str, pd.DataFrame],
        warmup_bars: int = 100,
    ) -> List[SignalSnapshot]:
        """
        Genera señales bar-by-bar SIN look-ahead.

        Para cada barra t (desde warmup_bars hasta len-1):
          1. Recorta todos los timeframes hasta t-1
          2. Ejecuta el motor DAPS
          3. Registra la señal si es válida
        """
        if not self.available:
            return self._generate_simulated_signals(symbol, data, warmup_bars)

        # Data debe tener 'entry' (5m), 'confirm' (15m), 'trend' (1h)
        entry_tf = self.config['data']['timeframes']['entry']
        confirm_tf = self.config['data']['timeframes']['confirm']
        trend_tf = self.config['data']['timeframes']['trend']

        entry_df = data.get(entry_tf)
        if entry_df is None or len(entry_df) < warmup_bars:
            return []

        signals = []
        for t in range(warmup_bars, len(entry_df)):
            ts = entry_df.index[t]

            # Recortar sin look-ahead
            window_5m = entry_df.iloc[:t]           # excluye t
            window_15m = self._slice_by_ts(
                data.get(confirm_tf), ts, lookback=200
            )
            window_1h = self._slice_by_ts(
                data.get(trend_tf), ts, lookback=200
            )

            if window_15m is None or window_1h is None:
                continue
            if len(window_15m) < 30 or len(window_1h) < 30:
                continue

            try:
                per_tf = {'5m': window_5m, '15m': window_15m, '1h': window_1h}
                base_sig = self.signal_engine.evaluate(symbol, per_tf)
                if base_sig is None:
                    continue
                sig_dict = self.signal_engine.to_dict(base_sig)
                if not sig_dict.get('is_valid'):
                    continue

                omega_list = self.ranker.rank([sig_dict], {symbol: window_5m})
                if not omega_list:
                    continue
                omega_sig = omega_list[0]

                signals.append(SignalSnapshot(
                    timestamp=ts,
                    symbol=symbol,
                    direction=omega_sig.direction,
                    omega_score=omega_sig.omega_score,
                    tier=omega_sig.tier,
                    consensus_score=omega_sig.consensus_score,
                    mtf_confirmed=omega_sig.mtf_confirmed,
                    regime=omega_sig.regime,
                    adx=sig_dict.get('adx', 0.0),
                    ker=sig_dict.get('ker', 0.0),
                    atr_pct=sig_dict.get('atr_pct', 0.0),
                    entry_price=omega_sig.entry_price,
                    stop_loss=omega_sig.stop_loss,
                    take_profit=omega_sig.take_profit,
                    break_even_price=omega_sig.break_even_price,
                    be_trigger_pct=omega_sig.be_trigger_pct,
                    trailing_distance=omega_sig.trailing_distance,
                    trailing_activation_pct=omega_sig.trailing_activation_pct,
                    leverage_max=omega_sig.leverage_max,
                    is_valid=True,
                    reason='OK',
                ))
            except Exception as e:
                logger.debug(f"Error en {symbol} {ts}: {e}")
                continue

        return signals

    @staticmethod
    def _slice_by_ts(df: Optional[pd.DataFrame],
                     ts: pd.Timestamp,
                     lookback: int = 200) -> Optional[pd.DataFrame]:
        """Recorta un DataFrame hasta ts (exclusive)."""
        if df is None or df.empty:
            return None
        try:
            sliced = df.loc[df.index < ts]
            return sliced.iloc[-lookback:] if len(sliced) > lookback else sliced
        except Exception:
            return None

    def _generate_simulated_signals(self, symbol: str,
                                     data: Dict[str, pd.DataFrame],
                                     warmup_bars: int) -> List[SignalSnapshot]:
        """Fallback: genera señales aleatorias para testing del motor."""
        entry_tf = self.config['data']['timeframes']['entry']
        df = data.get(entry_tf)
        if df is None or len(df) < warmup_bars:
            return []

        signals = []
        rng = np.random.default_rng(hash(symbol) % 2**32)
        for t in range(warmup_bars, len(df)):
            if rng.random() < 0.02:  # 2% de barras generan señal
                ts = df.index[t]
                close = df['close'].iloc[t]
                atr_pct = (df['high'].iloc[t-14:t] - df['low'].iloc[t-14:t]).mean() / close
                direction = 'LONG' if rng.random() > 0.5 else 'SHORT'
                score = rng.uniform(40, 98)
                signals.append(SignalSnapshot(
                    timestamp=ts, symbol=symbol, direction=direction,
                    omega_score=score,
                    tier=self._score_to_tier(score),
                    consensus_score=score, mtf_confirmed=rng.random() > 0.5,
                    regime='Tendencia Fuerte', adx=rng.uniform(25, 45),
                    ker=rng.uniform(0.4, 0.8), atr_pct=atr_pct,
                    entry_price=close,
                    stop_loss=close * (1 - 0.004) if direction == 'LONG' else close * (1 + 0.004),
                    take_profit=close * (1 + 0.008) if direction == 'LONG' else close * (1 - 0.008),
                    break_even_price=close * (1 + 0.001) if direction == 'LONG' else close * (1 - 0.001),
                    be_trigger_pct=0.002, trailing_distance=0.004,
                    trailing_activation_pct=0.005, leverage_max=5,
                    is_valid=True, reason='SIMULATED',
                ))
        return signals

    def _score_to_tier(self, score: float) -> str:
        if score >= self.omega_threshold:
            return 'Ω-TIER'
        if score >= self.stier_threshold:
            return 'S-TIER'
        if score >= 60:
            return 'A-TIER'
        if score >= 40:
            return 'B-TIER'
        return 'NO-TIER'
