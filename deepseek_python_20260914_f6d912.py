"""
DAPS Certification Lab — Descargador de datos históricos.
Soporta cripto (CCXT) y datos sintéticos para testing.
"""
import os
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path

import pandas as pd
import numpy as np
import ccxt

logger = logging.getLogger(__name__)


class HistoricalDownloader:
    """Descargador de datos OHLCV con caché en Parquet."""

    def __init__(self, config: dict):
        self.config = config
        self.cache_dir = Path(config['data']['cache_dir'])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.exchanges = {}
        self._connect()

    def _connect(self):
        """Conecta a exchanges en orden de preferencia."""
        primary = self.config['data']['exchange']
        fallbacks = self.config['data'].get('fallback_exchanges', [])
        for ex_id in [primary] + fallbacks:
            try:
                ex = getattr(ccxt, ex_id)({
                    'enableRateLimit': True,
                    'timeout': 30000,
                    'options': {'defaultType': 'spot'},
                })
                ex.load_markets()
                self.exchanges[ex_id] = ex
                logger.info(f"✅ Conectado a {ex_id}")
                if len(self.exchanges) >= 3:
                    break
            except Exception as e:
                logger.warning(f"⚠️ {ex_id}: {e}")

    def download_all(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Descarga todos los símbolos en todos los timeframes.
        Retorna {symbol: {timeframe: df}}.
        """
        symbols = self.config['data']['symbols']
        tfs = self.config['data']['timeframes']
        days = self.config['data']['lookback_days']
        result = {}

        for symbol in symbols:
            result[symbol] = {}
            for tf_name, tf_code in tfs.items():
                df = self.download_symbol(symbol, tf_code, days)
                if df is not None and not df.empty:
                    result[symbol][tf_code] = df
                    logger.info(f"  ✅ {symbol} {tf_code}: {len(df)} velas")
            if not result[symbol]:
                del result[symbol]

        return result

    def download_symbol(self, symbol: str, timeframe: str,
                        days: int) -> Optional[pd.DataFrame]:
        """Descarga un símbolo en un timeframe."""
        cache_file = self._cache_path(symbol, timeframe)
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                if not df.empty and self._is_fresh(df, days):
                    return df
            except Exception:
                pass

        # Calcular limit
        minutes = self._tf_to_minutes(timeframe)
        bars_per_day = 1440 // minutes
        limit = min(bars_per_day * days, 10000)

        for ex_id, ex in self.exchanges.items():
            try:
                ohlcv = ex.fetch_ohlcv(symbol, timeframe, limit=limit)
                if not ohlcv:
                    continue
                df = pd.DataFrame(
                    ohlcv,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'],
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                df = df.set_index('timestamp').sort_index()
                df = df[~df.index.duplicated(keep='last')]
                df.to_parquet(cache_file)
                return df
            except Exception as e:
                logger.warning(f"  {ex_id} falló para {symbol} {timeframe}: {e}")
                time.sleep(1)
        return None

    def _cache_path(self, symbol: str, timeframe: str) -> Path:
        safe = symbol.replace('/', '_')
        return self.cache_dir / f"{safe}_{timeframe}.parquet"

    def _is_fresh(self, df: pd.DataFrame, days: int) -> bool:
        try:
            last = df.index[-1]
            if last.tzinfo is None:
                last = last.tz_localize('UTC')
            age = (pd.Timestamp.now(tz='UTC') - last).total_seconds() / 86400
            return age < 1
        except Exception:
            return False

    @staticmethod
    def _tf_to_minutes(tf: str) -> int:
        mapping = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240, '1d': 1440}
        return mapping.get(tf, 5)