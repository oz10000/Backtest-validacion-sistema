# data/downloader.py
# ============================================================
# Descargador con paginación para superar el límite de 300 velas
# ============================================================
import os
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
from pathlib import Path

import pandas as pd
import numpy as np
import ccxt

logger = logging.getLogger(__name__)


class HistoricalDownloader:
    """
    Descargador con paginación automática.

    Los exchanges limitan a 300-1500 velas por request. Este descargador
    hace múltiples requests con `since` para acumular el histórico completo.
    """

    # Límites por exchange (velas por request)
    EXCHANGE_LIMITS = {
        'binance': 1000,
        'okx': 300,
        'kraken': 720,
        'mexc': 500,
        'kucoin': 1500,
        'bybit': 1000,
    }

    def __init__(self, config: dict):
        self.config = config
        self.cache_dir = Path(config['data']['cache_dir'])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.exchanges = {}
        self._connect()

    def _connect(self):
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
                if len(self.exchanges) >= 2:
                    break
            except Exception as e:
                logger.warning(f"⚠️ {ex_id}: {e}")

    def download_all(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        symbols = self.config['data']['symbols']
        tfs = self.config['data']['timeframes']
        days = self.config['data']['lookback_days']
        result = {}

        logger.info(f"📥 Descargando {len(symbols)} símbolos × {len(tfs)} TFs "
                    f"({days} días)...")

        for i, symbol in enumerate(symbols):
            logger.info(f"[{i+1}/{len(symbols)}] {symbol}")
            result[symbol] = {}
            for tf_name, tf_code in tfs.items():
                df = self.download_symbol_paginated(symbol, tf_code, days)
                if df is not None and not df.empty:
                    result[symbol][tf_code] = df
                    logger.info(f"  ✅ {symbol} {tf_code}: {len(df)} velas "
                                f"({df.index[0].date()} → {df.index[-1].date()})")
            if not result[symbol]:
                del result[symbol]

        return result

    # --------------------------------------------------------
    # DESCARGA CON PAGINACIÓN
    # --------------------------------------------------------
    def download_symbol_paginated(self, symbol: str, timeframe: str,
                                   days: int) -> Optional[pd.DataFrame]:
        """
        Descarga N días de velas usando paginación con `since`.
        """
        cache_file = self._cache_path(symbol, timeframe)
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                if not df.empty and self._is_fresh(df) and self._has_enough_bars(df, timeframe, days):
                    logger.info(f"    (caché: {len(df)} velas)")
                    return df
            except Exception:
                pass

        # Calcular cuántas velas necesitamos
        minutes = self._tf_to_minutes(timeframe)
        bars_needed = (days * 24 * 60) // minutes
        logger.info(f"    Necesitamos {bars_needed} velas de {timeframe}")

        # Probar exchanges en orden
        for ex_id, ex in self.exchanges.items():
            try:
                df = self._paginate(ex, symbol, timeframe, bars_needed)
                if df is not None and not df.empty:
                    df.to_parquet(cache_file)
                    logger.info(f"    ✅ {ex_id}: {len(df)} velas obtenidas")
                    return df
            except Exception as e:
                logger.warning(f"    {ex_id} falló: {e}")
                time.sleep(2)
        return None

    def _paginate(self, exchange, symbol: str, timeframe: str,
                   bars_needed: int) -> Optional[pd.DataFrame]:
        """
        Paginación hacia atrás con `since`.

        Comienza desde el timestamp más reciente y va hacia atrás
        hasta acumular bars_needed velas.
        """
        ex_id = exchange.id
        limit = self.EXCHANGE_LIMITS.get(ex_id, 300)

        # Calcular rango temporal
        now = exchange.milliseconds()
        minutes = self._tf_to_minutes(timeframe)
        ms_per_bar = minutes * 60 * 1000

        start_ts = now - bars_needed * ms_per_bar
        cursor = start_ts

        all_candles = []
        max_requests = 500  # límite de seguridad
        request_count = 0

        while cursor < now and request_count < max_requests:
            try:
                ohlcv = exchange.fetch_ohlcv(
                    symbol, timeframe, since=cursor, limit=limit
                )
                if not ohlcv:
                    break

                all_candles.extend(ohlcv)

                # Avanzar cursor al último timestamp + 1 vela
                last_ts = ohlcv[-1][0]
                if last_ts <= cursor:
                    break  # no avanza, cortar
                cursor = last_ts + ms_per_bar

                request_count += 1

                # Si obtuvimos menos que el límite, ya terminamos
                if len(ohlcv) < limit:
                    break

                # Rate limit
                time.sleep(exchange.rateLimit / 1000)

            except Exception as e:
                logger.warning(f"    Error paginando: {e}")
                break

        if not all_candles:
            return None

        # Convertir a DataFrame
        df = pd.DataFrame(
            all_candles,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'],
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df.set_index('timestamp').sort_index()
        df = df[~df.index.duplicated(keep='last')]

        # Recortar al rango deseado
        cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=days)
        df = df[df.index >= cutoff]

        return df

    # --------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------
    def _cache_path(self, symbol: str, timeframe: str) -> Path:
        safe = symbol.replace('/', '_')
        return self.cache_dir / f"{safe}_{timeframe}.parquet"

    def _is_fresh(self, df: pd.DataFrame) -> bool:
        try:
            last = df.index[-1]
            if last.tzinfo is None:
                last = last.tz_localize('UTC')
            age_days = (pd.Timestamp.now(tz='UTC') - last).total_seconds() / 86400
            return age_days < 1
        except Exception:
            return False

    def _has_enough_bars(self, df: pd.DataFrame, timeframe: str,
                          days: int) -> bool:
        """Verifica que el caché tenga suficientes velas."""
        minutes = self._tf_to_minutes(timeframe)
        bars_needed = (days * 24 * 60) // minutes
        # Aceptamos si tiene al menos el 80% de las velas necesarias
        return len(df) >= bars_needed * 0.8

    @staticmethod
    def _tf_to_minutes(tf: str) -> int:
        mapping = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440,
        }
        return mapping.get(tf, 5)
