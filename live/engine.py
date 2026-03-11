"""
LiveEngine — Real-time trading engine.
========================================
Inherits BacktestEngine for strategy loading.

Ce que fait le moteur :
  1. Bootstrap depuis l'exchange (HL/Bitget) : télécharge 2h de candles 1m
     et les expanse en rows synthétiques 1s pour un warmup rapide
  2. Les ticks live s'accumulent ensuite (HL/Bitget REST, 1 tick/s)
  3. compute_features() + generate_signal() appelés tous les decision_step s
  4. Exécution paper via exchange client (market / VWAP / TWAP)
  5. Monitoring TP/SL à chaque tick
  6. TradeResult enregistré + pnl_history mis à jour

Modes d'exécution :
  market : ordre immédiat (slippage Almgren-Chriss en paper)
  vwap   : accumule ticks pendant N sec, exécute au prix pondéré par volume
  twap   : découpe en N tranches à intervalle régulier, prix moyen simple
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from backtest.runner import BacktestEngine
from strategies.common.base_strategy import Signal, TradeResult

logger = logging.getLogger('agarthai.live')

# Safety cap for bootstrap per strategy
# staugustin needs (h_l + z_window) × agg = 318 × 60 ≈ 19,080 rows
# 6h bootstrap = 360 candles × 60s = 21,600 rows
_MAX_PRELOAD_ROWS = 25_000

# Rolling live buffer cap: ~17 days of 1s data
_MAX_BUFFER_ROWS = 1_500_000

# Price history kept for GUI charting (last N points per strategy)
_MAX_PRICE_HIST = 600


class LiveEngine(BacktestEngine):
    """
    Live / paper trading engine.

    Architecture par stratégie :
      - Un thread de fond (_tick_loop) par stratégie
      - Un buffer DataFrame roulant (_data_buffers)
      - Un buffer pair-data (_pair_data_buffers) pour innocent3 (ETH)
      - Un slot de position ouverte (_open_positions)
      - Un historique de prix (_price_history) pour les graphiques GUI
    """

    def __init__(self, config_path: str = 'config/strategies.yaml'):
        super().__init__(config_path)
        self.running   = False
        self.connected = False
        self._exchange_client = None

        # ── Per-strategy state ─────────────────────────────────────────────
        self.strategy_runtime:   Dict[str, Dict]                 = {}
        self._data_buffers:      Dict[str, pd.DataFrame]         = {}
        self._pair_data_buffers: Dict[str, pd.DataFrame]         = {}
        self._open_positions:    Dict[str, Optional[dict]]       = {}
        self._strategy_threads:  Dict[str, threading.Thread]     = {}
        self._stop_events:       Dict[str, threading.Event]      = {}
        self._coin_map:          Dict[str, str]                  = {}

        # ── Shared output (GIL-safe pour lectures GUI) ────────────────────
        self.current_positions: Dict      = {}
        self.pnl_history:       list      = []
        self._signal_log:       list      = []

        # PnL fermés par stratégie (équity curve) + historique de prix GUI
        self._per_strategy_pnl: Dict[str, list] = {}
        self._price_history:    Dict[str, list] = {}  # [{ts, price}, ...]

        # ── Execution mode (market / vwap / twap) ──────────────────────
        self.execution_mode: str  = 'market'   # set via config or connect()
        self.vwap_duration_sec: int = 60       # VWAP accumulation window
        self.twap_duration_sec: int = 120      # TWAP total duration
        self.twap_slices: int       = 5        # TWAP number of sub-orders
        self.bootstrap_minutes: int = 360      # minutes of candles to download
        self.tick_interval_sec: float = 1.0    # tick loop interval (1s → 600s)

        # Pending VWAP/TWAP executions: {strategy_name → pending_info}
        self._pending_executions: Dict[str, dict] = {}

        for name in self.strategies:
            target = self._get_target_rows(name)
            self.strategy_runtime[name] = {
                'active':              False,
                'warmup_required_sec': target,
                'started_at':          None,
                'buffered_sec':        0,
                # Warmup completion
                'warmup_done':         False,
                'warmup_done_at':      None,
                # Live metrics (mis à jour chaque tick)
                'ticks':               0,
                'last_price':          None,
                'state':               'IDLE',
                'unrealized_pnl':      0.0,
                'n_signals_today':     0,
                'n_trades':            0,
                'winning_trades':      0,
            }
            self._open_positions[name]    = None
            self._per_strategy_pnl[name]  = []
            self._price_history[name]     = []

    # ──────────────────────────────────────────────────────────────────────
    # Connection
    # ──────────────────────────────────────────────────────────────────────

    def connect(self, exchange_name: str = 'hyperliquid', paper: bool = True) -> bool:
        try:
            from exchanges.clients import get_client
            self._exchange_client = get_client(exchange_name, paper=paper)
            self.connected = self._exchange_client.connect()
            if self.connected:
                for strat_data in self.strategies.values():
                    strat_data['instance'].set_exchange_client(self._exchange_client)
                # Load execution / bootstrap settings from config/settings.yaml
                self._load_execution_settings()
                logger.info(f"Connected to {exchange_name} (paper={paper})")
            return self.connected
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.connected = False
            return False

    def _load_execution_settings(self):
        """Charge les paramètres d'exécution et bootstrap depuis settings.yaml."""
        import yaml
        from pathlib import Path
        settings_path = Path('config/settings.yaml')
        if not settings_path.exists():
            return
        try:
            with open(settings_path) as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            return

        # Execution mode
        exec_cfg = cfg.get('execution', {})
        if exec_cfg:
            mode = exec_cfg.get('mode', 'market').lower()
            if mode in ('market', 'vwap', 'twap'):
                self.execution_mode = mode
            self.vwap_duration_sec = int(exec_cfg.get('vwap_duration_sec', 60))
            self.twap_duration_sec = int(exec_cfg.get('twap_duration_sec', 120))
            self.twap_slices       = int(exec_cfg.get('twap_slices', 5))
            logger.info(
                f"Execution mode: {self.execution_mode} | "
                f"VWAP={self.vwap_duration_sec}s | "
                f"TWAP={self.twap_duration_sec}s/{self.twap_slices} slices"
            )

        # Bootstrap
        boot_cfg = cfg.get('bootstrap', {})
        if boot_cfg.get('enabled', True):
            self.bootstrap_minutes = int(boot_cfg.get('minutes_back', 360))
            logger.info(f"Bootstrap: {self.bootstrap_minutes} min from exchange")

        # Tick interval
        ti = cfg.get('tick_interval_sec')
        if ti is not None:
            self.tick_interval_sec = max(1, float(ti))
            logger.info(f"Tick interval: {self.tick_interval_sec}s")

    def disconnect(self):
        self.stop()
        self.connected = False
        logger.info("Disconnected")

    # ──────────────────────────────────────────────────────────────────────
    # Start / Stop
    # ──────────────────────────────────────────────────────────────────────

    def start(self, coin: str = 'BTC') -> bool:
        if not self.connected:
            logger.error("Cannot start: not connected")
            return False
        for name in self.strategies:
            self.start_strategy(name, coin=coin)
        return True

    def stop(self):
        for name in list(self.strategies):
            self.stop_strategy(name)
        self.running = False
        logger.info("Live trading STOPPED")

    def start_strategy(self, strategy_name: str, coin: str = 'BTC') -> bool:
        if not self.connected:
            logger.error("Cannot start strategy: not connected")
            return False
        if strategy_name not in self.strategies:
            logger.error(f"Unknown strategy: {strategy_name}")
            return False

        if self._stop_events.get(strategy_name) and \
                not self._stop_events[strategy_name].is_set():
            self.stop_strategy(strategy_name)

        self.strategies[strategy_name]['instance'].set_real_time_mode(True)
        rt = self.strategy_runtime[strategy_name]
        rt['active']       = True
        rt['started_at']   = time.time()
        rt['buffered_sec'] = 0
        rt['warmup_done']  = False
        rt['warmup_done_at'] = None
        self._coin_map[strategy_name] = coin
        self.running = True

        stop_evt = threading.Event()
        self._stop_events[strategy_name] = stop_evt
        t = threading.Thread(
            target=self._tick_loop,
            args=(strategy_name, coin, stop_evt),
            name=f"tick-{strategy_name}",
            daemon=True,
        )
        self._strategy_threads[strategy_name] = t
        t.start()
        logger.info(f"Strategy STARTED: {strategy_name} on {coin}")
        return True

    def stop_strategy(self, strategy_name: str) -> bool:
        if strategy_name not in self.strategies:
            return False

        evt = self._stop_events.get(strategy_name)
        if evt:
            evt.set()

        t = self._strategy_threads.get(strategy_name)
        if t and t.is_alive():
            t.join(timeout=5)

        self.strategies[strategy_name]['instance'].set_real_time_mode(False)
        rt = self.strategy_runtime[strategy_name]
        rt['active']     = False
        rt['started_at'] = None

        self.running = any(r['active'] for r in self.strategy_runtime.values())
        logger.info(f"Strategy STOPPED: {strategy_name}")
        return True

    def emergency_stop(self):
        self.stop()
        self.current_positions.clear()
        self._open_positions = {k: None for k in self._open_positions}
        logger.warning("EMERGENCY STOP — all positions closed")

    # ──────────────────────────────────────────────────────────────────────
    # Tick loop (thread de fond par stratégie)
    # ──────────────────────────────────────────────────────────────────────

    def _tick_loop(self, name: str, coin: str, stop_event: threading.Event):
        """
        Boucle principale pour une stratégie.
        Toutes les secondes : fetch prix → append → monitor TP/SL.
        Tous les decision_step s : compute_features → generate_signal → execute.
        """
        strategy      = self.strategies[name]['instance']
        params        = strategy.params
        decision_step_sec = int(params.get('decision_step', 60))
        # Convert seconds → tick count (adapt to tick_interval)
        decision_ticks = max(1, round(decision_step_sec / self.tick_interval_sec))

        # ── Adapter les paramètres time-based au tick_interval ─────────
        # Tous les seuils qui comparent t (index row) à un nombre de "secondes"
        # doivent être divisés par tick_interval pour rester cohérents.
        # Ex: warmup=7200 → à 5s/tick: 7200/5 = 1440 rows (= toujours 2h)
        if self.tick_interval_sec > 1:
            _TIME_PARAMS = [
                # Global (seconds → row count)
                'warmup', 'live_warmup_sec', 'cooldown_sec',
                # baudouin4
                'w_l', 'live_calib_lookback', 'calib_lookback',
                # innocent3
                'live_coint_window', 'live_ou_win',
                # urbain2
                'reg_window', 'regime_ref_window',
                # childeric1 — all row-count windows (designed for 1s rows)
                'w_fast', 'w_mid', 'w_slow', 'w_vol', 'w_mad',
                'w_flow', 'w_activity', 'beta_win', 'max_hold_sec',
                # Note: staugustin h_l/h_s/H_pi/h_I/h_E/z_window are
                # bar-count params (post-aggregation) — do NOT adapt here.
                # agg_interval is adapted separately above.
            ]
            adjusted = []
            for pname in _TIME_PARAMS:
                if pname in params:
                    orig = int(params[pname])
                    new_val = max(1, round(orig / self.tick_interval_sec))
                    params[pname] = new_val
                    adjusted.append(f"{pname}: {orig}->{new_val}")
            if adjusted:
                logger.info(
                    f"[{name}] Params adaptés (tick={self.tick_interval_sec}s): "
                    + ", ".join(adjusted)
                )

            # agg_interval: nombre de rows pour produire des bars ~1m
            # À 1s/tick → agg=60, à 5s/tick → agg=12, à 10s/tick → agg=6
            if 'agg_interval' in params:
                orig_agg = int(params['agg_interval'])
                effective_agg = max(1, round(orig_agg / self.tick_interval_sec))
                params['agg_interval'] = effective_agg
                logger.info(
                    f"[{name}] agg_interval adapté: {orig_agg} -> {effective_agg} "
                    f"(bars ~{effective_agg * self.tick_interval_sec:.0f}s)"
                )

        # ── Recalculer warmup_required_sec si tick_interval a changé ───
        # (_get_target_rows est appelé dans __init__ avec tick_interval=1.0,
        #  mais connect() peut changer tick_interval_sec plus tard)
        rt         = self.strategy_runtime[name]
        recalc     = self._get_target_rows(name)
        if recalc != int(rt['warmup_required_sec']):
            logger.info(
                f"[{name}] warmup recalculé: "
                f"{rt['warmup_required_sec']} → {recalc} rows "
                f"(tick={self.tick_interval_sec}s)"
            )
            rt['warmup_required_sec'] = recalc

        # ── 1. Bootstrap depuis l'exchange (HL/Bitget) ─────────────────
        buf = self._bootstrap_from_exchange(name, coin)
        self._data_buffers[name] = buf

        warmup_req = int(rt['warmup_required_sec'])

        # Vérification immédiate : warmup déjà atteint via bootstrap ?
        rt['buffered_sec'] = min(len(buf), warmup_req)
        if not rt['warmup_done'] and len(buf) >= warmup_req:
            rt['warmup_done']    = True
            rt['warmup_done_at'] = datetime.now().strftime('%H:%M:%S')
            logger.info(f"[{name}] WARMUP DONE ✓ (bootstrap, {len(buf)} rows)")

        logger.info(
            f"[{name}] Tick loop démarré — buffer={len(buf)} rows, "
            f"warmup_req={warmup_req}, decision_step={decision_step_sec}s "
            f"({decision_ticks} ticks @ {self.tick_interval_sec}s/tick)"
        )

        # ── 2. Pour innocent3 : bootstrap ETH depuis l'exchange ──────────
        has_pair  = hasattr(strategy, 'set_pair_data')
        pair_coin = 'ETH' if has_pair else None
        if has_pair:
            self._bootstrap_pair_data(name, strategy, len(buf))

        prev_price: Optional[float] = buf['last'].iloc[-1] if len(buf) > 0 else None
        prev_pair_price: Optional[float] = None
        if has_pair:
            pair_buf_init = self._pair_data_buffers.get(name, pd.DataFrame())
            if len(pair_buf_init) > 0:
                prev_pair_price = pair_buf_init['last'].iloc[-1]

        # ── 3. Seeder price_history depuis le bootstrap (pour chart GUI) ──
        if len(buf) > 0:
            ph_seed = []
            # Sous-échantillonner : 1 point / 10s pour charger le chart
            step = max(1, len(buf) // _MAX_PRICE_HIST)
            for idx in range(0, len(buf), step):
                row = buf.iloc[idx]
                ts_val = row.get('timestamp')
                if ts_val is not None:
                    ts_str = ts_val.isoformat() if hasattr(ts_val, 'isoformat') else str(ts_val)
                else:
                    ts_str = datetime.now().isoformat()
                ph_seed.append({'ts': ts_str, 'price': float(row['last'])})
            self._price_history[name] = ph_seed[-_MAX_PRICE_HIST:]
            logger.info(f"[{name}] Chart seedé avec {len(self._price_history[name])} points")

        tick_n = 0

        while not stop_event.is_set():
            t0 = time.time()
            try:
                # ── Fetch rich tick data (price + volume + trades + OFI) ─
                tick = self._exchange_client.get_tick_data(coin)
                price = tick['price']
                now   = datetime.now()

                ret_1s = (np.log(price / prev_price)
                          if prev_price and prev_price > 0 else 0.0)
                prev_price = price

                new_row = {
                    'timestamp': now,
                    'last':      price,
                    'ret_1s':    ret_1s,
                    'qty':       tick.get('qty', 0.0),
                    'buy_qty':   tick.get('buy_qty', 0.0),
                    'sell_qty':  tick.get('sell_qty', 0.0),
                    'ofi_proxy': tick.get('ofi_proxy', 0.0),
                    'n_trades':  tick.get('n_trades', 1),
                    'log_price': np.log(price),
                }

                buf = pd.concat([buf, pd.DataFrame([new_row])], ignore_index=True)
                if len(buf) > _MAX_BUFFER_ROWS:
                    buf = buf.iloc[-_MAX_BUFFER_ROWS:].reset_index(drop=True)
                self._data_buffers[name] = buf

                # ── Fetch rich tick ETH (innocent3) ─────────────────────
                if has_pair:
                    pair_tick  = self._exchange_client.get_tick_data(pair_coin)
                    pair_price = pair_tick['price']
                    pair_ret   = (np.log(pair_price / prev_pair_price)
                                  if prev_pair_price and prev_pair_price > 0 else 0.0)
                    prev_pair_price = pair_price

                    pair_row = {
                        'timestamp': now,
                        'last':      pair_price,
                        'ret_1s':    pair_ret,
                        'qty':       pair_tick.get('qty', 0.0),
                        'buy_qty':   pair_tick.get('buy_qty', 0.0),
                        'sell_qty':  pair_tick.get('sell_qty', 0.0),
                        'ofi_proxy': pair_tick.get('ofi_proxy', 0.0),
                        'n_trades':  pair_tick.get('n_trades', 1),
                        'log_price': np.log(pair_price),
                    }
                    pair_buf = self._pair_data_buffers.get(name, pd.DataFrame())
                    pair_buf = pd.concat([pair_buf, pd.DataFrame([pair_row])],
                                         ignore_index=True)
                    if len(pair_buf) > _MAX_BUFFER_ROWS:
                        pair_buf = pair_buf.iloc[-_MAX_BUFFER_ROWS:].reset_index(drop=True)
                    self._pair_data_buffers[name] = pair_buf
                    strategy.set_pair_data(pair_buf)

                # ── Mise à jour métriques runtime ────────────────────────
                rt['buffered_sec'] = min(len(buf), warmup_req)
                rt['ticks']        = rt.get('ticks', 0) + 1
                rt['last_price']   = price
                if hasattr(strategy, '_state'):
                    rt['state'] = strategy._state

                # Vérification warmup en cours d'accumulation live
                if not rt['warmup_done'] and len(buf) >= warmup_req:
                    rt['warmup_done']    = True
                    rt['warmup_done_at'] = now.strftime('%H:%M:%S')
                    logger.info(f"[{name}] WARMUP DONE ✓ (live ticks, {len(buf)} rows)")

                # ── Historique de prix pour graphiques GUI ───────────────
                ph = self._price_history[name]
                ph.append({'ts': now.isoformat(), 'price': price})
                if len(ph) > _MAX_PRICE_HIST:
                    self._price_history[name] = ph[-_MAX_PRICE_HIST:]

                # ── Monitor position ouverte (TP/SL) + unrealized PnL ───
                pos = self._open_positions.get(name)
                if pos:
                    d      = pos['direction']
                    unreal = (price / pos['exec_price'] - 1) * d * pos['size_usd']
                    unreal -= pos.get('fee_usd', 0)
                    pos['unrealized_pnl']  = unreal
                    pos['current_price']   = price
                    rt['unrealized_pnl']   = unreal
                    self.current_positions[name] = pos
                    self._check_tp_sl(name, pos, price, now)
                else:
                    rt['unrealized_pnl'] = 0.0

                # ── Avancer exécution VWAP/TWAP en cours ──────────────────
                pending = self._pending_executions.get(name)
                if pending is not None:
                    self._process_pending_execution(name, price, tick)

                # ── Génération signal tous les decision_ticks ticks ───────
                tick_n += 1
                if tick_n % decision_ticks == 0 and rt.get('warmup_done', False):
                    rt['last_scan_at'] = now.strftime('%H:%M:%S')
                    rt['n_scans'] = rt.get('n_scans', 0) + 1
                    if (self._open_positions.get(name) is None
                            and name not in self._pending_executions):
                        try:
                            df_feat = strategy.compute_features(buf)
                            signal  = strategy.generate_signal(df_feat)
                            if signal is not None:
                                rt['n_signals_today'] = rt.get('n_signals_today', 0) + 1
                                self._log_signal(name, signal)
                                self._execute_signal(name, signal, coin, price)
                                logger.info(
                                    f"[{name}] Signal! dir={signal.direction} "
                                    f"price={price:.2f} conf={signal.confidence:.3f}"
                                )
                            else:
                                logger.debug(
                                    f"[{name}] Scan #{rt['n_scans']} — no signal "
                                    f"(buf={len(buf)} rows, price={price:.2f})"
                                )
                        except Exception as e:
                            logger.error(f"[{name}] Signal error: {e}", exc_info=True)

            except Exception as e:
                logger.error(f"[{name}] Tick error: {e}")

            elapsed = time.time() - t0
            stop_event.wait(max(0.0, self.tick_interval_sec - elapsed))

        logger.info(f"[{name}] Tick loop exited")

    # ──────────────────────────────────────────────────────────────────────
    # Buffer helpers
    # ──────────────────────────────────────────────────────────────────────

    def _get_target_rows(self, name: str) -> int:
        """
        Nombre minimum de rows nécessaires avant le premier signal valide.
        Utilisé comme cible du progress bar de warmup dans le dashboard.

        baudouin4 : w_l + live_calib_lookback // 4
        innocent3 : live_coint_window + live_ou_win (défaut 900)
        urbain2   : max(reg_window, regime_ref_window)
        fallback  : live_warmup_sec
        """
        params = self.strategies[name]['instance'].params

        if 'baudouin4' in name:
            cb  = int(params.get('live_calib_lookback',
                                  params.get('calib_lookback', 7 * 86400)))
            w_l = int(params.get('w_l', 1800))
            return w_l + cb // 4

        if 'innocent3' in name:
            win    = int(params.get('live_coint_window',
                                     params.get('coint_window', 604800)))
            ou_win = int(params.get('live_ou_win', 900))
            return win + ou_win

        if 'urbain2' in name:
            rw  = int(params.get('reg_window',        1800))
            rrw = int(params.get('regime_ref_window', 1800))
            return max(rw, rrw)

        if 'staugustin' in name:
            # min_bars_needed = h_l + z_window → × effective_agg (adapté au tick_interval)
            # À 1s/tick: agg=60, rows = 318*60 = 19080
            # À 5s/tick: agg=12, rows = 318*12 = 3816
            h_l = int(params.get('h_l', 30))
            zw  = int(params.get('z_window', 288))
            orig_agg = int(params.get('agg_interval', 60))
            effective_agg = max(1, round(orig_agg / self.tick_interval_sec))
            return (h_l + zw) * effective_agg

        if 'childeric1' in name:
            # warmup gate in generate_signal checks t < warmup
            # Key windows: w_mad (1800), w_slow (900), warmup (3600)
            # warmup is already the max → use it directly
            return int(params.get('warmup', params.get('live_warmup_sec', 3600)))

        return int(self.strategies[name]['instance'].get_warmup_sec())

    def _bootstrap_from_exchange(self, name: str, coin: str) -> pd.DataFrame:
        """
        Bootstrap buffer depuis l'exchange (HL/Bitget).
        Télécharge N minutes de candles 1m, les expanse en rows synthétiques 1s.
        → Warmup rapide sans dépendre de fichiers CSV locaux.
        Fallback sur DataFrame vide si réseau indisponible.
        """
        if self._exchange_client is None:
            logger.warning(f"[{name}] Pas de client exchange — buffer vide")
            return pd.DataFrame()

        minutes = self.bootstrap_minutes
        logger.info(f"[{name}] Bootstrap {coin} — {minutes} min de candles 1m "
                     f"depuis l'exchange...")

        try:
            candles = self._exchange_client.fetch_bootstrap_candles(
                coin, minutes_back=minutes
            )
        except Exception as e:
            logger.warning(f"[{name}] Bootstrap échoué ({e}) — buffer vide")
            return pd.DataFrame()

        if not candles:
            logger.warning(f"[{name}] 0 candles reçues — buffer vide")
            return pd.DataFrame()

        # ── Expansion 1m candles → N rows par candle ────────────────────
        # N dépend du tick_interval : 1s → 60 rows, 10s → 6 rows, 60s → 1 row
        # Bruit réaliste : brownian bridge O→C, volume U-shape, OFI variable.
        tick_sec = max(1, int(self.tick_interval_sec))
        rows_per_candle = max(1, 60 // tick_sec)  # ex: 10s → 6 rows

        rng = np.random.default_rng(42)
        rows: list = []
        for c in candles:
            ts_ms  = c['ts_ms']
            o, h, l, cl = c['open'], c['high'], c['low'], c['close']
            vol    = c['volume']
            n_t    = max(c.get('n_trades', 1), 1)
            spread = max(h - l, o * 0.0001)  # au moins 1 bps

            n = rows_per_candle
            # Brownian bridge : O → C avec bruit borné [L, H]
            noise = rng.normal(0, spread * 0.15, size=n)
            bridge = np.linspace(0, 1, n)
            prices_raw = o + (cl - o) * bridge + np.cumsum(noise) * 0.3
            # Re-center pour finir exactement à close
            drift = prices_raw[-1] - cl
            prices_raw -= drift * bridge
            prices_n = np.clip(prices_raw, l, h)

            # Volume U-shape
            u_shape = 1.0 + 0.8 * np.cos(np.linspace(0, 2 * np.pi, n))
            vol_weights = u_shape / u_shape.sum()
            vols_n = vol * vol_weights

            # Buy/sell ratio variable
            bullish = cl >= o
            base_buy_frac = 0.58 if bullish else 0.42
            buy_fracs = base_buy_frac + rng.normal(0, 0.12, size=n)
            buy_fracs = np.clip(buy_fracs, 0.2, 0.8)

            for j in range(n):
                sec_offset = j * tick_sec
                price = float(prices_n[j])
                vol_s = float(vols_n[j])
                bf    = float(buy_fracs[j])
                buy_q  = vol_s * bf
                sell_q = vol_s * (1 - bf)
                trades_per_row = max(n_t // n, 1)

                rows.append({
                    'timestamp': datetime.fromtimestamp(
                        (ts_ms + sec_offset * 1000) / 1000.0
                    ),
                    'last':      price,
                    'qty':       vol_s,
                    'buy_qty':   buy_q,
                    'sell_qty':  sell_q,
                    'ofi_proxy': buy_q - sell_q,
                    'n_trades':  trades_per_row,
                    'log_price': np.log(max(price, 1e-10)),
                })

        df = pd.DataFrame(rows)

        # Compute ret_1s
        if len(df) > 1:
            prices = df['last'].values
            rets = np.zeros(len(prices))
            rets[1:] = np.log(prices[1:] / np.maximum(prices[:-1], 1e-10))
            df['ret_1s'] = rets
        elif len(df) == 1:
            df['ret_1s'] = 0.0

        # Cap à _MAX_PRELOAD_ROWS
        if len(df) > _MAX_PRELOAD_ROWS:
            df = df.iloc[-_MAX_PRELOAD_ROWS:].reset_index(drop=True)

        logger.info(
            f"[{name}] Bootstrap terminé : {len(candles)} candles → "
            f"{len(df)} rows synthétiques 1s"
        )
        return df

    def _bootstrap_pair_data(self, name: str, strategy, n_rows: int):
        """Pour innocent3 : bootstrap ETH depuis l'exchange (même durée que BTC)."""
        if self._exchange_client is None:
            strategy.set_pair_data(pd.DataFrame())
            self._pair_data_buffers[name] = pd.DataFrame()
            return

        try:
            candles = self._exchange_client.fetch_bootstrap_candles(
                'ETH', minutes_back=self.bootstrap_minutes
            )
        except Exception as e:
            logger.warning(f"[{name}] Bootstrap ETH échoué ({e})")
            strategy.set_pair_data(pd.DataFrame())
            self._pair_data_buffers[name] = pd.DataFrame()
            return

        if not candles:
            strategy.set_pair_data(pd.DataFrame())
            self._pair_data_buffers[name] = pd.DataFrame()
            return

        # Expansion avec bruit réaliste (même logique que BTC)
        tick_sec = max(1, int(self.tick_interval_sec))
        rows_per_candle = max(1, 60 // tick_sec)
        rng = np.random.default_rng(123)
        rows: list = []
        for c in candles:
            ts_ms  = c['ts_ms']
            o, cl  = c['open'], c['close']
            h, l   = c.get('high', max(o, cl)), c.get('low', min(o, cl))
            vol    = c['volume']
            n_t    = max(c.get('n_trades', 1), 1)
            spread = max(h - l, o * 0.0001)
            n = rows_per_candle

            noise = rng.normal(0, spread * 0.15, size=n)
            bridge = np.linspace(0, 1, n)
            prices_raw = o + (cl - o) * bridge + np.cumsum(noise) * 0.3
            drift = prices_raw[-1] - cl
            prices_raw -= drift * bridge
            prices_n = np.clip(prices_raw, l, h)

            u_shape = 1.0 + 0.8 * np.cos(np.linspace(0, 2 * np.pi, n))
            vol_weights = u_shape / u_shape.sum()
            vols_n = vol * vol_weights

            bullish = cl >= o
            base_bf = 0.58 if bullish else 0.42
            buy_fracs = base_bf + rng.normal(0, 0.12, size=n)
            buy_fracs = np.clip(buy_fracs, 0.2, 0.8)

            for j in range(n):
                sec_offset = j * tick_sec
                price = float(prices_n[j])
                vol_s = float(vols_n[j])
                bf    = float(buy_fracs[j])
                buy_q  = vol_s * bf
                sell_q = vol_s * (1 - bf)
                rows.append({
                    'timestamp': datetime.fromtimestamp(
                        (ts_ms + sec_offset * 1000) / 1000.0
                    ),
                    'last':      price,
                    'qty':       vol_s,
                    'buy_qty':   buy_q,
                    'sell_qty':  sell_q,
                    'ofi_proxy': buy_q - sell_q,
                    'n_trades':  max(n_t // n, 1),
                    'log_price': np.log(max(price, 1e-10)),
                })

        eth_df = pd.DataFrame(rows)
        if len(eth_df) > 1:
            prices = eth_df['last'].values
            rets = np.zeros(len(prices))
            rets[1:] = np.log(prices[1:] / np.maximum(prices[:-1], 1e-10))
            eth_df['ret_1s'] = rets
        elif len(eth_df) == 1:
            eth_df['ret_1s'] = 0.0

        if len(eth_df) > _MAX_PRELOAD_ROWS:
            eth_df = eth_df.iloc[-_MAX_PRELOAD_ROWS:].reset_index(drop=True)

        strategy.set_pair_data(eth_df)
        self._pair_data_buffers[name] = eth_df
        logger.info(f"[{name}] ETH bootstrap: {len(eth_df)} rows")

    # ──────────────────────────────────────────────────────────────────────
    # Order execution
    # ──────────────────────────────────────────────────────────────────────

    def _execute_signal(self, name: str, signal: Signal,
                        coin: str, live_price: float):
        """
        Dispatch exécution selon le mode configuré :
        - market : ordre immédiat
        - vwap   : accumule ticks, exécute au VWAP
        - twap   : découpe en N tranches à intervalles réguliers
        """
        mode = self.execution_mode

        if mode == 'vwap':
            self._start_vwap_execution(name, signal, coin, live_price)
        elif mode == 'twap':
            self._start_twap_execution(name, signal, coin, live_price)
        else:
            self._execute_market(name, signal, coin, live_price)

    def _execute_market(self, name: str, signal: Signal,
                        coin: str, live_price: float):
        """Exécution market immédiate (mode par défaut)."""
        strategy = self.strategies[name]['instance']
        sl_pct = abs(signal.sl_price / signal.entry_price - 1) if signal.entry_price else 0.003

        try:
            from core.position_sizing import compute_position
            sizing = compute_position(
                capital=strategy.capital_allocated,
                sl_pct=max(sl_pct, 0.001),
                price=live_price,
            )
        except Exception:
            sizing = {'position_usd': strategy.capital_allocated * 0.06,
                      'qty_coin':     strategy.capital_allocated * 0.06 / live_price}

        side = 'buy' if signal.direction == 1 else 'sell'
        qty  = sizing['qty_coin']

        try:
            order = self._exchange_client.place_order(coin, side, qty)
        except Exception as e:
            logger.error(f"[{name}] place_order failed: {e}")
            return

        self._open_position_from_order(name, signal, coin, live_price,
                                       order, sizing, 'MARKET')

    # ──────────────────────────────────────────────────────────────────────
    # VWAP execution
    # ──────────────────────────────────────────────────────────────────────

    def _start_vwap_execution(self, name: str, signal: Signal,
                              coin: str, live_price: float):
        """Démarre une exécution VWAP : accumule prix × volume pendant N sec."""
        strategy = self.strategies[name]['instance']
        sl_pct = abs(signal.sl_price / signal.entry_price - 1) if signal.entry_price else 0.003

        try:
            from core.position_sizing import compute_position
            sizing = compute_position(
                capital=strategy.capital_allocated,
                sl_pct=max(sl_pct, 0.001),
                price=live_price,
            )
        except Exception:
            sizing = {'position_usd': strategy.capital_allocated * 0.06,
                      'qty_coin':     strategy.capital_allocated * 0.06 / live_price}

        self._pending_executions[name] = {
            'mode':       'vwap',
            'signal':     signal,
            'coin':       coin,
            'sizing':     sizing,
            'start_time': time.time(),
            'duration':   self.vwap_duration_sec,
            # Accumulateurs VWAP : Σ(price × volume), Σ(volume)
            'sum_pv':     live_price * 1.0,  # premier tick
            'sum_v':      1.0,
            'n_ticks':    1,
        }
        logger.info(
            f"[{name}] VWAP exécution démarrée — "
            f"accumulation {self.vwap_duration_sec}s"
        )

    # ──────────────────────────────────────────────────────────────────────
    # TWAP execution
    # ──────────────────────────────────────────────────────────────────────

    def _start_twap_execution(self, name: str, signal: Signal,
                              coin: str, live_price: float):
        """Démarre une exécution TWAP : N tranches à intervalle régulier."""
        strategy = self.strategies[name]['instance']
        sl_pct = abs(signal.sl_price / signal.entry_price - 1) if signal.entry_price else 0.003

        try:
            from core.position_sizing import compute_position
            sizing = compute_position(
                capital=strategy.capital_allocated,
                sl_pct=max(sl_pct, 0.001),
                price=live_price,
            )
        except Exception:
            sizing = {'position_usd': strategy.capital_allocated * 0.06,
                      'qty_coin':     strategy.capital_allocated * 0.06 / live_price}

        interval = self.twap_duration_sec / self.twap_slices

        self._pending_executions[name] = {
            'mode':          'twap',
            'signal':        signal,
            'coin':          coin,
            'sizing':        sizing,
            'start_time':    time.time(),
            'duration':      self.twap_duration_sec,
            'n_slices':      self.twap_slices,
            'interval':      interval,
            # Accumulateurs TWAP : prix capturés à chaque tranche
            'prices':        [live_price],   # première tranche
            'next_slice_at': time.time() + interval,
            'slices_done':   1,
        }
        logger.info(
            f"[{name}] TWAP exécution démarrée — "
            f"{self.twap_slices} tranches sur {self.twap_duration_sec}s"
        )

    # ──────────────────────────────────────────────────────────────────────
    # Process pending VWAP / TWAP each tick
    # ──────────────────────────────────────────────────────────────────────

    def _process_pending_execution(self, name: str, price: float,
                                   tick: dict):
        """Appelé à chaque tick pour avancer une exécution VWAP/TWAP."""
        pending = self._pending_executions.get(name)
        if pending is None:
            return

        now = time.time()

        if pending['mode'] == 'vwap':
            vol = tick.get('qty', 1.0) or 1.0
            pending['sum_pv'] += price * vol
            pending['sum_v']  += vol
            pending['n_ticks'] += 1

            elapsed = now - pending['start_time']
            if elapsed >= pending['duration']:
                # Calcul du prix VWAP
                vwap_price = pending['sum_pv'] / pending['sum_v']
                logger.info(
                    f"[{name}] VWAP terminé : {pending['n_ticks']} ticks, "
                    f"prix VWAP=${vwap_price:.2f}"
                )
                self._finalize_algo_execution(
                    name, pending, vwap_price, 'VWAP'
                )

        elif pending['mode'] == 'twap':
            if now >= pending['next_slice_at'] and \
                    pending['slices_done'] < pending['n_slices']:
                pending['prices'].append(price)
                pending['slices_done'] += 1
                pending['next_slice_at'] = now + pending['interval']

            elapsed = now - pending['start_time']
            if elapsed >= pending['duration'] or \
                    pending['slices_done'] >= pending['n_slices']:
                # Calcul du prix TWAP (moyenne simple)
                twap_price = np.mean(pending['prices'])
                logger.info(
                    f"[{name}] TWAP terminé : {len(pending['prices'])} tranches, "
                    f"prix TWAP=${twap_price:.2f}"
                )
                self._finalize_algo_execution(
                    name, pending, twap_price, 'TWAP'
                )

    def _finalize_algo_execution(self, name: str, pending: dict,
                                 exec_price: float, mode_label: str):
        """Finalise une exécution VWAP/TWAP en ouvrant la position."""
        signal  = pending['signal']
        coin    = pending['coin']
        sizing  = pending['sizing']

        side = 'buy' if signal.direction == 1 else 'sell'
        qty  = sizing['qty_coin']

        # Créer un ordre "virtuel" au prix algorithmique
        order = {
            'id':           f'{mode_label.lower()}_{int(time.time() * 1000)}',
            'price':        exec_price,
            'mid_price':    exec_price,
            'qty':          qty,
            'fee':          sizing['position_usd'] * 0.0001,  # maker fee HL
            'slippage_usd': 0.0,  # VWAP/TWAP réduit le slippage
            'paper':        True,
        }
        self._open_position_from_order(
            name, signal, coin, exec_price, order, sizing, mode_label
        )

        # Nettoyer pending
        self._pending_executions.pop(name, None)

    # ──────────────────────────────────────────────────────────────────────
    # Shared position opener
    # ──────────────────────────────────────────────────────────────────────

    def _open_position_from_order(self, name: str, signal: Signal,
                                  coin: str, live_price: float,
                                  order: dict, sizing: dict,
                                  exec_mode: str = 'MARKET'):
        """Ouvre une position à partir d'un ordre exécuté (market/VWAP/TWAP)."""
        exec_price   = order.get('exec_price', order.get('price', live_price))
        fee_usd      = order.get('fee', 0.0)
        slippage_usd = order.get('slippage_usd', 0.0)

        d          = signal.direction
        tp_pct     = abs(signal.tp_price / signal.entry_price - 1) if signal.entry_price else 0.0075
        sl_pct_adj = abs(signal.sl_price / signal.entry_price - 1) if signal.entry_price else 0.0035

        position = {
            'name':           name,
            'coin':           coin,
            'direction':      d,
            'entry_price':    live_price,
            'exec_price':     exec_price,
            'qty':            sizing['qty_coin'],
            'size_usd':       sizing['position_usd'],
            'tp_price':       exec_price * (1 + d * tp_pct),
            'sl_price':       exec_price * (1 - d * sl_pct_adj),
            'fee_usd':        fee_usd,
            'slippage_usd':   slippage_usd,
            'opened_at':      datetime.now(),
            'unrealized_pnl': 0.0,
            'current_price':  live_price,
            'signal':         signal,
            'exec_mode':      exec_mode,
        }
        self._open_positions[name]   = position
        self.current_positions[name] = position

        logger.info(
            f"[{name}] OPENED {'LONG' if d==1 else 'SHORT'} {coin} "
            f"[{exec_mode}] @ ${exec_price:.2f} | "
            f"size=${sizing['position_usd']:.0f} "
            f"| TP=${position['tp_price']:.2f} SL=${position['sl_price']:.2f}"
        )

    # ──────────────────────────────────────────────────────────────────────
    # TP / SL monitoring
    # ──────────────────────────────────────────────────────────────────────

    def _check_tp_sl(self, name: str, pos: dict, price: float, now: datetime):
        d      = pos['direction']
        tp_hit = (d == 1  and price >= pos['tp_price']) or \
                 (d == -1 and price <= pos['tp_price'])
        sl_hit = (d == 1  and price <= pos['sl_price']) or \
                 (d == -1 and price >= pos['sl_price'])
        if tp_hit or sl_hit:
            self._close_position(name, pos, price, 'TP' if tp_hit else 'SL', now)

    def _close_position(self, name: str, pos: dict, exit_price: float,
                        reason: str, now: datetime):
        """Clôture position, calcule PnL net, enregistre TradeResult."""
        try:
            from exchanges.clients import _apply_paper_slippage
            exit_side = 'sell' if pos['direction'] == 1 else 'buy'
            exec_exit, slip_exit = _apply_paper_slippage(
                exit_price, exit_side, pos['size_usd']
            )
        except Exception:
            exec_exit, slip_exit = exit_price, 0.0

        d         = pos['direction']
        gross_pnl = (exec_exit / pos['exec_price'] - 1) * d * pos['size_usd']
        exit_fee  = pos['size_usd'] * 0.00035
        net_pnl   = (gross_pnl
                     - exit_fee
                     - slip_exit
                     - pos.get('fee_usd', 0)
                     - pos.get('slippage_usd', 0))
        duration  = int((now - pos['opened_at']).total_seconds())

        result = TradeResult(
            signal=pos['signal'],
            entry_price=pos['exec_price'],
            exit_price=exec_exit,
            pnl_usd=net_pnl,
            pnl_pct=net_pnl / pos['size_usd'] if pos['size_usd'] else 0,
            fees_usd=exit_fee + pos.get('fee_usd', 0),
            slippage_usd=slip_exit + pos.get('slippage_usd', 0),
            exit_reason=reason,
            duration_sec=duration,
            position_usd=pos['size_usd'],
        )
        self.strategies[name]['instance'].on_trade_close(result)
        self.pnl_history.append(net_pnl)
        self._per_strategy_pnl[name].append(net_pnl)

        # Mise à jour compteurs trades
        rt = self.strategy_runtime.get(name, {})
        rt['n_trades']       = rt.get('n_trades', 0) + 1
        rt['winning_trades'] = rt.get('winning_trades', 0) + (1 if net_pnl > 0 else 0)

        self._open_positions[name] = None
        self.current_positions.pop(name, None)
        if name in self.strategy_runtime:
            self.strategy_runtime[name]['unrealized_pnl'] = 0.0

        logger.info(
            f"[{name}] CLOSED [{reason}] @ ${exec_exit:.2f} "
            f"| net PnL=${net_pnl:+.2f} | duration={duration}s"
        )

    # ──────────────────────────────────────────────────────────────────────
    # Signal log
    # ──────────────────────────────────────────────────────────────────────

    def _log_signal(self, name: str, signal: Signal):
        entry = {
            'strategy':  name,
            'timestamp': signal.timestamp,
            'direction': 'LONG' if signal.direction == 1 else 'SHORT',
            'price':     signal.entry_price,
            'conf':      round(signal.confidence, 3),
        }
        self._signal_log.append(entry)
        if len(self._signal_log) > 200:
            self._signal_log = self._signal_log[-200:]

    # ──────────────────────────────────────────────────────────────────────
    # GUI helpers — métriques, historique, statut
    # ──────────────────────────────────────────────────────────────────────

    def update_runtime(self):
        """Met à jour les compteurs warmup (appelé par chaque refresh du dashboard)."""
        for name, rt in self.strategy_runtime.items():
            if rt['active']:
                buf = self._data_buffers.get(name)
                if buf is not None and len(buf) > 0:
                    rt['buffered_sec'] = min(len(buf), int(rt['warmup_required_sec']))
                elif rt['started_at']:
                    elapsed = int(time.time() - rt['started_at'])
                    rt['buffered_sec'] = min(elapsed, int(rt['warmup_required_sec']))

    def get_risk_metrics(self, name: str) -> Dict:
        """
        Calcule les métriques de risque à partir des trades clôturés.
        Retourne un dict avec : realized, win_rate, max_dd, sharpe, avg_trade, n_trades.
        """
        pnls = self._per_strategy_pnl.get(name, [])
        if not pnls:
            return {
                'realized':  0.0,
                'win_rate':  0.0,
                'max_dd':    0.0,
                'sharpe':    0.0,
                'avg_trade': 0.0,
                'n_trades':  0,
            }

        arr    = np.array(pnls, dtype=float)
        equity = np.cumsum(arr)
        peak   = np.maximum.accumulate(equity)
        dd     = equity - peak

        wins     = int((arr > 0).sum())
        n        = len(arr)
        win_rate = wins / n if n > 0 else 0.0
        max_dd   = float(dd.min()) if len(dd) > 0 else 0.0
        sharpe   = (float(arr.mean() / arr.std()) * np.sqrt(252)
                    if n > 1 and arr.std() > 0 else 0.0)

        return {
            'realized':  float(equity[-1]),
            'win_rate':  win_rate,
            'max_dd':    max_dd,
            'sharpe':    sharpe,
            'avg_trade': float(arr.mean()),
            'n_trades':  n,
        }

    def get_price_history(self, name: str) -> List[Dict]:
        """Retourne l'historique de prix pour les graphiques GUI."""
        return list(self._price_history.get(name, []))

    def get_available_coins(self) -> list:
        if self._exchange_client and self.connected:
            try:
                return self._exchange_client.get_available_coins()
            except Exception:
                return []
        return []

    def get_current_pnl(self) -> list:
        return self.pnl_history

    def get_per_strategy_pnl(self) -> Dict[str, list]:
        return dict(self._per_strategy_pnl)

    def get_live_metrics(self) -> Dict[str, Dict]:
        """Métriques live par stratégie (pour le dashboard GUI)."""
        return {
            name: {
                'ticks':           rt.get('ticks', 0),
                'last_price':      rt.get('last_price'),
                'state':           rt.get('state', '—'),
                'unrealized_pnl':  rt.get('unrealized_pnl', 0.0),
                'n_signals_today': rt.get('n_signals_today', 0),
                'active':          rt.get('active', False),
                'warmup_done':     rt.get('warmup_done', False),
                'warmup_done_at':  rt.get('warmup_done_at'),
                'n_trades':        rt.get('n_trades', 0),
                'winning_trades':  rt.get('winning_trades', 0),
            }
            for name, rt in self.strategy_runtime.items()
        }

    def get_signal_log(self) -> list:
        return list(reversed(self._signal_log))

    def get_status(self) -> Dict:
        self.update_runtime()
        return {
            'running':          self.running,
            'connected':        self.connected,
            'n_strategies':     len(self.strategies),
            'strategies':       {n: s['instance'].get_status()
                                 for n, s in self.strategies.items()},
            'strategy_runtime': self.strategy_runtime,
            'positions':        self.current_positions,
            'total_pnl':        sum(self.pnl_history) if self.pnl_history else 0,
            'per_strategy_pnl': self.get_per_strategy_pnl(),
            'live_metrics':     self.get_live_metrics(),
            'available_coins':  self.get_available_coins(),
            'signal_log':       self.get_signal_log()[:20],
        }
