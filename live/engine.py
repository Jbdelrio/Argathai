"""
LiveEngine — Real-time trading engine.
========================================
Inherits BacktestEngine for strategy loading.

Ce que fait le moteur :
  1. Pré-charge le buffer CSV complet (= target warmup) → warmup quasi-instantané
  2. Les ticks live s'accumulent ensuite (Binance REST, 1 tick/s)
  3. compute_features() + generate_signal() appelés tous les decision_step s
  4. Exécution paper via exchange client (prix réel + slippage Almgren-Chriss)
  5. Monitoring TP/SL à chaque tick
  6. TradeResult enregistré + pnl_history mis à jour

Warmup timeline avec pré-chargement CSV :
  baudouin4 : ~instantané (CSV pré-chargé = target rows, ex. 2700 rows ~45 min)
  innocent3 : ~instantané (CSV pré-chargé = target rows, ex. 1800 rows ~30 min)
  urbain2   : ~instantané (CSV pré-chargé = target rows, ex. 1800 rows ~30 min)
  (sans CSV : live accumulation — quelques dizaines de minutes selon params)
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

# Safety cap for CSV pre-load per strategy (never load more than this)
_MAX_PRELOAD_ROWS = 10_000

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
                logger.info(f"Connected to {exchange_name} (paper={paper})")
            return self.connected
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.connected = False
            return False

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
        decision_step = int(params.get('decision_step', 60))

        # ── 1. Pré-charger buffer CSV (= target warmup rows) ─────────────
        buf = self._preload_buffer(name, coin)
        self._data_buffers[name] = buf

        rt         = self.strategy_runtime[name]
        warmup_req = int(rt['warmup_required_sec'])

        # Vérification immédiate : warmup déjà atteint via CSV ?
        rt['buffered_sec'] = min(len(buf), warmup_req)
        if not rt['warmup_done'] and len(buf) >= warmup_req:
            rt['warmup_done']    = True
            rt['warmup_done_at'] = datetime.now().strftime('%H:%M:%S')
            logger.info(f"[{name}] WARMUP DONE ✓ (CSV pré-chargé, {len(buf)} rows)")

        logger.info(
            f"[{name}] Tick loop démarré — buffer={len(buf)} rows, "
            f"warmup_req={warmup_req}, decision_step={decision_step}s"
        )

        # ── 2. Pour innocent3 : charger ETH startup buffer ───────────────
        has_pair  = hasattr(strategy, 'set_pair_data')
        pair_coin = 'ETH' if has_pair else None
        if has_pair:
            self._load_pair_data(name, strategy, buf)

        prev_price: Optional[float] = buf['last'].iloc[-1] if len(buf) > 0 else None
        prev_pair_price: Optional[float] = None
        if has_pair:
            pair_buf_init = self._pair_data_buffers.get(name, pd.DataFrame())
            if len(pair_buf_init) > 0:
                prev_pair_price = pair_buf_init['last'].iloc[-1]

        tick_n = 0

        while not stop_event.is_set():
            t0 = time.time()
            try:
                # ── Fetch prix live BTC ──────────────────────────────────
                price = self._exchange_client.get_price(coin)
                now   = datetime.now()

                ret_1s = (np.log(price / prev_price)
                          if prev_price and prev_price > 0 else 0.0)
                prev_price = price

                new_row = {
                    'timestamp': now,
                    'last':      price,
                    'ret_1s':    ret_1s,
                    'qty':       0.0,
                    'ofi_proxy': 0.0,
                    'n_trades':  1,
                    'log_price': np.log(price),
                }

                buf = pd.concat([buf, pd.DataFrame([new_row])], ignore_index=True)
                if len(buf) > _MAX_BUFFER_ROWS:
                    buf = buf.iloc[-_MAX_BUFFER_ROWS:].reset_index(drop=True)
                self._data_buffers[name] = buf

                # ── Fetch prix ETH (innocent3) ───────────────────────────
                if has_pair:
                    pair_price = self._exchange_client.get_price(pair_coin)
                    pair_ret   = (np.log(pair_price / prev_pair_price)
                                  if prev_pair_price and prev_pair_price > 0 else 0.0)
                    prev_pair_price = pair_price

                    pair_row = {
                        'timestamp': now,
                        'last':      pair_price,
                        'ret_1s':    pair_ret,
                        'qty':       0.0,
                        'ofi_proxy': 0.0,
                        'n_trades':  1,
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

                # ── Génération signal tous les decision_step ticks ───────
                tick_n += 1
                if tick_n % decision_step == 0:
                    if self._open_positions.get(name) is None:
                        try:
                            df_feat = strategy.compute_features(buf)
                            signal  = strategy.generate_signal(df_feat)
                            if signal is not None:
                                rt['n_signals_today'] = rt.get('n_signals_today', 0) + 1
                                self._log_signal(name, signal)
                                self._execute_signal(name, signal, coin, price)
                        except Exception as e:
                            logger.error(f"[{name}] Signal error: {e}", exc_info=True)

            except Exception as e:
                logger.error(f"[{name}] Tick error: {e}")

            elapsed = time.time() - t0
            stop_event.wait(max(0.0, 1.0 - elapsed))

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

        return int(self.strategies[name]['instance'].get_warmup_sec())

    def _preload_buffer(self, name: str, coin: str) -> pd.DataFrame:
        """
        Pré-charge depuis le CSV le nombre exact de rows nécessaires au warmup.
        → Warmup quasi-instantané dès le démarrage.
        Fallback sur DataFrame vide si CSV indisponible.
        """
        target = self._get_target_rows(name)
        rows_to_load = min(target, _MAX_PRELOAD_ROWS)
        symbol = coin.lower().replace('usdt', '').replace('-perp', '')
        try:
            from data.loader import load_1s
            df = load_1s(symbol)
            if len(df) > rows_to_load:
                df = df.iloc[-rows_to_load:].copy()
            df = df.reset_index(drop=True)
            logger.info(
                f"[{name}] CSV pré-chargé : {len(df)}/{rows_to_load} rows "
                f"(target warmup={target})"
            )
            return df
        except Exception as e:
            logger.warning(
                f"[{name}] CSV indisponible ({e}) — démarrage buffer vide"
            )
            return pd.DataFrame()

    def _load_pair_data(self, name: str, strategy, buf: pd.DataFrame):
        """Pour innocent3 : charge le buffer ETH (même taille que BTC)."""
        try:
            from data.loader import load_1s
            eth = load_1s('eth')
            n = len(buf)
            if len(eth) >= n:
                eth = eth.iloc[-n:].reset_index(drop=True)
            strategy.set_pair_data(eth)
            self._pair_data_buffers[name] = eth
            logger.info(f"[{name}] ETH startup buffer: {len(eth)} rows")
        except Exception as e:
            empty = pd.DataFrame()
            strategy.set_pair_data(empty)
            self._pair_data_buffers[name] = empty
            logger.warning(f"[{name}] ETH pair data non disponible ({e})")

    # ──────────────────────────────────────────────────────────────────────
    # Order execution
    # ──────────────────────────────────────────────────────────────────────

    def _execute_signal(self, name: str, signal: Signal,
                        coin: str, live_price: float):
        """Ouvre une position paper à partir d'un signal."""
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
            'qty':            qty,
            'size_usd':       sizing['position_usd'],
            'tp_price':       exec_price * (1 + d * tp_pct),
            'sl_price':       exec_price * (1 - d * sl_pct_adj),
            'fee_usd':        fee_usd,
            'slippage_usd':   slippage_usd,
            'opened_at':      datetime.now(),
            'unrealized_pnl': 0.0,
            'current_price':  live_price,
            'signal':         signal,
        }
        self._open_positions[name]   = position
        self.current_positions[name] = position

        logger.info(
            f"[{name}] OPENED {'LONG' if d==1 else 'SHORT'} {coin} "
            f"@ ${exec_price:.2f} | size=${sizing['position_usd']:.0f} "
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
