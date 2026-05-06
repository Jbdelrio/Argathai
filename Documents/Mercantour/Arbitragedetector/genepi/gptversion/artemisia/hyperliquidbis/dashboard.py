"""
╔══════════════════════════════════════════════════════════════╗
║             ARTEMISIA GLACIALIS v5 — CANDLE              ║
║     BB Squeeze + RSI (Overbought/Oversold) + VWAP (Bounce)                 ║
║     Hyperliquid Paper Trading Dashboard                      ║
╚══════════════════════════════════════════════════════════════╝
"""

import dash
from dash import dcc, html, no_update, callback_context
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timezone
import threading, time, logging, traceback

from config import Config
from ticker import HLPoller, CandleStore
from power_law import PowerLaw
from strategies import StrategyManager
from engine import PaperTrader

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)-14s] %(message)s")
logger = logging.getLogger("glacialis.ui")

# ═══════════════════════════════════════════════
# CYBORG PALETTE
# ═══════════════════════════════════════════════
C = {
    "bg": "#060606", "surface": "#0c0c0c", "card": "#101010",
    "border": "#1a1a1a", "border2": "#252525",
    "text": "#c8c8c8", "dim": "#555555", "bright": "#f0f0f0",
    "cyan": "#00e5ff", "green": "#00e676", "red": "#ff1744",
    "yellow": "#ffd600", "orange": "#ff9100", "purple": "#d500f9",
    "blue": "#2979ff", "teal": "#1de9b6", "pink": "#f50057",
}

PLOT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(8,8,8,0.9)",
    font=dict(family="JetBrains Mono, Consolas, monospace", size=10, color=C["dim"]),
    xaxis=dict(gridcolor="#141414", zerolinecolor="#141414"),
    yaxis=dict(gridcolor="#141414", zerolinecolor="#141414"),
    margin=dict(l=45, r=10, t=20, b=25),
    legend=dict(bgcolor="rgba(0,0,0,0.6)", font=dict(size=9)),
)

# ═══════════════════════════════════════════════
# STATE
# ═══════════════════════════════════════════════
S = {
    "cfg": Config(),
    "poller": None, "store": None, "pl": None, "strats": None, "trader": None,
    "phase": "idle",            # idle → warmup → live → killed
    "connected": False, "status": "Press ▶ START",
    "universe": [], "ctxs": {}, "btc_price": 0, "macro": {},
    "signals": [], "scan_count": 0, "candle_count": 0,
    "errors": [],
    "_thread": None, "_stop_event": threading.Event(),
}


def _init():
    cfg = S["cfg"]
    S["poller"] = HLPoller(cfg)
    S["pl"] = PowerLaw(cfg)
    S["strats"] = StrategyManager(cfg)
    S["trader"] = PaperTrader(cfg)
_init()


# ═══════════════════════════════════════════════
# CORE LOOP (runs in background thread)
# ═══════════════════════════════════════════════
def _tick_loop():
    """Background: poll every 2s, aggregate into 1-min candles, scan on close."""
    cfg = S["cfg"]
    poller = S["poller"]
    stop = S["_stop_event"]
    logger.info("Tick loop started")

    while not stop.is_set():
        try:
            prices = poller.poll_mids()
            if not prices:
                time.sleep(1)
                continue

            store = S["store"]
            if store is None:
                time.sleep(0.5)
                continue

            S["btc_price"] = prices.get("BTC", 0)

            # Feed tick → candle aggregator (returns True on candle close)
            candle_closed = store.add_tick(prices)
            S["candle_count"] = store.candle_num

            # Warmup
            if S["phase"] == "warmup":
                if store.is_warmed_up:
                    S["phase"] = "live"
                    S["status"] = "LIVE"
                    logger.info(f"Warmup done ({store.n_candles} candles) → LIVE")
                else:
                    S["status"] = f"Warmup: {store.n_candles}/{cfg.warmup_candles} candles"
                continue

            if S["phase"] != "live":
                continue

            # Power Law
            btc = prices.get("BTC", 0)
            if btc > 0 and cfg.power_law_enabled:
                S["macro"] = S["pl"].bias(btc)
            else:
                S["macro"] = {"bias": "neutral", "deviation": 0, "exposure": 1.0, "corridor": {}}

            exposure = S["macro"].get("exposure", 1.0)
            trader = S["trader"]
            current_prices = store.get_prices()

            # Check exits EVERY tick (stops/TP need real-time monitoring)
            trader.check_exits(current_prices, store.candle_num)
            trader.update_unrealized(current_prices)

            # Scan strategies ONLY on candle close (not every 2s)
            if candle_closed:
                # Get funding rates
                try:
                    _, ctxs = poller.get_meta_contexts()
                    S["ctxs"] = ctxs
                    funding = {s: ctxs.get(s, {}).get("funding", 0) for s in store.symbols}
                except:
                    funding = {}

                signals = S["strats"].scan_all(store, funding)
                S["signals"] = signals
                S["scan_count"] += 1

                for sig in signals:
                    p = current_prices.get(sig.symbol, 0)
                    if p > 0:
                        trader.open(sig, p, exposure)
                        if len(trader.positions) >= cfg.max_simultaneous:
                            break

                if S["scan_count"] % 30 == 0:
                    trader.save()

        except Exception as e:
            S["errors"].append({"ts": datetime.now(timezone.utc).isoformat(), "e": str(e)})
            logger.error(f"Loop error: {e}")
            time.sleep(1)


def _start():
    """Start the trading loop."""
    if S["_thread"] and S["_thread"].is_alive():
        return

    cfg = S["cfg"]

    # Get universe
    poller = S["poller"]
    ok, msg = poller.test()
    S["connected"] = ok
    if not ok:
        S["status"] = f"Connection failed: {msg}"
        return

    all_syms, ctxs = poller.get_meta_contexts()
    S["ctxs"] = ctxs

    # Filter by volume + preferred symbols
    if cfg.universe_filter:
        universe = [s for s in cfg.universe_filter if s in ctxs]
    elif cfg.preferred_symbols:
        # Use preferred list, filtered by volume
        universe = [s for s in cfg.preferred_symbols
                    if s in ctxs and ctxs[s].get("vol24h", 0) >= cfg.min_daily_volume]
    else:
        ranked = sorted(
            [(s, ctxs[s]["vol24h"]) for s in ctxs if ctxs[s].get("vol24h", 0) >= cfg.min_daily_volume],
            key=lambda x: x[1], reverse=True
        )
        universe = [s for s, _ in ranked[:cfg.max_universe_size]]

    if "BTC" not in universe:
        universe.insert(0, "BTC")

    S["universe"] = universe
    logger.info(f"Universe: {len(universe)} — {', '.join(universe[:10])}...")

    # Init tick store
    S["store"] = CandleStore(cfg, universe)
    S["phase"] = "warmup"
    S["status"] = "Warmup starting..."

    # Start background thread
    S["_stop_event"].clear()
    S["_thread"] = threading.Thread(target=_tick_loop, daemon=True)
    S["_thread"].start()


def _stop():
    S["_stop_event"].set()
    S["phase"] = "stopped"
    S["status"] = "STOPPED"
    if S["trader"]:
        S["trader"].save()


def _reset():
    _stop()
    time.sleep(0.5)
    S["phase"] = "idle"
    S["status"] = "Press ▶ START"
    S["scan_count"] = 0
    S["candle_count"] = 0
    S["signals"] = []
    if S["trader"]:
        S["trader"].reset()
    S["store"] = None


# ═══════════════════════════════════════════════
# DASH APP
# ═══════════════════════════════════════════════
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG],
                title="Artemisia Glacialis", update_title=None,
                suppress_callback_exceptions=True)

# ─── Helpers ─────────────────────────────────
def badge(label, vid, color=C["cyan"]):
    return html.Div([
        html.Div(label, style={"fontSize": "8px", "color": C["dim"],
                                "textTransform": "uppercase", "letterSpacing": "1.5px"}),
        html.Div(id=vid, style={"fontSize": "14px", "fontWeight": 700,
                                 "color": color, "marginTop": "1px"}),
    ], style={"display": "inline-block", "padding": "5px 10px", "margin": "2px",
              "backgroundColor": C["surface"], "borderRadius": "3px",
              "border": f"1px solid {C['border']}", "minWidth": "70px", "textAlign": "center"})


def pinput(label, id_, val, **kw):
    return html.Div([
        html.Label(label, style={"fontSize": "9px", "color": C["dim"], "display": "block", "marginBottom": "1px"}),
        dcc.Input(id=id_, value=val, type=kw.get("type", "number"),
                  step=kw.get("step"), min=kw.get("min_"), max=kw.get("max_"),
                  style={"backgroundColor": C["surface"], "color": C["text"],
                         "border": f"1px solid {C['border']}", "borderRadius": "3px",
                         "padding": "3px 6px", "width": "100%", "fontSize": "11px"}),
    ], style={"marginBottom": "4px"})


CARD = {"backgroundColor": C["card"], "border": f"1px solid {C['border']}",
        "borderRadius": "4px", "padding": "10px"}


# ─── Layout ──────────────────────────────────
app.layout = html.Div(style={
    "backgroundColor": C["bg"], "minHeight": "100vh", "padding": "10px 14px",
    "fontFamily": "JetBrains Mono, Consolas, monospace",
    "color": C["text"], "fontSize": "11px",
}, children=[
    dcc.Interval(id="tick-ui", interval=1200, n_intervals=0),
    dcc.Store(id="store-x"),

    # HEADER
    html.Div([
        html.Span("ARTEMISIA", style={
            "fontSize": "20px", "fontWeight": 800, "letterSpacing": "3px",
            "background": f"linear-gradient(90deg, {C['cyan']}, {C['teal']})",
            "WebkitBackgroundClip": "text", "WebkitTextFillColor": "transparent"}),
        html.Span(" GLACIALIS", style={
            "fontSize": "20px", "fontWeight": 300, "letterSpacing": "3px", "color": C["dim"]}),
        html.Span("  SUB-MINUTE MID-FREQ", style={
            "fontSize": "9px", "color": C["dim"], "marginLeft": "14px", "letterSpacing": "2px"}),
        html.Div([
            html.Div(id="phase-badge"),
            html.Span(id="conn-badge", style={"marginLeft": "8px"}),
        ], style={"float": "right", "marginTop": "3px"}),
    ], style={"borderBottom": f"1px solid {C['border']}", "paddingBottom": "6px", "marginBottom": "8px"}),

    # ── WARMUP BAR (full width, prominent) ──
    html.Div(id="warmup-container", style={"marginBottom": "8px"}),

    # STATS BAR
    html.Div([
        badge("BTC", "s-btc", C["yellow"]),
        badge("Bias", "s-bias", C["teal"]),
        badge("Exp", "s-exp", C["cyan"]),
        badge("Capital", "s-cap", C["bright"]),
        badge("P&L", "s-pnl", C["green"]),
        badge("Win%", "s-wr", C["green"]),
        badge("Sharpe", "s-sharpe", C["purple"]),
        badge("Sortino", "s-sortino", C["purple"]),
        badge("PF", "s-pf", C["teal"]),
        badge("MaxDD", "s-dd", C["red"]),
        badge("Trades", "s-trades", C["dim"]),
        badge("Open", "s-open", C["blue"]),
        badge("Sigs", "s-sigs", C["orange"]),
        badge("Fees", "s-fees", C["red"]),
        badge("AvgHold", "s-hold", C["dim"]),
        badge("Ticks", "s-ticks", C["dim"]),
    ], style={"marginBottom": "8px", "overflowX": "auto", "whiteSpace": "nowrap"}),

    # MAIN: 75% charts + 25% controls
    dbc.Row([
        dbc.Col([
            # Row 1
            dbc.Row([
                dbc.Col(html.Div([
                    html.Div("POWER LAW CORRIDOR", style={"fontSize": "9px", "color": C["dim"],
                             "letterSpacing": "1.5px", "marginBottom": "4px"}),
                    dcc.Graph(id="g-pl", style={"height": "240px"}, config={"displayModeBar": False}),
                ], style=CARD), width=6),
                dbc.Col(html.Div([
                    html.Div("SPREAD Z-SCORES (TICK)", style={"fontSize": "9px", "color": C["dim"],
                             "letterSpacing": "1.5px", "marginBottom": "4px"}),
                    dcc.Graph(id="g-hm", style={"height": "240px"}, config={"displayModeBar": False}),
                ], style=CARD), width=6),
            ], className="g-2 mb-2"),
            # Row 2
            dbc.Row([
                dbc.Col(html.Div([
                    html.Div("EQUITY CURVE", style={"fontSize": "9px", "color": C["dim"],
                             "letterSpacing": "1.5px", "marginBottom": "4px"}),
                    dcc.Graph(id="g-eq", style={"height": "200px"}, config={"displayModeBar": False}),
                ], style=CARD), width=6),
                dbc.Col(html.Div([
                    html.Div("LIVE SIGNALS", style={"fontSize": "9px", "color": C["dim"],
                             "letterSpacing": "1.5px", "marginBottom": "4px"}),
                    html.Div(id="tbl-sig", style={"maxHeight": "190px", "overflowY": "auto"}),
                ], style=CARD), width=6),
            ], className="g-2 mb-2"),
            # Row 3
            dbc.Row([
                dbc.Col(html.Div([
                    html.Div("POSITIONS", style={"fontSize": "9px", "color": C["dim"],
                             "letterSpacing": "1.5px", "marginBottom": "4px"}),
                    html.Div(id="tbl-pos", style={"maxHeight": "180px", "overflowY": "auto"}),
                ], style=CARD), width=4),
                dbc.Col(html.Div([
                    html.Div("TRADE LOG (timestamps)", style={"fontSize": "9px", "color": C["dim"],
                             "letterSpacing": "1.5px", "marginBottom": "4px"}),
                    html.Div(id="tbl-log", style={"maxHeight": "180px", "overflowY": "auto", "fontSize": "9px"}),
                ], style=CARD), width=5),
                dbc.Col(html.Div([
                    html.Div("STRATEGY MIX", style={"fontSize": "9px", "color": C["dim"],
                             "letterSpacing": "1.5px", "marginBottom": "4px"}),
                    html.Div(id="strat-mix"),
                ], style=CARD), width=3),
            ], className="g-2"),
        ], width=9),

        # ── RIGHT PANEL: CONTROLS ──
        dbc.Col(html.Div([
            html.Div("CONTROLS", style={"fontSize": "9px", "color": C["cyan"],
                                         "letterSpacing": "2px", "marginBottom": "8px", "fontWeight": 700}),
            html.Div([
                html.Button("▶ START", id="btn-go", n_clicks=0, style={
                    "backgroundColor": C["green"], "color": "#000", "border": "none",
                    "padding": "5px 14px", "borderRadius": "3px", "fontWeight": 700,
                    "fontSize": "10px", "cursor": "pointer", "marginRight": "4px", "fontFamily": "inherit"}),
                html.Button("■ STOP", id="btn-stop", n_clicks=0, style={
                    "backgroundColor": C["red"], "color": "#fff", "border": "none",
                    "padding": "5px 14px", "borderRadius": "3px", "fontWeight": 700,
                    "fontSize": "10px", "cursor": "pointer", "marginRight": "4px", "fontFamily": "inherit"}),
                html.Button("↺ RESET", id="btn-reset", n_clicks=0, style={
                    "backgroundColor": "#444", "color": "#000", "border": "none",
                    "padding": "5px 14px", "borderRadius": "3px", "fontWeight": 700,
                    "fontSize": "10px", "cursor": "pointer", "fontFamily": "inherit"}),
            ], style={"marginBottom": "10px"}),

            html.Hr(style={"borderColor": C["border"], "margin": "6px 0"}),
            html.Div("STRATEGY TOGGLES", style={"fontSize": "8px", "color": C["teal"], "letterSpacing": "1.5px", "marginBottom": "4px"}),
            dcc.Checklist(id="p-toggles",
                options=[
                    {"label": " EMA (EMA Crossover)", "value": "ema"},
                    {"label": " MB (RSI (Overbought/Oversold))", "value": "rsi"},
                    {"label": " VB (VWAP (Bounce))", "value": "vwap"},
                ],
                value=["ema", "rsi", "sp"],
                style={"fontSize": "10px", "color": C["text"]},
                inputStyle={"marginRight": "4px"},
                labelStyle={"display": "block", "marginBottom": "2px"},
            ),

            html.Hr(style={"borderColor": C["border"], "margin": "6px 0"}),
            html.Div("TIMING", style={"fontSize": "8px", "color": C["cyan"], "letterSpacing": "1.5px", "marginBottom": "4px"}),
            pinput("Poll (ms)", "p-tick", 2000, step=500, min_=1000),
            pinput("Warmup (candles)", "p-warmup", 30, step=5, min_=10),

            html.Hr(style={"borderColor": C["border"], "margin": "6px 0"}),
            html.Div("RISK & SIZING", style={"fontSize": "8px", "color": C["red"], "letterSpacing": "1.5px", "marginBottom": "4px"}),
            pinput("Capital $", "p-cap", 500, step=50, min_=50),
            pinput("Max Leverage", "p-lev", 8, step=1, min_=2),
            pinput("Max Positions", "p-maxpos", 3, step=1, min_=1),
            pinput("Stop Loss %", "p-stop", 0.8, step=0.1, min_=0.2),
            pinput("Take Profit %", "p-tp", 0.5, step=0.1, min_=0.1),
            pinput("Kill DD %", "p-kill", 10, step=1, min_=3),
            pinput("Min Edge (bps)", "p-edge", 15, step=1, min_=5),
            pinput("Cooldown (candles)", "p-cooldown", 2, step=1, min_=1),
            pinput("Loss CD (candles)", "p-cooldown-loss", 5, step=1, min_=1),

            html.Hr(style={"borderColor": C["border"], "margin": "6px 0"}),
            html.Button("APPLY CONFIG", id="btn-apply", n_clicks=0, style={
                "backgroundColor": C["cyan"], "color": "#000", "border": "none",
                "padding": "6px 0", "width": "100%", "borderRadius": "3px",
                "fontWeight": 700, "fontSize": "10px", "cursor": "pointer",
                "fontFamily": "inherit", "letterSpacing": "1px"}),
            html.Div(id="cfg-msg", style={"marginTop": "4px", "fontSize": "9px"}),

        ], style={**CARD, "overflowY": "auto", "maxHeight": "calc(100vh - 100px)"}), width=3),
    ], className="g-2"),
])


# ═══════════════════════════════════════════════
# CALLBACKS
# ═══════════════════════════════════════════════

@app.callback(Output("store-x", "data"),
              Input("btn-go", "n_clicks"), Input("btn-stop", "n_clicks"),
              Input("btn-reset", "n_clicks"), prevent_initial_call=True)
def handle_btns(go, stop, reset):
    btn = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else ""
    if btn == "btn-go" and S["phase"] in ("idle", "stopped"):
        _start()
    elif btn == "btn-stop":
        _stop()
    elif btn == "btn-reset":
        _reset()
    return no_update


@app.callback(Output("cfg-msg", "children"),
              Input("btn-apply", "n_clicks"),
              [State("p-toggles", "value"),
               State("p-tick", "value"), State("p-warmup", "value"),
               State("p-cap", "value"), State("p-lev", "value"),
               State("p-maxpos", "value"), State("p-stop", "value"),
               State("p-tp", "value"),
               State("p-kill", "value"), State("p-edge", "value"),
               State("p-cooldown", "value"), State("p-cooldown-loss", "value")],
              prevent_initial_call=True)
def apply_cfg(n, toggles, tick, warmup,
              cap, lev, maxpos, stop, tp, kill, edge,
              cooldown, cooldown_loss):
    try:
        cfg = S["cfg"]
        toggles = toggles or []
        cfg.enable_ema = "ema" in toggles
        cfg.enable_rsi = "rsi" in toggles
        cfg.enable_vwap = "vwap" in toggles
        cfg.tick_interval_ms = int(tick or 2000)
        cfg.warmup_candles = int(warmup or 30)
        cfg.initial_capital = float(cap or 500)
        cfg.max_leverage = float(lev or 8)
        cfg.max_simultaneous = int(maxpos or 3)
        cfg.stop_pct = float(stop or 0.8) / 100
        cfg.tp_pct = float(tp or 0.5) / 100
        cfg.max_drawdown_kill = float(kill or 10) / 100
        cfg.min_edge_bps = float(edge or 15)
        cfg.cooldown_candles = int(cooldown or 2)
        cfg.cooldown_after_loss = int(cooldown_loss or 5)

        S["strats"] = StrategyManager(cfg)
        if S["poller"]:
            S["poller"].cfg = cfg
            S["poller"]._delay = cfg.tick_interval_ms / 1000
        cfg.save()
        en = ("EMA" if cfg.enable_ema else "") + ("+RSI" if cfg.enable_rsi else "") + ("+VWAP" if cfg.enable_vwap else "")
        return html.Span(f"✓ [{en}] lev={cfg.max_leverage}x stop={cfg.stop_pct*100:.1f}%", style={"color": C["green"]})
    except Exception as e:
        return html.Span(f"✗ {e}", style={"color": C["red"]})


# ── Warmup bar ───────────────────────────────
@app.callback(Output("warmup-container", "children"), Input("tick-ui", "n_intervals"))
def warmup_bar(n):
    store = S.get("store")
    phase = S["phase"]
    if phase not in ("warmup",) or store is None:
        if phase == "live":
            return html.Div("● LIVE", style={
                "textAlign": "center", "padding": "4px",
                "backgroundColor": "#002a10", "borderRadius": "4px",
                "border": f"1px solid {C['green']}33", "color": C["green"],
                "fontSize": "11px", "fontWeight": 700, "letterSpacing": "2px"})
        return ""

    pct = store.warmup_pct
    cur = store.n_candles
    target = S["cfg"].warmup_candles
    elapsed = cur * S["cfg"].tick_interval_ms / 1000
    eta = (target - cur) * S["cfg"].tick_interval_ms / 1000

    return html.Div([
        html.Div([
            html.Span(f"WARMING UP  ", style={"color": C["yellow"], "fontWeight": 700}),
            html.Span(f"{cur}/{target} ticks  •  {pct:.0f}%  •  ",
                      style={"color": C["dim"]}),
            html.Span(f"~{eta:.0f}s remaining", style={"color": C["yellow"]}),
        ], style={"fontSize": "10px", "marginBottom": "4px"}),
        html.Div(style={
            "height": "6px", "backgroundColor": C["border"], "borderRadius": "3px",
            "overflow": "hidden",
        }, children=[
            html.Div(style={
                "height": "100%", "width": f"{pct}%",
                "background": f"linear-gradient(90deg, {C['yellow']}, {C['orange']})",
                "borderRadius": "3px", "transition": "width 0.3s ease",
            })
        ]),
    ], style={
        "padding": "8px 12px", "backgroundColor": "#1a1500",
        "border": f"1px solid {C['yellow']}33", "borderRadius": "4px",
    })


# ── Badges ───────────────────────────────────
@app.callback(
    Output("phase-badge", "children"), Output("conn-badge", "children"),
    Input("tick-ui", "n_intervals"))
def badges(n):
    phase = S["phase"]
    pmap = {"idle": (C["dim"], "IDLE"), "warmup": (C["yellow"], "WARMUP"),
            "live": (C["green"], "LIVE"), "stopped": (C["red"], "STOPPED"),
            "killed": ("#7b0000", "KILLED")}
    col, txt = pmap.get(phase, (C["dim"], "?"))
    pb = html.Span(txt, style={"padding": "3px 10px", "borderRadius": "3px",
                                "backgroundColor": col, "color": "#000" if col != C["dim"] else C["text"],
                                "fontSize": "10px", "fontWeight": 700})
    conn = "●" if S["connected"] else "○"
    cc = C["green"] if S["connected"] else C["red"]
    nu = len(S["universe"])
    cb = html.Span(f"{conn} {nu} perps", style={"color": cc, "fontSize": "10px"})
    return pb, cb


# ── Stats ────────────────────────────────────
@app.callback(
    [Output("s-btc", "children"), Output("s-bias", "children"),
     Output("s-exp", "children"), Output("s-cap", "children"),
     Output("s-pnl", "children"), Output("s-pnl", "style"),
     Output("s-wr", "children"), Output("s-wr", "style"),
     Output("s-sharpe", "children"), Output("s-sortino", "children"),
     Output("s-pf", "children"),
     Output("s-dd", "children"), Output("s-trades", "children"),
     Output("s-open", "children"), Output("s-sigs", "children"),
     Output("s-fees", "children"), Output("s-hold", "children"),
     Output("s-ticks", "children")],
    Input("tick-ui", "n_intervals"))
def stats(n):
    tr = S["trader"]
    m = tr.get_metrics() if tr else {}
    macro = S.get("macro", {})
    btc = S.get("btc_price", 0)
    base = {"fontSize": "14px", "fontWeight": 700, "marginTop": "1px"}
    pnl = m.get("total_pnl", 0)
    pnl_s = {**base, "color": C["green"] if pnl >= 0 else C["red"]}
    wr = m.get("win_rate", 0)
    wr_s = {**base, "color": C["green"] if wr >= 50 else C["red"]}

    hold = m.get("avg_hold_sec", 0)
    hs = f"{hold:.0f}s" if hold < 60 else f"{hold/60:.1f}m"

    return (
        f"${btc:,.0f}", macro.get("bias", "—").replace("_", " ").upper(),
        f"{macro.get('exposure', 1):.1f}x", f"${tr.capital:.2f}" if tr else "$500",
        f"${pnl:+.3f} ({m.get('total_pnl_pct', 0):+.1f}%)", pnl_s,
        f"{wr:.0f}%", wr_s,
        f"{m.get('sharpe', 0):.2f}", f"{m.get('sortino', 0):.2f}",
        f"{m.get('profit_factor', 0):.2f}",
        f"{m.get('max_dd_pct', 0):.1f}%", str(m.get("trades", 0)),
        str(len(tr.positions) if tr else 0),
        str(len(S.get("signals", []))),
        f"${m.get('total_fees', 0):.3f}", hs,
        str(S.get("candle_count", 0)),
    )


# ── Charts ───────────────────────────────────
@app.callback(Output("g-pl", "figure"), Input("tick-ui", "n_intervals"))
def chart_pl(n):
    fig = go.Figure()
    pl = S["pl"]
    if pl:
        c = pl.corridor_history(2024)
        fig.add_trace(go.Scatter(x=c["date"], y=c["upper"], mode="lines",
                                 line=dict(width=1, color="rgba(255,23,68,0.25)"), showlegend=False))
        fig.add_trace(go.Scatter(x=c["date"], y=c["lower"], mode="lines",
                                 line=dict(width=1, color="rgba(0,230,118,0.25)"),
                                 fill="tonexty", fillcolor="rgba(0,229,255,0.02)", name="Corridor"))
        fig.add_trace(go.Scatter(x=c["date"], y=c["median"], mode="lines",
                                 line=dict(width=2, color=C["cyan"], dash="dot"), name="Median"))
    btc = S.get("btc_price", 0)
    if btc > 0:
        fig.add_trace(go.Scatter(x=[datetime.now(timezone.utc)], y=[btc], mode="markers+text",
                                 marker=dict(size=9, color=C["yellow"], symbol="diamond"),
                                 text=[f"${btc:,.0f}"], textposition="top center",
                                 textfont=dict(color=C["yellow"], size=9), showlegend=False))
    fig.update_layout(**PLOT, yaxis_type="log")
    return fig


@app.callback(Output("g-hm", "figure"), Input("tick-ui", "n_intervals"))
def chart_hm(n):
    fig = go.Figure()
    store = S.get("store")
    if not store or not store.indicators:
        fig.add_annotation(text="Awaiting ticks...", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color=C["dim"]))
        fig.update_layout(**PLOT)
        return fig

    items = [(s, ind.get("btc_ratio_z", 0)) for s, ind in store.indicators.items()
             if s != "BTC" and np.isfinite(ind.get("btc_ratio_z", 0))]
    items.sort(key=lambda x: x[1])
    if not items:
        fig.update_layout(**PLOT)
        return fig

    syms = [x[0] for x in items]
    zs = [x[1] for x in items]
    colors = [C["green"] if z < -2 else C["red"] if z > 2
              else C["yellow"] if abs(z) > 1.5 else "#222" for z in zs]

    fig.add_trace(go.Bar(y=syms, x=zs, orientation="h", marker_color=colors,
                         text=[f"{z:.1f}σ" for z in zs], textposition="outside",
                         textfont=dict(size=8, color=C["dim"])))
    fig.add_vline(x=-2, line_dash="dash", line_color=C["green"], line_width=1, opacity=0.4)
    fig.add_vline(x=2, line_dash="dash", line_color=C["red"], line_width=1, opacity=0.4)
    fig.update_layout(**PLOT, xaxis_title="Z-Score")
    return fig


@app.callback(Output("g-eq", "figure"), Input("tick-ui", "n_intervals"))
def chart_eq(n):
    fig = go.Figure()
    tr = S["trader"]
    if tr and tr.equity_curve:
        ts = [e["timestamp"] if isinstance(e["timestamp"], datetime) else
              datetime.fromisoformat(e["timestamp"]) for e in tr.equity_curve]
        vs = [e["equity"] for e in tr.equity_curve]
        fig.add_trace(go.Scatter(x=ts, y=vs, mode="lines",
                                 line=dict(color=C["cyan"], width=2),
                                 fill="tozeroy", fillcolor="rgba(0,229,255,0.04)"))
    cap = S["cfg"].initial_capital
    fig.add_hline(y=cap, line_dash="dash", line_color=C["dim"], line_width=1)
    fig.update_layout(**PLOT, yaxis_title="$")
    return fig


# ── Tables ───────────────────────────────────
@app.callback(Output("tbl-sig", "children"), Input("tick-ui", "n_intervals"))
def tbl_signals(n):
    sigs = S.get("signals", [])
    if not sigs:
        return html.Div("—", style={"color": C["dim"], "textAlign": "center", "padding": "20px"})
    rows = []
    for s in sigs[:15]:
        sc = C["green"] if s.side == "long" else C["red"]
        hc = C["green"] if s.hurst < 0.45 else C["orange"] if s.hurst < 0.55 else C["purple"]
        rows.append(html.Tr([
            html.Td(s.timestamp.strftime("%H:%M:%S"), style={"color": C["dim"], "fontSize": "9px"}),
            html.Td(s.symbol, style={"fontWeight": 700}),
            html.Td(s.side.upper()[:1], style={"color": sc, "fontWeight": 700}),
            html.Td(s.strategy, style={"fontSize": "9px", "color": C["teal"]}),
            html.Td(f"{s.edge_bps:.0f}", style={"color": C["green"] if s.edge_bps > 15 else C["dim"]}),
            html.Td(f"{s.hurst:.2f}", style={"fontSize": "9px", "color": hc}),
            html.Td(f"{s.half_life:.0f}", style={"fontSize": "9px", "color": C["dim"]}),
        ], style={"borderBottom": f"1px solid {C['border']}"}))
    return html.Table([
        html.Thead(html.Tr([html.Th(h, style={"padding": "2px 5px", "fontSize": "8px",
                   "color": C["dim"], "textTransform": "uppercase"})
                   for h in ["Time", "Sym", "", "Strat", "Edge", "H", "HL"]])),
        html.Tbody(rows)
    ], style={"width": "100%", "borderCollapse": "collapse", "fontSize": "10px"})


@app.callback(Output("tbl-pos", "children"), Input("tick-ui", "n_intervals"))
def tbl_positions(n):
    tr = S["trader"]
    if not tr or not tr.positions:
        return html.Div("—", style={"color": C["dim"], "textAlign": "center", "padding": "20px"})
    rows = []
    for pos in tr.positions.values():
        pc = C["green"] if pos.pnl >= 0 else C["red"]
        sc = C["green"] if pos.side == "long" else C["red"]
        ht = pos.hold_ticks if hasattr(pos, "hold_ticks") else 0
        rows.append(html.Tr([
            html.Td(pos.symbol, style={"fontWeight": 700}),
            html.Td(pos.side[0].upper(), style={"color": sc}),
            html.Td(pos.strategy, style={"fontSize": "9px", "color": C["teal"]}),
            html.Td(f"${pos.pnl:+.4f}", style={"color": pc, "fontWeight": 700}),
            html.Td(f"{pos.pnl_pct:+.1f}%", style={"color": pc, "fontSize": "9px"}),
            html.Td(f"{ht}", style={"fontSize": "9px", "color": C["dim"]}),
        ], style={"borderBottom": f"1px solid {C['border']}"}))
    return html.Table([
        html.Thead(html.Tr([html.Th(h, style={"padding": "2px 4px", "fontSize": "8px",
                   "color": C["dim"], "textTransform": "uppercase"})
                   for h in ["Sym", "", "Strat", "P&L", "%", "Ticks"]])),
        html.Tbody(rows)
    ], style={"width": "100%", "borderCollapse": "collapse", "fontSize": "10px"})


@app.callback(Output("tbl-log", "children"), Input("tick-ui", "n_intervals"))
def tbl_log(n):
    tr = S["trader"]
    if not tr or not tr.trade_log:
        return html.Div("—", style={"color": C["dim"], "textAlign": "center", "padding": "20px"})
    logs = list(reversed(tr.trade_log[-S["cfg"].max_log_display:]))
    items = []
    for t in logs:
        a = t.get("action")
        pnl = t.get("pnl", 0)
        if a == "OPEN":
            col = C["blue"]
            txt = (f"[{t['ts']}] → {t['side']} {t['symbol']} "
                   f"@${t['price']:.4f} {t['lev']}x {t['strat']} "
                   f"edge:{t.get('edge_bps', 0):.0f}bp H={t.get('hurst',0):.2f}")
        else:
            col = C["green"] if pnl > 0 else C["red"]
            txt = (f"[{t['ts']}] ← {t['side']} {t['symbol']} "
                   f"@${t['price']:.4f} P&L:${pnl:+.5f} "
                   f"({t.get('pnl_pct', 0):+.2f}%) {t.get('exit_reason', '')} "
                   f"{t.get('hold_sec', 0):.1f}s")
        items.append(html.Div(txt, style={"padding": "1px 0", "color": col,
                                           "borderBottom": f"1px solid {C['border']}",
                                           "lineHeight": "1.3"}))
    return items


@app.callback(Output("strat-mix", "children"), Input("tick-ui", "n_intervals"))
def strat_mix(n):
    tr = S["trader"]
    if not tr:
        return ""
    m = tr.get_metrics()
    by_s = m.get("by_strategy", {})
    items = []
    colors = {"EMA": C["orange"], "RSI": C["purple"], "VWAP": C["teal"]}
    names = {"EMA": "EMA Cross", "RSI": "RSI Scalp", "VWAP": "VWAP Bounce"}
    for strat in ("EMA", "RSI", "VWAP"):
        st = by_s.get(strat, {})
        nt = st.get("trades", 0)
        wr = st.get("win_rate", 0)
        pnl = st.get("total_pnl", 0)
        col = colors[strat]
        items.append(html.Div([
            html.Div(names[strat], style={"fontSize": "9px", "color": col, "fontWeight": 700}),
            html.Div(f"{nt} trades", style={"fontSize": "10px"}),
            html.Div(f"WR: {wr:.0f}%", style={"fontSize": "10px",
                      "color": C["green"] if wr >= 50 else C["red"]}),
            html.Div(f"P&L: ${pnl:+.4f}", style={"fontSize": "10px",
                      "color": C["green"] if pnl >= 0 else C["red"]}),
        ], style={"padding": "6px", "marginBottom": "4px",
                  "backgroundColor": C["surface"], "borderRadius": "3px",
                  "border": f"1px solid {C['border']}"}))
    return items


# ═══════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════
def main():
    cfg = S["cfg"]
    print()
    print("╔═══════════════════════════════════════════════════════╗")
    print("║         ARTEMISIA GLACIALIS v4                        ║")
    print("║   Sub-Minute Mid-Frequency | Tick-Based Trading       ║")
    print("╠═══════════════════════════════════════════════════════╣")

    poller = S["poller"]
    ok, msg = poller.test()
    s = "✓" if ok else "✗"
    print(f"║  API: {s} {msg:<46} ║")

    if ok:
        _, ctxs = poller.get_meta_contexts()
        btc = ctxs.get("BTC", {}).get("mid", 0)
        if btc > 0:
            b = S["pl"].bias(btc)
            print(f"║  BTC: ${btc:>9,.0f}  |  Bias: {b['bias']:<20}   ║")

    rt = cfg.roundtrip_cost_bps()
    print(f"║  Poll: {cfg.tick_interval_ms}ms  |  Warmup: {cfg.warmup_candles} candles      ║")
    print(f"║  Capital: ${cfg.initial_capital:,.0f}  |  Lev: {cfg.max_leverage}x  |  Fee RT: {rt:.1f}bp    ║")
    print(f"║  Stop: {cfg.stop_pct*100:.1f}%  TP: {cfg.tp_pct*100:.1f}%  Edge>{cfg.min_edge_bps}bp  Lev:{cfg.max_leverage}x  ║")
    print("╠═══════════════════════════════════════════════════════╣")
    print(f"║  Dashboard: http://127.0.0.1:{cfg.dash_port}                   ║")
    print("║  Press ▶ START in GUI → warmup → live trading         ║")
    print("╚═══════════════════════════════════════════════════════╝")
    print()

    app.run(debug=False, port=cfg.dash_port, host="0.0.0.0")


if __name__ == "__main__":
    main()
