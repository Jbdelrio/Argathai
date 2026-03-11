"""
Agarthai — Live Trading Dashboard (Dash + Bootstrap Cyborg)
============================================================
python live/dashboard.py  →  http://localhost:8055

Layout:
  - Sidebar: network, coins, config, per-strategy controls
  - Main:    tabs (Marché Live | Signaux | Portfolio | Config)
"""

import sys
import json
import time
import urllib.request
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, ctx, dcc, html, dash_table

sys.path.insert(0, str(Path(__file__).parent.parent))

# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        "https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;500&display=swap",
    ],
    title="Agarthai — Live",
    suppress_callback_exceptions=True,
)

# ── Extra CSS on top of Cyborg ────────────────────────────────────────────────
app.index_string = """
<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
      /* ── Fonts ─────────────────────────────────────────── */
      body, .dash-table-container * {
        font-family: 'Roboto Mono', 'Courier New', monospace !important;
      }

      /* ── Sidebar ───────────────────────────────────────── */
      #sidebar {
        min-height: 100vh;
        background: #111418;
        border-right: 1px solid #2a9fd620;
        padding: 0 !important;
      }
      .sidebar-logo {
        background: #0c1014;
        padding: 18px 16px 12px;
        border-bottom: 1px solid #2a9fd620;
      }
      .sidebar-section {
        padding: 10px 16px;
        border-bottom: 1px solid #1a2530;
        font-size: .75rem;
      }
      .sidebar-section-label {
        color: #2a9fd6;
        letter-spacing: .12em;
        text-transform: uppercase;
        font-size: .65rem;
        margin-bottom: 8px;
        font-weight: 500;
      }

      /* ── Strategy row in sidebar ────────────────────────── */
      .strat-row {
        display: flex; align-items: center; gap: 6px;
        padding: 5px 0;
      }
      .strat-row .strat-name {
        flex: 1; font-size: .75rem; color: #ccc;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
      }
      .strat-btn { padding: 2px 8px !important; font-size: .65rem !important; }

      /* ── Main header ────────────────────────────────────── */
      #main-header {
        background: #0c1014;
        padding: 10px 20px;
        border-bottom: 1px solid #2a9fd620;
        display: flex; align-items: center; gap: 12px;
      }

      /* ── Tabs ───────────────────────────────────────────── */
      .nav-tabs .nav-link {
        font-size: .78rem; letter-spacing: .06em;
        text-transform: uppercase; color: #888 !important;
        border: none !important; border-bottom: 2px solid transparent !important;
        padding: 8px 16px;
      }
      .nav-tabs .nav-link.active {
        color: #2a9fd6 !important;
        border-bottom-color: #2a9fd6 !important;
        background: transparent !important;
      }
      .nav-tabs { border-bottom: 1px solid #2a9fd620 !important; }

      /* ── Metric cards ───────────────────────────────────── */
      .metric-card {
        background: #111418; border: 1px solid #2a9fd620;
        border-radius: 4px; padding: 12px 16px;
      }
      .metric-card .metric-label {
        font-size: .65rem; color: #2a9fd6;
        letter-spacing: .12em; text-transform: uppercase;
        margin-bottom: 4px;
      }
      .metric-card .metric-value {
        font-size: 1.35rem; font-weight: 500; color: #eee;
      }
      .metric-card .metric-sub { font-size: .7rem; color: #555; margin-top: 2px; }

      /* ── Strategy table ─────────────────────────────────── */
      .strat-table-row {
        display: flex; align-items: center; gap: 0;
        padding: 10px 12px; border-bottom: 1px solid #1a2530;
        font-size: .78rem; color: #bbb;
      }
      .strat-table-row:hover { background: #151c24; }
      .strat-table-header { color: #2a9fd6; font-size: .65rem;
        letter-spacing: .1em; text-transform: uppercase; }
      .strat-table-row .col-idx { width: 32px; color: #555; }
      .strat-table-row .col-name { flex: 3; font-weight: 500; color: #ddd; }
      .strat-table-row .col-type { flex: 2; }
      .strat-table-row .col-turnover { flex: 2; }
      .strat-table-row .col-warmup { flex: 3; }
      .strat-table-row .col-status { flex: 1; text-align: center; }

      /* ── Status pill ─────────────────────────────────────── */
      .pill-active { background: #77b300; color: #000; border-radius: 10px;
        padding: 2px 10px; font-size: .65rem; font-weight: 500; }
      .pill-idle   { background: #333; color: #888; border-radius: 10px;
        padding: 2px 10px; font-size: .65rem; }

      /* ── Warmup bar ──────────────────────────────────────── */
      .wu-bg { background: #1a2530; height: 4px; border-radius: 2px; }
      .wu-fg { background: #2a9fd6; height: 4px; border-radius: 2px;
        transition: width .4s ease; }
      .wu-fg-done { background: #77b300 !important; }

      /* ── Warmup done badge ───────────────────────────────── */
      .warmup-done-badge {
        display: inline-block; background: #77b300; color: #000;
        border-radius: 10px; padding: 1px 8px;
        font-size: .6rem; font-weight: 600; margin-left: 6px;
        letter-spacing: .05em; text-transform: uppercase;
      }

      /* ── Per-strategy live section (post-warmup) ─────────── */
      .strat-live-section {
        background: #080d12;
        padding: 12px 16px 14px 44px;
        border-top: 1px solid #2a9fd620;
      }
      .risk-metrics-row {
        display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap;
      }
      .risk-metric-item {
        background: #111418; border: 1px solid #2a9fd615;
        border-left: 2px solid #2a9fd6; border-radius: 3px;
        padding: 6px 12px; flex: 1; min-width: 100px;
      }
      .risk-metric-item .rmi-label {
        font-size: .6rem; color: #2a9fd6;
        letter-spacing: .1em; text-transform: uppercase; margin-bottom: 2px;
      }
      .risk-metric-item .rmi-value { font-size: .95rem; font-weight: 500; }

      /* ── Charts ─────────────────────────────────────────── */
      .js-plotly-plot .plotly { border-radius: 4px; }

      /* ── Live metrics bar (below each strategy row) ─────── */
      .live-metrics-bar {
        background: #0c1014; padding: 5px 44px 7px;
        border-bottom: 1px solid #1a2530;
        font-size: .7rem; color: #555;
        display: flex; gap: 20px; flex-wrap: wrap;
      }
      .live-metrics-bar .lm-item { display: flex; gap: 5px; align-items: center; }
      .live-metrics-bar .lm-key  { color: #2a9fd640; text-transform: uppercase;
                                    font-size: .6rem; letter-spacing: .08em; }
      .live-metrics-bar .lm-val  { color: #888; }
      .live-metrics-bar .lm-val-pos { color: #77b300; }
      .live-metrics-bar .lm-val-neg { color: #cc0000; }
      .live-metrics-bar .lm-val-blue { color: #2a9fd6; }

      /* ── Per-strategy cards in portfolio ─────────────────── */
      .strat-pnl-card {
        background: #111418; border: 1px solid #2a9fd620;
        border-radius: 4px; padding: 10px 14px; flex: 1; min-width: 160px;
      }
      .strat-pnl-card .spc-name { font-size: .65rem; color: #2a9fd6;
        letter-spacing: .1em; text-transform: uppercase; margin-bottom: 4px; }
      .strat-pnl-card .spc-pnl  { font-size: 1.15rem; font-weight: 500; }
      .strat-pnl-card .spc-sub  { font-size: .65rem; color: #555; margin-top: 2px; }

      /* ── Warmup spinner ────────────────────────────────── */
      @keyframes spin {
        0%   { transform: rotate(0deg);   }
        100% { transform: rotate(360deg); }
      }
      .warmup-spinner {
        display: inline-block;
        width: 12px; height: 12px;
        border: 2px solid #2a9fd640;
        border-top: 2px solid #2a9fd6;
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
        vertical-align: middle;
        margin-right: 6px;
      }
      .warmup-spinner-done {
        display: inline-block;
        width: 12px; height: 12px;
        border: 2px solid #77b300;
        border-radius: 50%;
        vertical-align: middle;
        margin-right: 6px;
        background: #77b300;
      }
      .min-data-tag {
        display: inline-block; background: #1a2530;
        color: #2a9fd680; border-radius: 3px;
        padding: 1px 6px; font-size: .55rem;
        margin-left: 6px; letter-spacing: .03em;
      }

      /* ── Misc ───────────────────────────────────────────── */
      .text-price { font-size: 1.05rem; color: #2a9fd6; }
      .text-muted  { color: #555 !important; font-size: .7rem; }
      .badge-conn-ok  { background: #77b300; color: #000; }
      .badge-conn-off { background: #555; color: #aaa; }
      .btn-sidebar { font-size: .65rem !important; padding: 2px 8px !important; line-height: 1.4; }
      .separator { border-top: 1px solid #1a2530; margin: 4px 0; }
    </style>
  </head>
  <body>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
  </body>
</html>
"""

# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────
try:
    from live.engine import LiveEngine
    engine = LiveEngine()
except Exception as e:
    engine = None
    print(f"[dashboard] Warning: LiveEngine not available — {e}")

STRATEGY_META = {
    "baudouin4":   {"label": "Baudouin 4",   "type": "Mean-Rev",       "turnover": "Moyen",
                    "min_data": "30min calib + lookback",  "min_data_sec": 2700},
    "innocent3":   {"label": "Innocent 3",   "type": "Stat-Arb",       "turnover": "Faible",
                    "min_data": "30min coint + OU window", "min_data_sec": 1800},
    "urbain2":     {"label": "Urbain 2",     "type": "Res. Momentum",  "turnover": "Faible",
                    "min_data": "30min regime window",     "min_data_sec": 1800},
    "staugustin":  {"label": "Staugustin",   "type": "Liq. Release",   "turnover": "Moyen",
                    "min_data": "5h18 (318 bars x 60s)",   "min_data_sec": 19080},
    "childeric1":  {"label": "Childeric 1",  "type": "Resid. Fade",    "turnover": "Moyen",
                    "min_data": "1h (MAD z-score window)",  "min_data_sec": 3600},
}

STRAT_COLORS = {
    "baudouin4":  "#2a9fd6",
    "innocent3":  "#77b300",
    "urbain2":    "#ff7518",
    "staugustin": "#d500f9",
    "childeric1": "#e91e63",
}

EXCHANGES = ["Hyperliquid", "Bitget Futures"]
DEFAULT_COINS = {
    "Hyperliquid": ["BTC", "ETH", "SOL", "DOGE", "ARB", "OP", "AVAX", "LINK"],
    "Bitget Futures": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"],
}
TIMEFRAMES = ["1s", "5s", "10s", "15s", "30s", "1m", "5m", "10m"]

# Map timeframe label → seconds
_TF_TO_SEC = {
    "1s": 1, "5s": 5, "10s": 10, "15s": 15, "30s": 30,
    "1m": 60, "5m": 300, "10m": 600,
}

strategy_names = list(engine.get_strategy_names()) if engine else []


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_price(symbol: str) -> tuple:
    """Returns (price, source)."""
    try:
        from exchanges.clients import _fetch_binance_price, _FALLBACK_PRICES
        p = _fetch_binance_price(symbol)
        if p:
            return p, "Binance REST"
        return _FALLBACK_PRICES.get(symbol, 73000.0), "static"
    except Exception:
        return 73000.0, "static"


# ── Hyperliquid universe ──────────────────────────────────────────────────────
_hl_universe_cache: dict = {}   # {ts: float, coins: list[dict]}

def _fetch_hl_universe(max_age_sec: int = 30) -> list[dict]:
    """
    Fetch full Hyperliquid perp universe via POST /info (metaAndAssetCtxs).
    Returns list of dicts with: name, markPx, prevDayPx, dayNtlVlm,
    openInterest, funding, premium, maxLeverage, chg24h_pct.
    Cached for max_age_sec seconds.
    """
    now = time.time()
    if _hl_universe_cache.get('ts', 0) > now - max_age_sec:
        return _hl_universe_cache.get('coins', [])

    try:
        payload = json.dumps({"type": "metaAndAssetCtxs"}).encode()
        req = urllib.request.Request(
            "https://api.hyperliquid.xyz/info",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read())

        meta_list   = data[0].get("universe", [])   # [{name, szDecimals, maxLeverage, ...}]
        ctx_list    = data[1]                         # [{markPx, dayNtlVlm, openInterest, ...}]

        coins = []
        for i, meta in enumerate(meta_list):
            ctx = ctx_list[i] if i < len(ctx_list) else {}
            try:
                mark   = float(ctx.get("markPx") or 0)
                prev   = float(ctx.get("prevDayPx") or mark or 1)
                vol    = float(ctx.get("dayNtlVlm") or 0)
                oi     = float(ctx.get("openInterest") or 0)
                fund   = float(ctx.get("funding") or 0)
                chg    = (mark / prev - 1) * 100 if prev else 0
                coins.append({
                    "name":        meta.get("name", "?"),
                    "maxLeverage": int(meta.get("maxLeverage", 0)),
                    "markPx":      mark,
                    "chg24h_pct":  round(chg, 2),
                    "dayNtlVlm":   vol,
                    "openInterest_usd": oi * mark,
                    "funding_8h":  round(fund * 100, 5),  # % per 8h
                    "funding_ann": round(fund * 100 * 3 * 365, 2),  # annualized %
                })
            except Exception:
                continue

        # Sort by 24h volume descending
        coins.sort(key=lambda x: x["dayNtlVlm"], reverse=True)

        _hl_universe_cache['ts']    = now
        _hl_universe_cache['coins'] = coins
        return coins

    except Exception as e:
        print(f"[dashboard] HL universe fetch failed: {e}")
        return _hl_universe_cache.get('coins', [])


def _live_metrics_placeholder():
    """Default content for the live metrics bar (before first refresh)."""
    return [
        html.Div([html.Span("TICKS", className="lm-key"), html.Span("—", className="lm-val")],
                 className="lm-item"),
        html.Div([html.Span("PRIX",  className="lm-key"), html.Span("—", className="lm-val")],
                 className="lm-item"),
        html.Div([html.Span("ÉTAT",  className="lm-key"), html.Span("IDLE", className="lm-val")],
                 className="lm-item"),
        html.Div([html.Span("SIGNAUX", className="lm-key"), html.Span("0", className="lm-val")],
                 className="lm-item"),
        html.Div([html.Span("P&L OUVERT", className="lm-key"),
                  html.Span("$0.00", className="lm-val")], className="lm-item"),
    ]


def _build_live_metrics_bar(rt: dict) -> list:
    """Build live metrics bar content from runtime dict."""
    ticks   = rt.get('ticks', 0)
    price   = rt.get('last_price')
    state   = rt.get('state', 'IDLE')
    signals = rt.get('n_signals_today', 0)
    unreal  = rt.get('unrealized_pnl', 0.0)
    n_scans = rt.get('n_scans', 0)
    last_scan = rt.get('last_scan_at', '—')

    price_str  = f"${price:,.2f}" if price else "—"
    unreal_str = f"${unreal:+.2f}" if unreal != 0 else "$0.00"
    unreal_cls = ("lm-val-pos" if unreal > 0 else "lm-val-neg") if unreal != 0 else "lm-val"
    state_cls  = ("lm-val-blue" if state == "IMPULSE"
                  else "lm-val-pos" if state == "STAB" else "lm-val")

    return [
        html.Div([html.Span("TICKS",      className="lm-key"),
                  html.Span(f"{ticks:,}", className="lm-val lm-val-blue")],
                 className="lm-item"),
        html.Div([html.Span("PRIX",       className="lm-key"),
                  html.Span(price_str,    className="lm-val")],
                 className="lm-item"),
        html.Div([html.Span("ÉTAT",       className="lm-key"),
                  html.Span(state,        className=f"lm-val {state_cls}")],
                 className="lm-item"),
        html.Div([html.Span("SCANS",      className="lm-key"),
                  html.Span(f"{n_scans} @ {last_scan}", className="lm-val")],
                 className="lm-item"),
        html.Div([html.Span("SIGNAUX",    className="lm-key"),
                  html.Span(str(signals), className="lm-val")],
                 className="lm-item"),
        html.Div([html.Span("P&L OUVERT", className="lm-key"),
                  html.Span(unreal_str,   className=unreal_cls)],
                 className="lm-item"),
    ]


def _warmup_bar(pct: int) -> html.Div:
    return html.Div(
        html.Div(className="wu-fg", style={"width": f"{pct}%"}),
        className="wu-bg", style={"marginTop": "4px"}
    )


def _pill(active: bool) -> html.Span:
    return html.Span("ACTIF" if active else "IDLE",
                     className="pill-active" if active else "pill-idle")


# ─────────────────────────────────────────────────────────────────────────────
# Chart / metric helpers
# ─────────────────────────────────────────────────────────────────────────────
def _empty_dark_fig(height: int = 200, title: str = "") -> go.Figure:
    """Figure Plotly dark vide (placeholder avant données)."""
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark", height=height,
        paper_bgcolor="#0c1014", plot_bgcolor="#0c1014",
        margin=dict(l=55, r=10, t=30, b=25),
        font=dict(family="Roboto Mono, Courier New", color="#555", size=9),
        xaxis=dict(gridcolor="#1a2530"),
        yaxis=dict(gridcolor="#1a2530"),
        title=dict(text=title, font=dict(size=10, color="#2a9fd6"), x=0),
        showlegend=False,
    )
    return fig


def _rmi(label: str, value: str, color: str = "#eee") -> html.Div:
    """Cellule d'une métrique de risque dans la section live."""
    return html.Div([
        html.Div(label, className="rmi-label"),
        html.Div(value, className="rmi-value", style={"color": color}),
    ], className="risk-metric-item")


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
def _sidebar():
    strat_rows = []
    for name in strategy_names:
        meta = STRATEGY_META.get(name, {})
        label = meta.get("label", name)
        strat_rows.append(
            html.Div([
                dbc.Switch(id={"type": "strat-toggle", "index": name}, value=False,
                           className="me-1", style={"marginTop": "2px"}),
                html.Span(label, className="strat-name"),
                dbc.Button("▶", id={"type": "start-strat-btn", "index": name},
                           color="success", size="sm", className="strat-btn"),
                dbc.Button("■", id={"type": "stop-strat-btn", "index": name},
                           color="warning", size="sm", className="strat-btn"),
                dbc.Button("↺", id={"type": "reset-strat-btn", "index": name},
                           color="secondary", size="sm", className="strat-btn"),
            ], className="strat-row")
        )

    return html.Div([
        # Logo
        html.Div([
            html.Div("AGARTHAI", style={
                "color": "#2a9fd6", "fontSize": "1.1rem", "fontWeight": "500",
                "letterSpacing": ".2em"
            }),
            html.Div(f"v3.0 — {len(strategy_names)} stratégie(s)",
                     className="text-muted", style={"fontSize": ".65rem", "marginTop": "2px"}),
        ], className="sidebar-logo"),

        # Réseau
        html.Div([
            html.Div("Réseau", className="sidebar-section-label"),
            dbc.RadioItems(
                options=[{"label": " mainnet", "value": "mainnet"},
                         {"label": " testnet",  "value": "testnet"}],
                value="mainnet", id="network-select",
                inline=True, className="text-light",
                style={"fontSize": ".75rem"},
            ),
        ], className="sidebar-section"),

        # Coins
        html.Div([
            html.Div("Coins", className="sidebar-section-label"),
            dcc.Dropdown(
                id="coin-select", multi=True,
                placeholder="Sélectionner…",
                style={"fontSize": ".75rem"},
            ),
        ], className="sidebar-section"),

        # Configuration
        html.Div([
            html.Div("Configuration", className="sidebar-section-label"),
            html.Div("Exchange", className="text-muted"),
            dcc.Dropdown(
                id="exchange-select",
                options=[{"label": e, "value": e} for e in EXCHANGES],
                value="Hyperliquid", clearable=False,
                style={"fontSize": ".75rem", "marginBottom": "6px"},
            ),
            html.Div("Timeframe", className="text-muted"),
            dcc.Dropdown(
                id="tf-select",
                options=[{"label": t, "value": t} for t in TIMEFRAMES],
                value="5m", clearable=False,
                style={"fontSize": ".75rem", "marginBottom": "6px"},
            ),
            html.Div("Capital / stratégie ($)", className="text-muted"),
            dbc.Input(id="capital-input", type="number", value=1500, step=100,
                      style={"fontSize": ".8rem", "marginBottom": "6px"}),
            html.Div("Gross leverage", className="text-muted"),
            dcc.Slider(id="leverage-slider", min=1, max=5, step=0.5, value=1.5,
                       marks={1: "1×", 2: "2×", 3: "3×", 5: "5×"},
                       tooltip={"placement": "bottom", "always_visible": True}),
        ], className="sidebar-section"),

        # Connexion
        html.Div([
            html.Div("Connexion", className="sidebar-section-label"),
            dbc.ButtonGroup([
                dbc.Button("Connecter", id="connect-btn", color="primary",
                           size="sm", className="btn-sidebar"),
                dbc.Button("Couper", id="disconnect-btn", color="secondary",
                           size="sm", className="btn-sidebar"),
            ], style={"width": "100%"}),
        ], className="sidebar-section"),

        # Stratégies
        html.Div([
            html.Div("Stratégies", className="sidebar-section-label"),
            html.Div(strat_rows, id="sidebar-strat-list"),
        ], className="sidebar-section"),

        # Global controls
        html.Div([
            dbc.ButtonGroup([
                dbc.Button("▶ Tout lancer", id="start-btn", color="success",
                           size="sm", className="btn-sidebar"),
                dbc.Button("■ Arrêter",   id="stop-btn", color="warning",
                           size="sm", className="btn-sidebar"),
            ], style={"width": "100%", "marginBottom": "6px"}),
            dbc.Button("⚠ STOP D'URGENCE", id="emergency-btn", color="danger",
                       size="sm", style={"width": "100%", "letterSpacing": ".05em"}),
        ], className="sidebar-section"),

    ], id="sidebar")


# ─────────────────────────────────────────────────────────────────────────────
# Tab content builders
# ─────────────────────────────────────────────────────────────────────────────
def _tab_marche():
    """Marché Live tab — strategy overview table + warmup + live metrics."""
    header = html.Div([
        html.Div("#",         className="col-idx"),
        html.Div("Stratégie", className="col-name"),
        html.Div("Type",      className="col-type"),
        html.Div("Turnover",  className="col-turnover"),
        html.Div("Warmup",    className="col-warmup"),
        html.Div("Statut",    className="col-status"),
    ], className="strat-table-row strat-table-header")

    rows = [header]
    for i, name in enumerate(strategy_names, 1):
        meta = STRATEGY_META.get(name, {})
        min_data_label = meta.get("min_data", "")
        rows.append(html.Div([
            # Main strategy row
            html.Div([
                html.Div(str(i),                      className="col-idx"),
                html.Div(meta.get("label", name),     className="col-name"),
                html.Div(meta.get("type", "—"),       className="col-type"),
                html.Div(meta.get("turnover", "—"),   className="col-turnover"),
                html.Div([
                    html.Div([
                        "0% — en attente",
                        html.Span(f"min: {min_data_label}", className="min-data-tag")
                        if min_data_label else None,
                    ], style={"fontSize": ".7rem", "color": "#555"},
                             id={"type": "warmup-label", "index": name}),
                    _warmup_bar(0),
                ], className="col-warmup", id={"type": "warmup-cell", "index": name}),
                html.Div(_pill(False), className="col-status",
                         id={"type": "status-pill", "index": name}),
            ], className="strat-table-row",
               style={"borderBottom": "none", "paddingBottom": "6px"}),
            # Live metrics sub-bar (updated every 3s)
            html.Div(
                _live_metrics_placeholder(),
                className="live-metrics-bar",
                id={"type": "live-metrics-bar", "index": name},
            ),
            # ── Section live — visible après warmup ─────────────────────
            html.Div([
                # Métriques de risque (1 div par métrique)
                html.Div(
                    id={"type": "risk-metrics-div", "index": name},
                    className="risk-metrics-row",
                ),
                # Chart : PnL equity (pleine largeur)
                dcc.Graph(
                    id={"type": "strat-pnl-chart", "index": name},
                    config={"displayModeBar": False},
                    figure=_empty_dark_fig(220, "P&L cumulatif ($)"),
                ),
            ],
                id={"type": "live-section", "index": name},
                className="strat-live-section",
                style={"display": "none"},
            ),
        ], id={"type": "strat-row", "index": name}))

    return html.Div([
        html.Div("Configure et clique Lancer dans le sidebar",
                 style={"color": "#555", "fontSize": ".75rem", "padding": "8px 0 12px",
                        "display": "flex", "alignItems": "center", "gap": "8px"},
                 id="marche-hint"),
        html.Div(rows, style={
            "background": "#111418", "border": "1px solid #2a9fd620",
            "borderRadius": "4px", "overflow": "hidden"
        }),
        dbc.Button("↗ Exporter CSV — toutes stratégies", id="export-csv-btn",
                   color="primary", outline=True, size="sm",
                   style={"marginTop": "14px", "fontSize": ".72rem", "letterSpacing": ".08em"}),
        dcc.Download(id="download-trades-csv"),
    ], style={"padding": "16px 0"})


def _tab_signaux():
    return html.Div([
        html.Div("Signal Feed — derniers signaux générés",
                 style={"color": "#555", "fontSize": ".75rem", "paddingBottom": "10px"}),
        html.Div(id="signal-feed",
                 style={"background": "#0c1014", "border": "1px solid #1a2530",
                        "borderRadius": "4px", "padding": "12px",
                        "minHeight": "200px", "fontSize": ".78rem", "color": "#888"}),
    ], style={"padding": "16px 0"})


def _tab_portfolio():
    return html.Div([
        # Global metrics row
        html.Div(id="portfolio-metrics",
                 style={"display": "flex", "gap": "10px", "marginBottom": "10px"}),
        # Per-strategy PnL cards
        html.Div(id="strat-pnl-cards",
                 style={"display": "flex", "gap": "10px", "marginBottom": "16px",
                        "flexWrap": "wrap"}),
        # Multi-strategy PnL chart
        dcc.Graph(id="pnl-chart", config={"displayModeBar": False}),
        # Allocation pie + positions
        html.Div([
            dbc.Row([
                dbc.Col(dcc.Graph(id="alloc-pie", config={"displayModeBar": False}), width=5),
                dbc.Col(html.Div(id="positions-view"), width=7),
            ])
        ]),
    ], style={"padding": "16px 0"})


def _tab_config():
    return html.Div([
        html.Div("Paramètres avancés — édition via config/strategies.yaml",
                 style={"color": "#555", "fontSize": ".75rem", "paddingBottom": "10px"}),
        html.Pre(id="config-dump",
                 style={"background": "#0c1014", "border": "1px solid #1a2530",
                        "borderRadius": "4px", "padding": "14px",
                        "fontSize": ".72rem", "color": "#7fbfdf",
                        "maxHeight": "60vh", "overflow": "auto"}),
    ], style={"padding": "16px 0"})


def _tab_univers():
    """Hyperliquid universe tab — live coin characteristics."""
    _TH = {"background": "#0c1014", "color": "#2a9fd6", "fontSize": ".65rem",
           "letterSpacing": ".1em", "textTransform": "uppercase",
           "padding": "8px 10px", "border": "none",
           "borderBottom": "1px solid #2a9fd630"}
    _TD = {"padding": "7px 10px", "fontSize": ".75rem", "color": "#ccc",
           "border": "none", "borderBottom": "1px solid #1a2530",
           "background": "#111418"}

    return html.Div([
        html.Div([
            html.Span("Univers Hyperliquid Perps",
                      style={"color": "#2a9fd6", "fontSize": ".85rem",
                             "fontWeight": "500", "letterSpacing": ".06em"}),
            html.Span(id="hl-universe-ts",
                      style={"color": "#444", "fontSize": ".65rem", "marginLeft": "12px"}),
            dbc.Button("↻ Rafraîchir", id="hl-refresh-btn", color="primary",
                       outline=True, size="sm",
                       style={"marginLeft": "auto", "fontSize": ".65rem",
                              "padding": "2px 10px"}),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "12px"}),

        # Summary metrics row
        html.Div(id="hl-universe-metrics",
                 style={"display": "flex", "gap": "10px", "marginBottom": "14px"}),

        # Coin table
        dash_table.DataTable(
            id="hl-universe-table",
            columns=[
                {"name": "Coin",        "id": "name"},
                {"name": "Mark Price",  "id": "markPx",           "type": "numeric",
                 "format": {"specifier": ",.4f"}},
                {"name": "24h Chg %",   "id": "chg24h_pct",       "type": "numeric",
                 "format": {"specifier": "+.2f"}},
                {"name": "Vol 24h $M",  "id": "dayNtlVlm_m",      "type": "numeric",
                 "format": {"specifier": ",.1f"}},
                {"name": "OI $M",       "id": "oi_m",             "type": "numeric",
                 "format": {"specifier": ",.1f"}},
                {"name": "Fund 8h %",   "id": "funding_8h",       "type": "numeric",
                 "format": {"specifier": "+.4f"}},
                {"name": "Fund Ann %",  "id": "funding_ann",      "type": "numeric",
                 "format": {"specifier": "+.1f"}},
                {"name": "Lev Max",     "id": "maxLeverage"},
            ],
            data=[],
            sort_action="native",
            filter_action="native",
            page_size=50,
            style_table={"overflowX": "auto", "background": "#111418",
                         "border": "1px solid #2a9fd620", "borderRadius": "4px"},
            style_header={**_TH},
            style_data={**_TD},
            style_cell={"fontFamily": "Roboto Mono, Courier New",
                        "minWidth": "80px"},
            style_data_conditional=[
                # Positive 24h change → green
                {"if": {"filter_query": "{chg24h_pct} > 0", "column_id": "chg24h_pct"},
                 "color": "#77b300"},
                # Negative 24h change → red
                {"if": {"filter_query": "{chg24h_pct} < 0", "column_id": "chg24h_pct"},
                 "color": "#cc0000"},
                # Positive funding → bearish (longs pay)
                {"if": {"filter_query": "{funding_8h} > 0.01", "column_id": "funding_8h"},
                 "color": "#ff7518"},
                # Negative funding → bullish (shorts pay)
                {"if": {"filter_query": "{funding_8h} < -0.01", "column_id": "funding_8h"},
                 "color": "#77b300"},
                # Row hover
                {"if": {"state": "active"}, "background": "#151c24", "border": "none"},
            ],
            style_filter={"background": "#0c1014", "color": "#aaa",
                          "border": "none", "borderBottom": "1px solid #1a2530",
                          "fontSize": ".72rem"},
        ),
    ], style={"padding": "16px 0"})


# ─────────────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────────────
app.layout = html.Div([
    dbc.Row([
        # ── Sidebar ────────────────────────────────────────────────────
        dbc.Col(_sidebar(), id="sidebar-col", width=2,
                style={"padding": "0", "minHeight": "100vh"}),

        # ── Main ───────────────────────────────────────────────────────
        dbc.Col([
            # Header bar
            html.Div([
                html.Span("Agarthai — Live Multi-Strategy",
                          style={"fontSize": "1rem", "fontWeight": "500", "color": "#ddd",
                                 "letterSpacing": ".06em", "flex": "1"}),
                dbc.Switch(id="global-toggle", value=False, label="",
                           style={"marginBottom": "0"}),
                html.Span(id="global-status-label",
                          style={"fontSize": ".75rem", "color": "#555", "width": "70px"}),
                html.Span("—", id="global-mode-badge",
                          style={"fontSize": ".72rem", "color": "#555"}),
                dbc.Badge("0 actifs", id="badge-actifs", color="secondary",
                          style={"fontSize": ".7rem"}),
                dbc.Badge("0 LIVE",  id="badge-live",   color="secondary",
                          style={"fontSize": ".7rem", "marginLeft": "6px"}),
            ], id="main-header"),

            # Price ticker bar
            html.Div([
                html.Span(id="price-ticker",
                          style={"fontSize": ".8rem", "color": "#2a9fd6"}),
                html.Span(id="price-source-badge",
                          style={"fontSize": ".65rem", "color": "#444", "marginLeft": "8px"}),
                html.Span(id="clock",
                          style={"fontSize": ".75rem", "color": "#444",
                                 "marginLeft": "auto"}),
            ], style={
                "display": "flex", "alignItems": "center",
                "padding": "5px 20px", "background": "#0c1014",
                "borderBottom": "1px solid #2a9fd610",
            }),

            # Status bar
            dbc.Alert(id="status-alert", color="secondary", dismissable=False,
                      style={"margin": "8px 16px 0", "padding": "7px 14px",
                             "fontSize": ".75rem", "borderRadius": "3px"}),

            # Tabs
            html.Div([
                dbc.Tabs([
                    dbc.Tab(_tab_marche(),    label="Marché Live",    tab_id="tab-marche"),
                    dbc.Tab(_tab_signaux(),   label="Signaux",        tab_id="tab-signaux"),
                    dbc.Tab(_tab_portfolio(), label="Portfolio",      tab_id="tab-portfolio"),
                    dbc.Tab(_tab_univers(),   label="Univers HL",     tab_id="tab-univers"),
                    dbc.Tab(_tab_config(),    label="Config",         tab_id="tab-config"),
                ], id="main-tabs", active_tab="tab-marche"),
            ], style={"padding": "0 16px"}),

        ], width=10, style={"padding": "0"}),
    ], style={"margin": "0"}),

    # Intervals
    dcc.Interval(id="refresh-interval",         interval=3000,  n_intervals=0),
    dcc.Interval(id="hl-universe-interval",     interval=30000, n_intervals=0),  # 30s
    dcc.Store(id="conn-store", data={"connected": False, "exchange": "Hyperliquid"}),

], style={"minHeight": "100vh", "background": "#0c1014"})


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("coin-select", "options"),
    Output("coin-select", "value"),
    Input("exchange-select", "value"),
)
def update_coins(exchange):
    coins = []
    # For Hyperliquid: try to fetch full universe for dynamic coin list
    if "hyperliquid" in exchange.lower():
        try:
            univ = _fetch_hl_universe()
            if univ:
                coins = [c["name"] for c in univ]
        except Exception:
            pass
    # Fallback to engine or hardcoded
    if not coins and engine:
        try:
            coins = engine.get_available_coins() or []
        except Exception:
            pass
    if not coins:
        coins = DEFAULT_COINS.get(exchange, ["BTC"])
    default_sel = [c for c in ["BTC", "ETH", "SOL"] if c in coins][:3] or coins[:3]
    return [{"label": c, "value": c} for c in coins], default_sel


@app.callback(
    Output("status-alert", "children"),
    Output("status-alert", "color"),
    Output("conn-store", "data"),
    Input("connect-btn",    "n_clicks"),
    Input("disconnect-btn", "n_clicks"),
    Input("start-btn",      "n_clicks"),
    Input("stop-btn",       "n_clicks"),
    Input("emergency-btn",  "n_clicks"),
    State("exchange-select", "value"),
    State("tf-select",       "value"),
    State("coin-select",     "value"),
    State("conn-store",      "data"),
    prevent_initial_call=True,
)
def handle_controls(conn, disc, start, stop, emerg, exchange, tf_val, coin_sel, store):
    btn = ctx.triggered_id
    store = store or {}

    if engine is None:
        return "Engine non chargé — vérifier les logs.", "danger", store

    # Determine primary coin from dropdown selection
    primary_coin = "BTC"
    if isinstance(coin_sel, list) and coin_sel:
        primary_coin = coin_sel[0]
    elif isinstance(coin_sel, str) and coin_sel:
        primary_coin = coin_sel

    if btn == "connect-btn":
        ex_key = exchange.lower().replace(" ", "_").replace("_futures", "")
        # Apply tick interval from timeframe dropdown
        tick_sec = _TF_TO_SEC.get(tf_val, 1)
        ok = engine.connect(ex_key, paper=True)
        if ok:
            engine.tick_interval_sec = tick_sec
            store["connected"] = True
            store["exchange"] = exchange
            store["tick_interval"] = tick_sec
            store["coin"] = primary_coin
            return (f"Connecté à {exchange} (paper) — tick {tf_val} — coin {primary_coin}",
                    "success", store)
        return "Connexion échouée.", "danger", store

    if btn == "disconnect-btn":
        engine.disconnect()
        store["connected"] = False
        return "Déconnecté.", "secondary", store

    if btn == "start-btn":
        if engine.start(coin=primary_coin):
            coins_str = ", ".join(coin_sel) if isinstance(coin_sel, list) else primary_coin
            return f"Toutes les stratégies démarrées sur {coins_str}.", "success", store
        return "Impossible de démarrer — connecter d'abord.", "warning", store

    if btn == "stop-btn":
        engine.stop()
        return "Toutes les stratégies arrêtées.", "warning", store

    if btn == "emergency-btn":
        engine.emergency_stop()
        return "STOP D'URGENCE — toutes positions fermées.", "danger", store

    return dash.no_update, dash.no_update, store


@app.callback(
    Output("status-alert", "children", allow_duplicate=True),
    Output("status-alert", "color",    allow_duplicate=True),
    Input({"type": "start-strat-btn", "index": dash.ALL}, "n_clicks"),
    Input({"type": "stop-strat-btn",  "index": dash.ALL}, "n_clicks"),
    State("coin-select", "value"),
    prevent_initial_call=True,
)
def handle_per_strategy(starts, stops, coin_sel):
    if engine is None:
        return "Engine non chargé.", "danger"
    trig = ctx.triggered_id
    if not isinstance(trig, dict):
        return dash.no_update, dash.no_update
    name   = trig.get("index")
    action = trig.get("type")

    # Determine coin from dropdown
    primary_coin = "BTC"
    if isinstance(coin_sel, list) and coin_sel:
        primary_coin = coin_sel[0]
    elif isinstance(coin_sel, str) and coin_sel:
        primary_coin = coin_sel

    if action == "start-strat-btn":
        ok = engine.start_strategy(name, coin=primary_coin)
        return (f"Stratégie démarrée : {name} sur {primary_coin}", "success") if ok else (f"Impossible de démarrer {name}", "warning")
    if action == "stop-strat-btn":
        ok = engine.stop_strategy(name)
        return (f"Stratégie arrêtée : {name}", "warning") if ok else (f"Impossible d'arrêter {name}", "danger")
    return dash.no_update, dash.no_update


@app.callback(
    # Header
    Output("global-status-label", "children"),
    Output("global-mode-badge",   "children"),
    Output("badge-actifs",        "children"),
    Output("badge-live",          "children"),
    Output("badge-actifs",        "color"),
    # Clock + price
    Output("clock",               "children"),
    Output("price-ticker",        "children"),
    Output("price-source-badge",  "children"),
    # Marché Live tab — per-strategy cells
    Output({"type": "warmup-label", "index": dash.ALL}, "children"),
    Output({"type": "status-pill",  "index": dash.ALL}, "children"),
    Output("marche-hint",          "children"),
    # Live metrics bars (one per strategy row)
    Output({"type": "live-metrics-bar", "index": dash.ALL}, "children"),
    # Portfolio
    Output("portfolio-metrics",   "children"),
    Output("strat-pnl-cards",     "children"),
    Output("pnl-chart",           "figure"),
    Output("alloc-pie",           "figure"),
    Output("positions-view",      "children"),
    # Config dump
    Output("config-dump",         "children"),
    # Signals
    Output("signal-feed",         "children"),
    # ── Section live par stratégie (post-warmup) ────────────────────────
    Output({"type": "live-section",      "index": dash.ALL}, "style"),
    Output({"type": "risk-metrics-div",  "index": dash.ALL}, "children"),
    Output({"type": "strat-pnl-chart",   "index": dash.ALL}, "figure"),
    Input("refresh-interval",     "n_intervals"),
    State("conn-store",           "data"),
    State("coin-select",          "value"),
    State("capital-input",        "value"),
)
def refresh(_, store, coins, capital_input):
    now  = datetime.now().strftime("%H:%M:%S")
    capital = float(capital_input or 1500)

    # ── Price ──────────────────────────────────────────────────────────
    primary_coin = (coins[0] if coins else "BTC") if isinstance(coins, list) else "BTC"
    price, price_src = _fetch_price(primary_coin)
    ticker_txt = f"{primary_coin}  ${price:,.2f}"
    src_txt    = f"▸ {price_src}"

    # ── Engine status ───────────────────────────────────────────────────
    dark_fig = go.Figure(layout=dict(
        template="plotly_dark", height=300,
        paper_bgcolor="#0c1014", plot_bgcolor="#0c1014",
        margin=dict(l=40, r=20, t=30, b=30),
        font=dict(family="Roboto Mono, Courier New", color="#666"),
    ))

    n_strats = len(strategy_names)
    warmup_labels       = ["—"] * n_strats
    status_pills        = [_pill(False)] * n_strats
    live_metrics_bars   = [_live_metrics_placeholder()] * n_strats
    strat_pnl_cards_content = []
    hint_txt       = "Configure et clique Lancer dans le sidebar"
    global_status  = "IDLE"
    global_mode    = "paper"
    n_active       = 0
    # Listes pour les 3 Outputs de la section live
    live_section_styles   = [{"display": "none"}] * n_strats
    risk_metrics_contents = [html.Div()] * n_strats
    pnl_chart_figs        = [_empty_dark_fig(220, "P&L cumulatif ($)")] * n_strats
    portfolio_metrics = _metric_cards([
        ("Capital",        f"${capital:,.0f}",  ""),
        ("Réalisé PnL",    "$0.00",              ""),
        ("Non-réalisé",    "$0.00",              ""),
        ("Trades",         "0",                  ""),
        ("Win rate",       "—",                  ""),
        ("Actifs",         "0",                  ""),
    ])
    fig_pnl  = dark_fig
    fig_pie  = dark_fig
    pos_view = html.Span("Aucune position ouverte.", style={"color": "#555", "fontSize": ".78rem"})
    config_txt = _load_config_yaml()
    signal_txt = "Aucun signal — stratégies en attente ou en warmup."

    if engine is not None:
        status = engine.get_status()
        strats  = status.get("strategies", {})
        runtime          = status.get("strategy_runtime", {})
        per_strategy_pnl = status.get("per_strategy_pnl", {})
        live_metrics     = status.get("live_metrics", {})
        n_active = sum(1 for r in runtime.values() if r.get("active"))
        global_status = "ACTIF" if status.get("running") else "IDLE"

        # Warmup bars + section live
        for i, name in enumerate(strategy_names):
            rt        = runtime.get(name, {})
            req       = max(1, int(rt.get("warmup_required_sec", 1)))
            buf       = int(rt.get("buffered_sec", 0))
            pct       = min(100, int(buf / req * 100))
            done_flag = rt.get("warmup_done", False)
            done_at   = rt.get("warmup_done_at", "")

            if done_flag:
                # ── Barre verte "WARMUP COMPLET" ────────────────────────
                warmup_labels[i] = html.Div([
                    html.Span([
                        html.Span(className="warmup-spinner-done"),
                        "WARMUP COMPLET",
                        html.Span("✓ LIVE", className="warmup-done-badge"),
                        html.Span(
                            f" @ {done_at} — {buf:,} rows",
                            style={"fontSize": ".65rem", "color": "#555",
                                   "marginLeft": "6px"}
                        ),
                    ], style={"fontSize": ".72rem", "color": "#77b300"}),
                    html.Div(
                        html.Div(className="wu-fg wu-fg-done",
                                 style={"width": "100%"}),
                        className="wu-bg", style={"marginTop": "4px"}
                    ),
                ])
                # ── Section live : métriques + graphiques ────────────────
                if rt.get("active", False):
                    live_section_styles[i] = {"display": "block"}
                    lm_n   = live_metrics.get(name, {})
                    pnls_n = engine._per_strategy_pnl.get(name, [])
                    m      = engine.get_risk_metrics(name)
                    unreal_n = lm_n.get("unrealized_pnl", 0.0)

                    pnl_col = "#77b300" if m["realized"] >= 0 else "#cc0000"
                    ur_col  = "#77b300" if unreal_n > 0 else "#cc0000" if unreal_n < 0 else "#888"
                    wr_col  = "#77b300" if m["win_rate"] >= 0.5 else "#cc0000"
                    dd_col  = "#cc0000" if m["max_dd"] < -0.01 else "#888"
                    sh_col  = "#77b300" if m["sharpe"] >= 1.0 else "#888"

                    risk_metrics_contents[i] = [
                        _rmi("RÉALISÉ",  f"${m['realized']:+.2f}",           pnl_col),
                        _rmi("NON-RÉA.", f"${unreal_n:+.2f}",                ur_col),
                        _rmi("WIN RATE", f"{m['win_rate']*100:.0f}%",         wr_col),
                        _rmi("MAX DD",   f"${m['max_dd']:.2f}",               dd_col),
                        _rmi("SHARPE",   f"{m['sharpe']:.2f}" if m["n_trades"] > 1 else "—",
                             sh_col),
                        _rmi("TRADES",   str(m["n_trades"]),                  "#2a9fd6"),
                    ]

                    # PnL equity curve
                    if pnls_n:
                        eq_n  = np.cumsum(pnls_n)
                        col_n = "#77b300" if eq_n[-1] >= 0 else "#cc0000"
                        rgb_n = "119,179,0" if eq_n[-1] >= 0 else "204,0,0"
                        fig_p = go.Figure(data=[go.Scatter(
                            x=list(range(len(eq_n) + 1)),
                            y=[0.0] + list(eq_n),
                            mode="lines",
                            line=dict(color=col_n, width=1.5),
                            fill="tozeroy",
                            fillcolor=f"rgba({rgb_n},0.08)",
                        )])
                    else:
                        fig_p = _empty_dark_fig(180, "P&L cumulatif ($)")
                    fig_p.update_layout(
                        template="plotly_dark", height=180,
                        paper_bgcolor="#0c1014", plot_bgcolor="#0c1014",
                        margin=dict(l=55, r=10, t=28, b=25),
                        font=dict(family="Roboto Mono, Courier New", color="#555", size=9),
                        xaxis=dict(gridcolor="#1a2530",
                                   title=dict(text="# trade", font=dict(size=8))),
                        yaxis=dict(gridcolor="#1a2530",
                                   title=dict(text="PnL $", font=dict(size=8)),
                                   zeroline=True, zerolinecolor="#2a2a2a"),
                        title=dict(
                            text=f"Équity — {STRATEGY_META.get(name, {}).get('label', name)}",
                            font=dict(size=10, color="#2a9fd6"), x=0,
                        ),
                        showlegend=False,
                    )
                    pnl_chart_figs[i] = fig_p

            else:
                # ── Barre de progression normale ─────────────────────────
                is_active = rt.get("active", False)
                meta_i = STRATEGY_META.get(name, {})
                min_data_label = meta_i.get("min_data", "")
                if is_active and pct < 100:
                    # Spinner animé pendant le warmup actif
                    warmup_labels[i] = html.Div([
                        html.Span([
                            html.Span(className="warmup-spinner"),
                            f"Chargement… {pct}%",
                            html.Span(f" — {buf:,} / {req:,} rows",
                                      style={"fontSize": ".63rem", "color": "#555",
                                             "marginLeft": "6px"}),
                        ], style={"fontSize": ".7rem", "color": "#2a9fd6"}),
                        _warmup_bar(pct),
                    ])
                else:
                    warmup_labels[i] = html.Div([
                        html.Span([
                            f"0% — en attente",
                            html.Span(f"min: {min_data_label}", className="min-data-tag")
                            if min_data_label else None,
                        ], style={"fontSize": ".7rem", "color": "#555"}),
                        _warmup_bar(pct),
                    ])

            status_pills[i] = _pill(rt.get("active", False))

        hint_txt = (
            f"{n_active} stratégie(s) active(s) — scan en cours…"
            if n_active > 0 else
            "Configure et clique Lancer dans le sidebar"
        )

        # Live metrics bars (one per strategy)
        for i, name in enumerate(strategy_names):
            lm = live_metrics.get(name, {})
            live_metrics_bars[i] = _build_live_metrics_bar(lm)

        # Portfolio
        total_pnl        = status.get("total_pnl", 0.0)
        n_trades         = sum(s.get("n_trades", 0) for s in strats.values())
        total_unrealized = sum(lm.get('unrealized_pnl', 0.0) for lm in live_metrics.values())
        win_rate         = "—"

        portfolio_metrics = _metric_cards([
            ("Capital",      f"${capital + total_pnl:,.0f}",    ""),
            ("Réalisé PnL",  f"${total_pnl:+.2f}",
             "success" if total_pnl >= 0 else "danger"),
            ("Non-réalisé",  f"${total_unrealized:+.2f}",
             "success" if total_unrealized > 0 else "danger" if total_unrealized < 0 else ""),
            ("Trades",       str(n_trades),                      ""),
            ("Win rate",     win_rate,                           ""),
            ("Actifs",       str(n_active),                      "primary" if n_active else ""),
        ])

        # Per-strategy PnL cards
        strat_pnl_cards_content = []
        for sn in strategy_names:
            plist      = per_strategy_pnl.get(sn, [])
            s_pnl      = sum(plist) if plist else 0.0
            n_trades_s = len(plist)
            lm_s       = live_metrics.get(sn, {})
            unreal_s   = lm_s.get('unrealized_pnl', 0.0)
            label_s    = STRATEGY_META.get(sn, {}).get("label", sn)
            color_s    = STRAT_COLORS.get(sn, "#aaa")
            pnl_clr    = "#77b300" if s_pnl > 0 else "#cc0000" if s_pnl < 0 else "#888"
            ur_clr     = "#77b300" if unreal_s > 0 else "#cc0000" if unreal_s < 0 else "#555"
            ticks_s    = lm_s.get('ticks', 0)
            strat_pnl_cards_content.append(html.Div([
                html.Div(label_s, className="spc-name", style={"color": color_s}),
                html.Div(f"${s_pnl:+.2f}", className="spc-pnl", style={"color": pnl_clr}),
                html.Div([
                    html.Span(f"{n_trades_s} trade(s)", className="spc-sub"),
                    html.Span(f" · ouvert: ", className="spc-sub"),
                    html.Span(f"${unreal_s:+.2f}", className="spc-sub", style={"color": ur_clr}),
                ]),
                html.Div(f"{ticks_s:,} ticks", className="spc-sub",
                         style={"color": "#2a9fd640", "marginTop": "2px"}),
            ], className="strat-pnl-card"))

        # PnL chart — multi-strategy
        pnl_hist  = engine.get_current_pnl() or []
        has_trades = bool(pnl_hist)
        if has_trades:
            traces = []
            for sn in strategy_names:
                plist_s = per_strategy_pnl.get(sn, [])
                if not plist_s:
                    continue
                eq_s    = np.cumsum(plist_s)
                color_s = STRAT_COLORS.get(sn, "#aaa")
                label_s = STRATEGY_META.get(sn, {}).get("label", sn)
                traces.append(go.Scatter(
                    x=list(range(len(eq_s))),
                    y=eq_s.tolist(),
                    mode="lines",
                    name=label_s,
                    line=dict(color=color_s, width=1.5),
                    hovertemplate=f"<b>{label_s}</b><br>Trade #%{{x}}<br>Cumul: $%{{y:+.2f}}<extra></extra>",
                ))
            # Total portfolio trace
            eq_total = np.cumsum(pnl_hist)
            traces.append(go.Scatter(
                x=list(range(len(eq_total))),
                y=eq_total.tolist(),
                mode="lines",
                name="Total",
                line=dict(color="#ffffff", width=2, dash="dot"),
                hovertemplate="<b>Total</b><br>Trade #%{x}<br>Cumul: $%{y:+.2f}<extra></extra>",
            ))
            fig_pnl = go.Figure(data=traces)
        else:
            names_bar = list(strats.keys())
            warmup_pcts = []
            for n in names_bar:
                rt2  = runtime.get(n, {})
                req2 = max(1, int(rt2.get("warmup_required_sec", 1)))
                buf2 = int(rt2.get("buffered_sec", 0))
                warmup_pcts.append(min(100, int(buf2 / req2 * 100)))
            colors_bar = [STRAT_COLORS.get(n, "#2a9fd6") for n in names_bar]
            fig_pnl = go.Figure(data=[go.Bar(
                x=[STRATEGY_META.get(n, {}).get("label", n) for n in names_bar] or ["—"],
                y=warmup_pcts or [0],
                marker_color=colors_bar,
                text=[f"{p}%" for p in warmup_pcts],
                textposition="outside",
            )])
            fig_pnl.add_annotation(
                text="Warmup en cours — aucun trade exécuté",
                xref="paper", yref="paper", x=0.5, y=1.05,
                showarrow=False, font=dict(color="#555", size=11),
            )

        fig_pnl.update_layout(
            template="plotly_dark", height=300,
            paper_bgcolor="#111418", plot_bgcolor="#111418",
            margin=dict(l=50, r=20, t=40, b=30),
            font=dict(family="Roboto Mono, Courier New", color="#666"),
            title=dict(text="PnL cumulatif par stratégie ($)",
                       font=dict(size=12, color="#2a9fd6"), x=0),
            yaxis=dict(gridcolor="#1a2530", zeroline=True,
                       zerolinecolor="#2a2a2a", zerolinewidth=1),
            xaxis=dict(gridcolor="#1a2530", title=dict(text="# trade", font=dict(size=10))),
            legend=dict(
                bgcolor="#0c1014", bordercolor="#1e4a5e", borderwidth=1,
                font=dict(size=10), x=0.01, y=0.99, xanchor="left", yanchor="top",
            ),
            hovermode="x unified",
        )

        # Pie
        cap_vals  = [s.get("capital", 0) for s in strats.values()]
        pie_names = [STRATEGY_META.get(n, {}).get("label", n) for n in strats.keys()]
        fig_pie = go.Figure(data=[go.Pie(
            labels=pie_names or ["—"],
            values=cap_vals or [1],
            hole=0.45,
            marker_colors=[STRAT_COLORS.get(n, "#aaa") for n in strats.keys()],
            textfont=dict(size=10),
        )])
        fig_pie.update_layout(
            template="plotly_dark", height=280,
            paper_bgcolor="#111418", plot_bgcolor="#111418",
            margin=dict(l=20, r=20, t=30, b=20),
            font=dict(family="Roboto Mono, Courier New", color="#666"),
            title=dict(text="Allocation capital", font=dict(size=12, color="#2a9fd6"), x=0),
            legend=dict(font=dict(size=10)),
        )

        # Positions
        positions = status.get("positions", {}) or {}
        if positions:
            pos_rows = []
            for sn, pos in positions.items():
                d      = pos.get("direction", 0)
                side   = "LONG" if d == 1 else "SHORT"
                col    = "#77b300" if d == 1 else "#cc0000"
                unreal = pos.get('unrealized_pnl', 0.0)
                ur_clr = "#77b300" if unreal > 0 else "#cc0000" if unreal < 0 else "#888"
                cur_px = pos.get('current_price', pos.get('exec_price', pos.get('entry_price', 0)))
                exec_px = pos.get('exec_price', pos.get('entry_price', 0))
                coin   = pos.get('coin', '—')
                pos_rows.append(html.Div([
                    html.Span(f"{sn}", style={"flex": "2", "color": "#bbb"}),
                    html.Span(f"{side} {coin}", style={"flex": "2", "color": col}),
                    html.Span(f"${exec_px:,.2f}", style={"flex": "2"}),
                    html.Span(f"${cur_px:,.2f}", style={"flex": "2", "color": "#888"}),
                    html.Span(f"${pos.get('size_usd', 0):,.0f}", style={"flex": "1"}),
                    html.Span(f"${unreal:+.2f}", style={"flex": "1", "color": ur_clr,
                                                         "fontWeight": "500"}),
                ], style={"display": "flex", "gap": "8px", "padding": "7px 0",
                          "fontSize": ".78rem", "borderBottom": "1px solid #1a2530"}))
            pos_view = html.Div([
                html.Div([
                    html.Span("Stratégie",   style={"flex": "2"}),
                    html.Span("Side / Coin", style={"flex": "2"}),
                    html.Span("Entry",       style={"flex": "2"}),
                    html.Span("Prix actuel", style={"flex": "2"}),
                    html.Span("Taille",      style={"flex": "1"}),
                    html.Span("Non-réalisé", style={"flex": "1"}),
                ], style={"display": "flex", "gap": "8px", "padding": "7px 0",
                          "fontSize": ".65rem", "color": "#2a9fd6",
                          "letterSpacing": ".1em", "textTransform": "uppercase"}),
                html.Div(pos_rows),
            ])
        else:
            pos_view = html.Span("Aucune position ouverte.",
                                 style={"color": "#555", "fontSize": ".78rem"})

        # Signal log
        sig_log = status.get("signal_log", [])
        if sig_log:
            rows = []
            for s in sig_log:
                col = "#77b300" if s.get("direction") == "LONG" else "#cc0000"
                rows.append(html.Div([
                    html.Span(str(s.get("timestamp", ""))[:19],
                              style={"flex": "3", "color": "#555"}),
                    html.Span(s.get("strategy", ""),
                              style={"flex": "2", "color": "#2a9fd6"}),
                    html.Span(s.get("direction", ""),
                              style={"flex": "1", "color": col, "fontWeight": "500"}),
                    html.Span(f"${s.get('price', 0):,.2f}",
                              style={"flex": "2"}),
                    html.Span(f"conf {s.get('conf', 0):.2f}",
                              style={"flex": "1", "color": "#555"}),
                ], style={"display": "flex", "gap": "8px", "padding": "5px 0",
                          "borderBottom": "1px solid #1a2530",
                          "fontSize": ".76rem", "color": "#bbb"}))
            signal_txt = html.Div(rows)
        else:
            signal_txt = "Aucun signal — stratégies en warmup ou en attente de conditions."

    badge_color  = "success" if n_active > 0 else "secondary"
    badge_actifs = f"{n_active} actif(s)"
    badge_live   = "PAPER"

    return (
        global_status, global_mode,
        badge_actifs, badge_live, badge_color,
        now, ticker_txt, src_txt,
        warmup_labels, status_pills, hint_txt,
        live_metrics_bars,
        portfolio_metrics, strat_pnl_cards_content, fig_pnl, fig_pie, pos_view,
        config_txt,
        signal_txt,
        # ── Section live par stratégie ──────────────────────────────────
        live_section_styles,
        risk_metrics_contents,
        pnl_chart_figs,
    )


@app.callback(
    Output("hl-universe-table",   "data"),
    Output("hl-universe-metrics", "children"),
    Output("hl-universe-ts",      "children"),
    Input("hl-universe-interval", "n_intervals"),
    Input("hl-refresh-btn",       "n_clicks"),
)
def refresh_hl_universe(_, __):
    coins = _fetch_hl_universe()
    now_str = f"màj {datetime.now().strftime('%H:%M:%S')}"

    if not coins:
        return [], [html.Span("Données non disponibles", style={"color": "#555"})], now_str

    # Build table rows
    rows = []
    for c in coins:
        rows.append({
            "name":        c["name"],
            "markPx":      c["markPx"],
            "chg24h_pct":  c["chg24h_pct"],
            "dayNtlVlm_m": round(c["dayNtlVlm"] / 1e6, 1),
            "oi_m":        round(c["openInterest_usd"] / 1e6, 1),
            "funding_8h":  c["funding_8h"],
            "funding_ann": c["funding_ann"],
            "maxLeverage": c["maxLeverage"],
        })

    # Summary metrics
    total_vol = sum(c["dayNtlVlm"] for c in coins) / 1e9
    total_oi  = sum(c["openInterest_usd"] for c in coins) / 1e9
    gainers   = sum(1 for c in coins if c["chg24h_pct"] > 0)
    losers    = sum(1 for c in coins if c["chg24h_pct"] < 0)
    avg_fund  = sum(c["funding_8h"] for c in coins) / len(coins) if coins else 0

    summary = _metric_cards([
        ("Coins listés",  str(len(coins)),                          "primary"),
        ("Vol 24h $B",    f"{total_vol:.2f}",                       ""),
        ("OI total $B",   f"{total_oi:.2f}",                        ""),
        ("Gainers / Losers", f"{gainers} / {losers}",              "success" if gainers > losers else "danger"),
        ("Funding moyen 8h", f"{avg_fund:+.4f}%",                  "danger" if avg_fund > 0.01 else ""),
    ])

    return rows, summary, now_str


@app.callback(
    Output("download-trades-csv", "data"),
    Input("export-csv-btn", "n_clicks"),
    prevent_initial_call=True,
)
def export_csv(_):
    if engine is None:
        return dash.no_update
    rows = []
    for name, sd in engine.strategies.items():
        for tr in sd["instance"].trade_history:
            rows.append({
                "strategy": name,
                "timestamp": tr.signal.timestamp,
                "entry_price": tr.entry_price,
                "exit_price": tr.exit_price,
                "pnl_usd": tr.pnl_usd,
                "pnl_pct": tr.pnl_pct,
                "fees_usd": tr.fees_usd,
                "slippage_usd": tr.slippage_usd,
                "exit_reason": tr.exit_reason,
                "duration_sec": tr.duration_sec,
                "position_usd": tr.position_usd,
            })
    if not rows:
        rows = [{"info": "Aucun trade pour l'instant"}]
    df = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    path = Path("results") / f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(path, index=False)
    return dcc.send_file(str(path))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers for UI elements
# ─────────────────────────────────────────────────────────────────────────────
def _metric_cards(items):
    cards = []
    for label, value, color in items:
        val_color = {"success": "#77b300", "danger": "#cc0000",
                     "primary": "#2a9fd6"}.get(color, "#eee")
        cards.append(html.Div([
            html.Div(label, className="metric-label"),
            html.Div(value, className="metric-value", style={"color": val_color}),
        ], className="metric-card", style={"flex": "1"}))
    return cards


def _load_config_yaml():
    try:
        import yaml
        p = Path("config/strategies.yaml")
        if p.exists():
            with open(p) as f:
                return yaml.dump(yaml.safe_load(f), default_flow_style=False, allow_unicode=True)
    except Exception:
        pass
    return "config/strategies.yaml non disponible"


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=8055, host="127.0.0.1")
