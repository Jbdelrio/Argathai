"""
Agarthai — Live Trading Dashboard (Dash + Bootstrap Cyborg)
============================================================
python live/dashboard.py  →  http://localhost:8050

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

      /* ── Charts ─────────────────────────────────────────── */
      .js-plotly-plot .plotly { border-radius: 4px; }

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
    "baudouin4": {"label": "Baudouin 4",     "type": "Mean-Rev",      "turnover": "Moyen"},
    "innocent3": {"label": "Innocent 3",     "type": "Stat-Arb",      "turnover": "Faible"},
    "urbain2":   {"label": "Urbain 2",       "type": "Res. Momentum", "turnover": "Faible"},
}

EXCHANGES = ["Hyperliquid", "Bitget Futures"]
DEFAULT_COINS = {
    "Hyperliquid": ["BTC", "ETH", "SOL", "DOGE", "ARB", "OP", "AVAX", "LINK"],
    "Bitget Futures": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"],
}
TIMEFRAMES = ["1s", "5s", "15s", "30s", "1m", "5m"]

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


def _warmup_bar(pct: int) -> html.Div:
    return html.Div(
        html.Div(className="wu-fg", style={"width": f"{pct}%"}),
        className="wu-bg", style={"marginTop": "4px"}
    )


def _pill(active: bool) -> html.Span:
    return html.Span("ACTIF" if active else "IDLE",
                     className="pill-active" if active else "pill-idle")


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
    """Marché Live tab — strategy overview table + warmup."""
    header = html.Div([
        html.Div("#",        className="col-idx"),
        html.Div("Stratégie", className="col-name"),
        html.Div("Type",      className="col-type"),
        html.Div("Turnover",  className="col-turnover"),
        html.Div("Warmup",    className="col-warmup"),
        html.Div("Statut",    className="col-status"),
    ], className="strat-table-row strat-table-header")

    rows = [header]
    for i, name in enumerate(strategy_names, 1):
        meta = STRATEGY_META.get(name, {})
        rows.append(html.Div([
            html.Div(str(i),                      className="col-idx"),
            html.Div(meta.get("label", name),     className="col-name"),
            html.Div(meta.get("type", "—"),       className="col-type"),
            html.Div(meta.get("turnover", "—"),   className="col-turnover"),
            html.Div([
                html.Div("0% — en attente", style={"fontSize": ".7rem", "color": "#555"},
                         id={"type": "warmup-label", "index": name}),
                _warmup_bar(0),
            ], className="col-warmup", id={"type": "warmup-cell", "index": name}),
            html.Div(_pill(False), className="col-status",
                     id={"type": "status-pill", "index": name}),
        ], className="strat-table-row", id={"type": "strat-row", "index": name}))

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
        # Metrics row
        html.Div(id="portfolio-metrics", style={"display": "flex", "gap": "10px", "marginBottom": "16px"}),
        # PnL chart
        dcc.Graph(id="pnl-chart", config={"displayModeBar": False}),
        # Allocation pie
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
                    dbc.Tab(_tab_config(),    label="Config",         tab_id="tab-config"),
                ], id="main-tabs", active_tab="tab-marche"),
            ], style={"padding": "0 16px"}),

        ], width=10, style={"padding": "0"}),
    ], style={"margin": "0"}),

    # Interval
    dcc.Interval(id="refresh-interval", interval=3000, n_intervals=0),
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
    if engine:
        try:
            coins = engine.get_available_coins() or []
        except Exception:
            pass
    if not coins:
        coins = DEFAULT_COINS.get(exchange, ["BTC"])
    return [{"label": c, "value": c} for c in coins], coins[:min(5, len(coins))]


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
    State("conn-store",      "data"),
    prevent_initial_call=True,
)
def handle_controls(conn, disc, start, stop, emerg, exchange, store):
    btn = ctx.triggered_id
    store = store or {}

    if engine is None:
        return "Engine non chargé — vérifier les logs.", "danger", store

    if btn == "connect-btn":
        ex_key = exchange.lower().replace(" ", "_").replace("_futures", "")
        ok = engine.connect(ex_key, paper=True)
        if ok:
            store["connected"] = True
            store["exchange"] = exchange
            return f"Connecté à {exchange} (paper mode)", "success", store
        return "Connexion échouée.", "danger", store

    if btn == "disconnect-btn":
        engine.disconnect()
        store["connected"] = False
        return "Déconnecté.", "secondary", store

    if btn == "start-btn":
        if engine.start():
            return "Toutes les stratégies démarrées.", "success", store
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
    prevent_initial_call=True,
)
def handle_per_strategy(starts, stops):
    if engine is None:
        return "Engine non chargé.", "danger"
    trig = ctx.triggered_id
    if not isinstance(trig, dict):
        return dash.no_update, dash.no_update
    name   = trig.get("index")
    action = trig.get("type")
    if action == "start-strat-btn":
        ok = engine.start_strategy(name)
        return (f"Stratégie démarrée : {name}", "success") if ok else (f"Impossible de démarrer {name}", "warning")
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
    # Portfolio metrics
    Output("portfolio-metrics",   "children"),
    Output("pnl-chart",           "figure"),
    Output("alloc-pie",           "figure"),
    Output("positions-view",      "children"),
    # Config dump
    Output("config-dump",         "children"),
    # Signals
    Output("signal-feed",         "children"),
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
    warmup_labels  = ["—"] * n_strats
    status_pills   = [_pill(False)] * n_strats
    hint_txt       = "Configure et clique Lancer dans le sidebar"
    global_status  = "IDLE"
    global_mode    = "paper"
    n_active       = 0
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
    signal_txt = "Aucun signal — stratégies en attente."

    if engine is not None:
        status = engine.get_status()
        strats  = status.get("strategies", {})
        runtime = status.get("strategy_runtime", {})
        n_active = sum(1 for r in runtime.values() if r.get("active"))
        global_status = "ACTIF" if status.get("running") else "IDLE"

        # Warmup bars
        for i, name in enumerate(strategy_names):
            rt  = runtime.get(name, {})
            req = max(1, int(rt.get("warmup_required_sec", 1)))
            buf = int(rt.get("buffered_sec", 0))
            pct = min(100, int(buf / req * 100))
            h_buf = buf // 3600; m_buf = (buf % 3600) // 60
            h_req = req // 3600; m_req = (req % 3600) // 60
            warmup_labels[i] = html.Div([
                html.Span(f"{h_buf:02d}h{m_buf:02d}m / {h_req:02d}h{m_req:02d}m  ({pct}%)",
                          style={"fontSize": ".7rem", "color": "#888"}),
                _warmup_bar(pct),
            ])
            status_pills[i] = _pill(rt.get("active", False))

        hint_txt = (
            f"{n_active} stratégie(s) active(s) — scan en cours…"
            if n_active > 0 else
            "Configure et clique Lancer dans le sidebar"
        )

        # Portfolio
        total_pnl  = status.get("total_pnl", 0.0)
        n_trades   = sum(s.get("n_trades", 0) for s in strats.values())
        pnls_all   = [p for s in strats.values() for p in ([s.get("total_pnl", 0)] if s.get("n_trades", 0) else [])]
        win_rate   = "—"

        portfolio_metrics = _metric_cards([
            ("Capital",      f"${capital + total_pnl:,.0f}", ""),
            ("Réalisé PnL",  f"${total_pnl:+.2f}",          "success" if total_pnl >= 0 else "danger"),
            ("Non-réalisé",  "$0.00",                        ""),
            ("Trades",       str(n_trades),                  ""),
            ("Win rate",     win_rate,                       ""),
            ("Actifs",       str(n_active),                  "primary" if n_active else ""),
        ])

        # PnL chart
        pnl_hist = engine.get_current_pnl() or []
        if pnl_hist:
            eq = capital + np.cumsum(pnl_hist)
            fig_pnl = go.Figure(data=[go.Scatter(
                y=np.concatenate([[capital], eq]),
                mode="lines", fill="tozeroy",
                line=dict(color="#2a9fd6", width=2),
                fillcolor="rgba(42,159,214,0.06)",
            )])
        else:
            names_bar = list(strats.keys())
            warmup_pcts = []
            for n in names_bar:
                rt2 = runtime.get(n, {})
                req2 = max(1, int(rt2.get("warmup_required_sec", 1)))
                buf2 = int(rt2.get("buffered_sec", 0))
                warmup_pcts.append(min(100, int(buf2 / req2 * 100)))
            fig_pnl = go.Figure(data=[go.Bar(
                x=names_bar or ["—"], y=warmup_pcts or [0],
                marker_color="#2a9fd6",
            )])
            fig_pnl.add_annotation(
                text="Aucun trade exécuté — warmup en cours",
                xref="paper", yref="paper", x=0.5, y=1.05,
                showarrow=False, font=dict(color="#555", size=11),
            )

        fig_pnl.update_layout(
            template="plotly_dark", height=280,
            paper_bgcolor="#111418", plot_bgcolor="#111418",
            margin=dict(l=50, r=20, t=40, b=30),
            font=dict(family="Roboto Mono, Courier New", color="#666"),
            title=dict(text="Cumulative PnL ($)", font=dict(size=12, color="#2a9fd6"), x=0),
            yaxis=dict(gridcolor="#1a2530"),
            xaxis=dict(gridcolor="#1a2530"),
        )

        # Pie
        cap_vals  = [s.get("capital", 0) for s in strats.values()]
        pie_names = [STRATEGY_META.get(n, {}).get("label", n) for n in strats.keys()]
        fig_pie = go.Figure(data=[go.Pie(
            labels=pie_names or ["—"],
            values=cap_vals or [1],
            hole=0.45,
            marker_colors=["#2a9fd6", "#77b300", "#ff7518", "#d500f9"],
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
                side = str(pos.get("side", "?")).upper()
                col = "#77b300" if side == "LONG" else "#cc0000"
                pos_rows.append(html.Div([
                    html.Span(sn, style={"flex": "2", "color": "#bbb"}),
                    html.Span(side, style={"flex": "1", "color": col}),
                    html.Span(f"${pos.get('entry_price', 0):,.2f}", style={"flex": "2"}),
                    html.Span(f"${pos.get('size', 0):,.0f}", style={"flex": "1"}),
                    html.Span(f"${pos.get('unrealized_pnl', 0):+.2f}", style={"flex": "1"}),
                ], style={"display": "flex", "gap": "8px", "padding": "7px 0",
                          "fontSize": ".78rem", "borderBottom": "1px solid #1a2530"}))
            pos_view = html.Div([
                html.Div([
                    html.Span("Stratégie",    style={"flex": "2"}),
                    html.Span("Side",          style={"flex": "1"}),
                    html.Span("Entry",         style={"flex": "2"}),
                    html.Span("Taille",        style={"flex": "1"}),
                    html.Span("Unrealized",    style={"flex": "1"}),
                ], style={"display": "flex", "gap": "8px", "padding": "7px 0",
                          "fontSize": ".65rem", "color": "#2a9fd6",
                          "letterSpacing": ".1em", "textTransform": "uppercase"}),
                html.Div(pos_rows),
            ])
        else:
            pos_view = html.Span("Aucune position ouverte.",
                                 style={"color": "#555", "fontSize": ".78rem"})

    badge_color  = "success" if n_active > 0 else "secondary"
    badge_actifs = f"{n_active} actif(s)"
    badge_live   = "PAPER"

    return (
        global_status, global_mode,
        badge_actifs, badge_live, badge_color,
        now, ticker_txt, src_txt,
        warmup_labels, status_pills, hint_txt,
        portfolio_metrics, fig_pnl, fig_pie, pos_view,
        config_txt,
        signal_txt,
    )


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
    app.run(debug=True, port=8050, host="0.0.0.0")
