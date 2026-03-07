"""
Agarthai — Live Trading Dashboard (Dash)
==========================================
python gui/live_app.py → http://localhost:8050

Interactive: Start/Stop, Connect/Disconnect, coin picker,
real-time PnL, strategy status, emergency stop.
"""

import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import sys, time, json
from pathlib import Path
from datetime import datetime
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── App Setup ────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],  # Dark theme
    title="Agarthai — Live Trading",
)

# ── Global Engine ────────────────────────────────────────────────────────
try:
    from live.engine import LiveEngine
    engine = LiveEngine()
except Exception as e:
    engine = None
    print(f"Warning: LiveEngine not available: {e}")

EXCHANGES = ['Hyperliquid', 'Bitget Futures']
COINS = {
    'Hyperliquid': ['BTC', 'ETH', 'SOL', 'DOGE', 'ARB', 'OP'],
    'Bitget Futures': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT'],
}

# ── Layout ───────────────────────────────────────────────────────────────
app.layout = dbc.Container([

    # Header
    dbc.Row([
        dbc.Col(html.H2("🔴 Agarthai — Live Trading", className="text-light"), width=6),
        dbc.Col(html.Div(id='clock', className="text-end text-muted"), width=3),
        dbc.Col(html.Div(id='connection-badge'), width=3),
    ], className="my-3"),

    # Controls Row
    dbc.Row([
        dbc.Col([
            dbc.Label("Exchange", className="text-light"),
            dcc.Dropdown(id='exchange-select', options=[{'label': e, 'value': e} for e in EXCHANGES],
                         value='Hyperliquid', className="mb-2"),
        ], width=2),
        dbc.Col([
            dbc.Label("Coin", className="text-light"),
            dcc.Dropdown(id='coin-select', value='BTC', className="mb-2"),
        ], width=2),
        dbc.Col([
            dbc.Button("🔌 Connect", id='connect-btn', color="primary", className="me-2 mt-4"),
            dbc.Button("⛔ Disconnect", id='disconnect-btn', color="secondary", className="mt-4"),
        ], width=3),
        dbc.Col([
            dbc.Button("▶️ START", id='start-btn', color="success", size="lg", className="me-2 mt-4"),
            dbc.Button("⏹️ STOP", id='stop-btn', color="warning", size="lg", className="me-2 mt-4"),
            dbc.Button("🚨 EMERGENCY", id='emergency-btn', color="danger", size="lg", className="mt-4"),
        ], width=5),
    ], className="mb-3"),

    # Status bar
    dbc.Alert(id='status-alert', color="info", className="mb-3"),

    # Metrics cards
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Unrealized PnL", className="text-muted"),
            html.H3(id='unrealized-pnl', className="text-success"),
        ]), color="dark"), width=2),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Realized PnL", className="text-muted"),
            html.H3(id='realized-pnl', className="text-info"),
        ]), color="dark"), width=2),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("# Trades", className="text-muted"),
            html.H3(id='n-trades-live'),
        ]), color="dark"), width=2),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Win Rate", className="text-muted"),
            html.H3(id='win-rate-live'),
        ]), color="dark"), width=2),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Capital", className="text-muted"),
            html.H3(id='capital-live'),
        ]), color="dark"), width=2),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Strategies", className="text-muted"),
            html.H3(id='active-strats'),
        ]), color="dark"), width=2),
    ], className="mb-3"),

    # Charts
    dbc.Row([
        dbc.Col(dcc.Graph(id='pnl-chart'), width=8),
        dbc.Col(dcc.Graph(id='strategy-pie'), width=4),
    ]),

    # Strategy status table
    dbc.Row([
        dbc.Col([
            html.H5("Strategy Status", className="text-light mt-3"),
            html.Div(id='strategy-table'),
        ]),
    ]),

    # Auto-refresh
    dcc.Interval(id='refresh-interval', interval=3000, n_intervals=0),
    dcc.Store(id='engine-state', data={'running': False, 'connected': False}),

], fluid=True, className="bg-dark")


# ── Callbacks ────────────────────────────────────────────────────────────

@app.callback(
    Output('coin-select', 'options'),
    Input('exchange-select', 'value'),
)
def update_coins(exchange):
    coins = COINS.get(exchange, ['BTC'])
    return [{'label': c, 'value': c} for c in coins]


@app.callback(
    [Output('status-alert', 'children'),
     Output('status-alert', 'color'),
     Output('connection-badge', 'children')],
    [Input('connect-btn', 'n_clicks'),
     Input('disconnect-btn', 'n_clicks'),
     Input('start-btn', 'n_clicks'),
     Input('stop-btn', 'n_clicks'),
     Input('emergency-btn', 'n_clicks')],
    [State('exchange-select', 'value')],
    prevent_initial_call=True,
)
def handle_buttons(conn, disc, start, stop, emerg, exchange):
    button = ctx.triggered_id
    badge = html.Span("🔴 Disconnected", className="badge bg-danger")

    if engine is None:
        return "Engine not loaded", "danger", badge

    if button == 'connect-btn':
        ok = engine.connect(exchange.lower().replace(' ', '_'), paper=True)
        if ok:
            badge = html.Span("🟢 Connected (Paper)", className="badge bg-success")
            return f"Connected to {exchange} (Paper Mode)", "success", badge
        return "Connection failed", "danger", badge

    elif button == 'disconnect-btn':
        engine.disconnect()
        return "Disconnected", "secondary", badge

    elif button == 'start-btn':
        if engine.start():
            badge = html.Span("🟢 TRADING", className="badge bg-success")
            return "⚡ Strategy ACTIVE — Scanning for signals...", "success", badge
        return "Cannot start: not connected", "warning", badge

    elif button == 'stop-btn':
        engine.stop()
        badge = html.Span("🟡 Idle", className="badge bg-warning")
        return "Strategy stopped", "warning", badge

    elif button == 'emergency-btn':
        engine.emergency_stop()
        return "🚨 EMERGENCY STOP — All positions closed", "danger", badge

    return "Ready", "info", badge


@app.callback(
    [Output('unrealized-pnl', 'children'),
     Output('realized-pnl', 'children'),
     Output('n-trades-live', 'children'),
     Output('win-rate-live', 'children'),
     Output('capital-live', 'children'),
     Output('active-strats', 'children'),
     Output('pnl-chart', 'figure'),
     Output('strategy-pie', 'figure'),
     Output('strategy-table', 'children'),
     Output('clock', 'children')],
    Input('refresh-interval', 'n_intervals'),
)
def refresh_dashboard(n):
    now = datetime.now().strftime('%H:%M:%S')

    if engine is None:
        empty = go.Figure(layout=dict(template='plotly_dark', height=300))
        return "$0", "$0", "0", "0%", "$1,500", "0", empty, empty, "", now

    status = engine.get_status()
    n_trades = sum(s.get('n_trades', 0) for s in status.get('strategies', {}).values())
    total_pnl = status.get('total_pnl', 0)

    # PnL chart
    pnl_data = engine.get_current_pnl()
    if not pnl_data:
        pnl_data = [0]
    fig_pnl = go.Figure(data=[go.Scatter(
        y=np.cumsum(pnl_data) if len(pnl_data) > 1 else [0],
        mode='lines', fill='tozeroy',
        line=dict(color='#00e676', width=2),
        fillcolor='rgba(0,230,118,0.1)',
    )])
    fig_pnl.update_layout(template='plotly_dark', height=350,
                          title='Cumulative PnL ($)', margin=dict(l=40, r=20, t=40, b=30))

    # Strategy allocation pie
    strats = status.get('strategies', {})
    fig_pie = go.Figure(data=[go.Pie(
        labels=list(strats.keys()) or ['No strategies'],
        values=[s.get('capital', 0) for s in strats.values()] or [1],
        hole=0.4,
        marker_colors=['#00e676', '#2979ff', '#ff9100', '#d500f9'],
    )])
    fig_pie.update_layout(template='plotly_dark', height=350,
                          title='Capital Allocation')

    # Strategy table
    rows = []
    for name, s in strats.items():
        rows.append(dbc.Row([
            dbc.Col(html.Strong(name), width=3),
            dbc.Col(f"${s.get('capital', 0):.0f}", width=2),
            dbc.Col(f"{s.get('n_trades', 0)} trades", width=2),
            dbc.Col(f"${s.get('total_pnl', 0):.2f}", width=2),
            dbc.Col(html.Span(s.get('mode', '?'), className="badge bg-info"), width=3),
        ], className="mb-1 text-light"))

    return (
        f"${0:.2f}",
        f"${total_pnl:.2f}",
        str(n_trades),
        f"{0:.0%}",
        f"${1500 + total_pnl:,.0f}",
        str(len(strats)),
        fig_pnl, fig_pie,
        html.Div(rows) if rows else "No strategies loaded",
        now,
    )


if __name__ == '__main__':
    app.run(debug=True, port=8050)
