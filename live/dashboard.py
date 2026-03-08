"""
Agarthai — Live Trading Dashboard (Dash)
==========================================
python live/dashboard.py → http://localhost:8050

Interactive: per-strategy start/stop, coin universe selection,
warmup progress, CSV export, positions monitor.
"""

import json
import sys
import dash
from datetime import datetime
from pathlib import Path
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import sys, time, json
from pathlib import Path
from datetime import datetime
import numpy as np
import plotly.graph_objs as go
from dash import Input, Output, State, ctx, dcc, html
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── App Setup ────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="Agarthai — Live Trading",
)

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body { background: #000000 !important; }
            .bg-dark { background: #000000 !important; }
            .card, .alert { border-color: #2a2a2a !important; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''
# ── Global Engine ────────────────────────────────────────────────────────
try:
    from live.engine import LiveEngine

    engine = LiveEngine()
except Exception as e:
    engine = None
    print(f"Warning: LiveEngine not available: {e}")

EXCHANGES = ['Hyperliquid', 'Bitget Futures']
DEFAULT_COINS = {
    'Hyperliquid': ['BTC', 'ETH', 'SOL', 'DOGE', 'ARB', 'OP'],
    'Bitget Futures': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT'],
}

def _strategy_controls_layout(strategy_names):
    rows = []
    for name in strategy_names:
        rows.append(
            dbc.Row(
                [
                    dbc.Col(html.Strong(name, className='text-light'), width=3),
                    dbc.Col(
                        dbc.Button(
                            '▶️ START',
                            id={'type': 'start-strat-btn', 'index': name},
                            color='success',
                            size='sm',
                            className='me-2',
                        ),
                        width=2,
                    ),
                    dbc.Col(
                        dbc.Button(
                            '⏹️ STOP',
                            id={'type': 'stop-strat-btn', 'index': name},
                            color='warning',
                            size='sm',
                        ),
                        width=2,
                    ),
                    dbc.Col(
                        dbc.Progress(
                            value=0,
                            striped=True,
                            animated=True,
                            id={'type': 'warmup-progress', 'index': name},
                            style={'height': '20px'},
                            label='warmup 0%',
                        ),
                        width=5,
                    ),
                ],
                className='mb-2 align-items-center',
            )
        )
    return rows
# ── Layout ───────────────────────────────────────────────────────────────
strategy_names = list(engine.get_strategy_names()) if engine else []

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(html.H2("🔴 Agarthai — Live Trading (Paper)", className="text-light"), width=6),
                dbc.Col(html.Div(id='clock', className="text-end text-muted"), width=3),
                dbc.Col(html.Div(id='connection-badge'), width=3),
            ],
            className="my-3",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label("Exchange", className="text-light"),
                        dcc.Dropdown(
                            id='exchange-select',
                            options=[{'label': e, 'value': e} for e in EXCHANGES],
                            value='Hyperliquid',
                            className="mb-2",
                        ),
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        dbc.Label("Univers coins (multi-select)", className="text-light"),
                        dcc.Dropdown(id='coin-select', multi=True, className="mb-2"),
                    ],
                    width=5,
                ),
                dbc.Col(
                    [
                        dbc.Button("🔌 Connect", id='connect-btn', color="primary", className="me-2 mt-4"),
                        dbc.Button("⛔ Disconnect", id='disconnect-btn', color="secondary", className="mt-4"),
                    ],
                    width=2,
                ),
                dbc.Col(
                    [
                        dbc.Button("▶️ START ALL", id='start-btn', color="success", className="me-2 mt-4"),
                        dbc.Button("⏹️ STOP ALL", id='stop-btn', color="warning", className="me-2 mt-4"),
                        dbc.Button("🚨 EMERGENCY", id='emergency-btn', color="danger", className="mt-4"),
                    ],
                    width=2,
                ),
            ],
            className="mb-3",
        ),
        dbc.Alert(id='status-alert', color="info", className="mb-3"),
        dbc.Row(
            [
                dbc.Col(dbc.Card(dbc.CardBody([html.H6("Unrealized PnL", className="text-muted"), html.H3(id='unrealized-pnl')]), color="dark"), width=2),
                dbc.Col(dbc.Card(dbc.CardBody([html.H6("Realized PnL", className="text-muted"), html.H3(id='realized-pnl')]), color="dark"), width=2),
                dbc.Col(dbc.Card(dbc.CardBody([html.H6("# Trades", className="text-muted"), html.H3(id='n-trades-live')]), color="dark"), width=2),
                dbc.Col(dbc.Card(dbc.CardBody([html.H6("Win Rate", className="text-muted"), html.H3(id='win-rate-live')]), color="dark"), width=2),
                dbc.Col(dbc.Card(dbc.CardBody([html.H6("Capital", className="text-muted"), html.H3(id='capital-live')]), color="dark"), width=2),
                dbc.Col(dbc.Card(dbc.CardBody([html.H6("Active strategies", className="text-muted"), html.H3(id='active-strats')]), color="dark"), width=2),
            ],
            className="mb-3",
        ),
        dbc.Row([dbc.Col(dcc.Graph(id='pnl-chart'), width=8), dbc.Col(dcc.Graph(id='strategy-pie'), width=4)]),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H5("Controls par stratégie + warmup", className="text-light mt-3"),
                        html.Div(id='strategy-controls', children=_strategy_controls_layout(strategy_names)),
                    ],
                    width=12,
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H5("Positions live par stratégie", className="text-light mt-3"),
                        html.Div(id='positions-table'),
                    ],
                    width=12,
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H5("Strategy status", className="text-light mt-3"),
                        html.Div(id='strategy-table'),
                    ],
                    width=9,
                ),
                dbc.Col(
                    [
                        html.H5("Export", className="text-light mt-3"),
                        dbc.Checklist(
                            options=[{'label': ' Keep results in CSV', 'value': 'csv'}],
                            value=['csv'],
                            id='csv-toggle',
                            switch=True,
                            className='text-light',
                        ),
                        dbc.Button("💾 Export now", id='export-csv-btn', color='info', className='mt-2'),
                        dcc.Download(id='download-trades-csv'),
                    ],
                    width=3,
                ),
            ]
        ),
        dcc.Interval(id='refresh-interval', interval=3000, n_intervals=0),
        dcc.Store(id='selected-coins-store', data=[]),
    ],
    fluid=True,
    className="bg-dark",
    style={'backgroundColor': '#000000', 'minHeight': '100vh'},
)
# ── Callbacks ────────────────────────────────────────────────────────────

@app.callback(
    Output('coin-select', 'options'),
    Output('coin-select', 'value'),
    Input('exchange-select', 'value'),
)
def update_coins(exchange):
    coins = []
    if engine is not None:
        try:
            status = engine.get_status()
            coins = status.get('available_coins', []) or []
        except Exception:
            coins = []
    if not coins:
        coins = DEFAULT_COINS.get(exchange, ['BTC'])
    return [{'label': c, 'value': c} for c in coins], coins[: min(5, len(coins))]


@app.callback(
    [Output('status-alert', 'children'), Output('status-alert', 'color'), Output('connection-badge', 'children')],
    [
        Input('connect-btn', 'n_clicks'),
        Input('disconnect-btn', 'n_clicks'),
        Input('start-btn', 'n_clicks'),
        Input('stop-btn', 'n_clicks'),
        Input('emergency-btn', 'n_clicks'),
    ],
    [State('exchange-select', 'value')],
    prevent_initial_call=True,
)
def handle_main_buttons(conn, disc, start, stop, emerg, exchange):
    button = ctx.triggered_id
    badge = html.Span("🔴 Disconnected", className="badge bg-danger")

    if engine is None:
        return "Engine not loaded", "danger", badge

    if button == 'connect-btn':
        ex_name = exchange.lower().replace(' ', '_')
        ok = engine.connect(ex_name, paper=True)
        if ok:
            badge = html.Span("🟢 Connected (Paper)", className="badge bg-success")
            return f"Connected to {exchange} (Paper Mode)", "success", badge
        return "Connection failed", "danger", badge

    if button == 'disconnect-btn':
        engine.disconnect()
        return "Disconnected", "secondary", badge

    if button == 'start-btn':
        if engine.start():
            badge = html.Span("🟢 TRADING", className="badge bg-success")
            #return "⚡ Strategy ACTIVE — Scanning for signals...", "success", badge
            return "⚡ All strategies STARTED", "success", badge
        return "Cannot start: not connected", "warning", badge

    if button == 'stop-btn':
        engine.stop()
        badge = html.Span("🟡 Idle", className="badge bg-warning")
        return "All strategies STOPPED", "warning", badge

    if button == 'emergency-btn':
        engine.emergency_stop()
        return "🚨 EMERGENCY STOP — All positions closed", "danger", badge

    return "Ready", "info", badge


@app.callback(
    Output('status-alert', 'children', allow_duplicate=True),
    Output('status-alert', 'color', allow_duplicate=True),
    Input({'type': 'start-strat-btn', 'index': dash.ALL}, 'n_clicks'),
    Input({'type': 'stop-strat-btn', 'index': dash.ALL}, 'n_clicks'),
    prevent_initial_call=True,
)
def handle_strategy_buttons(start_clicks, stop_clicks):
    if engine is None:
        return "Engine not loaded", "danger"

    trig = ctx.triggered_id
    if not trig or not isinstance(trig, dict):
        return dash.no_update, dash.no_update

    strat = trig.get('index')
    action = trig.get('type')
    if action == 'start-strat-btn':
        ok = engine.start_strategy(strat)
        return (f"Strategy STARTED: {strat}", "success") if ok else (f"Cannot start {strat}", "warning")
    if action == 'stop-strat-btn':
        ok = engine.stop_strategy(strat)
        return (f"Strategy STOPPED: {strat}", "warning") if ok else (f"Cannot stop {strat}", "danger")

    return dash.no_update, dash.no_update


@app.callback(
    [   Output('unrealized-pnl', 'children'),
        Output('realized-pnl', 'children'),
        Output('n-trades-live', 'children'),
        Output('win-rate-live', 'children'),
        Output('capital-live', 'children'),
        Output('active-strats', 'children'),
        Output('pnl-chart', 'figure'),
        Output('strategy-pie', 'figure'),
        Output('strategy-table', 'children'),
        Output('positions-table', 'children'),
        Output({'type': 'warmup-progress', 'index': dash.ALL}, 'value'),
        Output({'type': 'warmup-progress', 'index': dash.ALL}, 'label'),
        Output('clock', 'children'),
    ],
    Input('refresh-interval', 'n_intervals'),
)
def refresh_dashboard(_):
    now = datetime.now().strftime('%H:%M:%S')
    empty = go.Figure(layout=dict(template='plotly_dark', height=300))


    if engine is None:
        return "$0", "$0", "0", "0%", "$1,500", "0", empty, empty, "", "", [], [], now

    status = engine.get_status()
    strats = status.get('strategies', {})
    runtime = status.get('strategy_runtime', {})

    n_trades = sum(s.get('n_trades', 0) for s in strats.values())
    total_pnl = status.get('total_pnl', 0.0)
    active_count = sum(1 for v in runtime.values() if v.get('active'))

    # # PnL chart
    # pnl_data = engine.get_current_pnl() or [0]
    # fig_pnl = go.Figure(
    #     data=[go.Scatter(y=np.cumsum(pnl_data) if len(pnl_data) > 1 else [0], mode='lines', fill='tozeroy', line=dict(color='#00e676', width=2))]
    # )
    # fig_pnl.update_layout(template='plotly_dark', height=350, title='Cumulative PnL ($)', margin=dict(l=40, r=20, t=40, b=30))
    # PnL / warmup chart
    pnl_data = engine.get_current_pnl() or []
    if pnl_data:
        fig_pnl = go.Figure(
            data=[go.Scatter(y=np.cumsum(pnl_data), mode='lines', fill='tozeroy', line=dict(color='#00e676', width=2))]
        )
        fig_pnl.update_layout(template='plotly_dark', height=350, title='Cumulative PnL ($)', margin=dict(l=40, r=20, t=40, b=30))
    else:
        names = list(strats.keys())
        warmup_pct = []
        for n in names:
            rt = runtime.get(n, {})
            req = max(1, int(rt.get('warmup_required_sec', 1)))
            buf = int(rt.get('buffered_sec', 0))
            warmup_pct.append(min(100, int((buf / req) * 100)))
        fig_pnl = go.Figure(data=[go.Bar(x=names or ['No strategy'], y=warmup_pct or [0], marker_color='#2979ff')])
        fig_pnl.update_layout(template='plotly_dark', height=350, title='Warmup progress by strategy (%)', yaxis=dict(range=[0, 100]), margin=dict(l=40, r=20, t=40, b=30))
        fig_pnl.add_annotation(text='No executed trades yet: strategies warming up or waiting for signals.', xref='paper', yref='paper', x=0.5, y=1.08, showarrow=False, font=dict(color='#ffea00'))
    # Pie
    fig_pie = go.Figure(
        data=[
            go.Pie(
                labels=list(strats.keys()) or ['No strategies'],
                values=[s.get('capital', 0) for s in strats.values()] or [1],
                hole=0.4,
                marker_colors=['#00e676', '#2979ff', '#ff9100', '#d500f9', '#f44336'],
            )
        ]
    )
    fig_pie.update_layout(template='plotly_dark', height=350, title='Capital Allocation')

    # Strategy status table
    s_rows = []
    warmup_values = []
    warmup_labels = []

    for name, s in strats.items():
        rt = runtime.get(name, {})
        required = max(1, int(rt.get('warmup_required_sec', 1)))
        buffered = int(rt.get('buffered_sec', 0))
        p = int(min(100, (buffered / required) * 100))
        warmup_values.append(p)
        warmup_labels.append(f'warmup {buffered}/{required}s ({p}%)')

        s_rows.append(
            dbc.Row(
                [
                    dbc.Col(html.Strong(name), width=2),
                    dbc.Col(f"${s.get('capital', 0):.0f}", width=2),
                    dbc.Col(f"{s.get('n_trades', 0)} trades", width=2),
                    dbc.Col(f"${s.get('total_pnl', 0):.2f}", width=2),
                    dbc.Col(f"WR {s.get('win_rate', 0):.0%}", width=2),
                    dbc.Col(html.Span('ACTIVE' if rt.get('active') else 'IDLE', className=f"badge bg-{'success' if rt.get('active') else 'secondary'}"), width=2),
                ],
                className='mb-1 text-light',
            )
        )

    # Positions table by strategy
    positions = status.get('positions', {}) or {}
    if positions:
        pos_rows = []
        for strat_name, pos in positions.items():
            pos_rows.append(
                dbc.Row(
                    [
                        dbc.Col(html.Strong(strat_name), width=2),
                        dbc.Col(str(pos.get('symbol', '-')), width=2),
                        dbc.Col(str(pos.get('side', '-')), width=2),
                        dbc.Col(f"{pos.get('size', 0)}", width=2),
                        dbc.Col(f"{pos.get('entry_price', 0)}", width=2),
                        dbc.Col(f"{pos.get('unrealized_pnl', 0)}", width=2),
                    ],
                    className='mb-1 text-light',
                )
            )
        positions_view = html.Div(pos_rows)
    else:
        positions_view = html.Span('No open positions per strategy.', className='text-muted')

    # unrealized not wired yet in engine: keep 0 for now
    return (
        f"${0:.2f}",
        f"${total_pnl:.2f}",
        str(n_trades),
        f"{0:.0%}",
        f"${1500 + total_pnl:,.0f}",
        str(active_count),
        fig_pnl,
        fig_pie,
        html.Div(s_rows) if s_rows else 'No strategies loaded',
        positions_view,
        warmup_values,
        warmup_labels,
        now,
    )

@app.callback(
    Output('download-trades-csv', 'data'),
    Input('export-csv-btn', 'n_clicks'),
    State('csv-toggle', 'value'),
    prevent_initial_call=True,
)
def export_csv(_, csv_toggle):
    if engine is None or 'csv' not in (csv_toggle or []):
        return dash.no_update

    rows = []
    for name, strat_data in engine.strategies.items():
        strategy = strat_data['instance']
        for tr in strategy.trade_history:
            rows.append(
                {
                    'strategy': name,
                    'timestamp': tr.signal.timestamp,
                    'entry_price': tr.entry_price,
                    'exit_price': tr.exit_price,
                    'pnl_usd': tr.pnl_usd,
                    'pnl_pct': tr.pnl_pct,
                    'fees_usd': tr.fees_usd,
                    'slippage_usd': tr.slippage_usd,
                    'exit_reason': tr.exit_reason,
                    'duration_sec': tr.duration_sec,
                    'position_usd': tr.position_usd,
                }
            )

    if not rows:
        rows = [{'info': 'No trades yet'}]

    df = pd.DataFrame(rows)
    Path('results').mkdir(exist_ok=True)
    path = Path('results') / f"live_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(path, index=False)
    return dcc.send_file(str(path))

if __name__ == '__main__':
    app.run(debug=True, port=8050)
