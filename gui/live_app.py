"""
Agarthai — Live Paper Trading Terminal
=======================================
streamlit run gui/live_app.py

Features:
  - Connect to Hyperliquid / Bitget (paper mode)
  - Real-time price feed (Binance public REST, refreshed each cycle)
  - Start/Stop strategy per-strategy
  - Coin selection from exchange
  - Live PnL tracking with Almgren-Chriss slippage
  - Unrealized PnL display
  - Position monitor
  - Cyborg dark theme
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time, sys
from pathlib import Path
from datetime import datetime
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AGARTHAI // LIVE",
    page_icon="⬡",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# CYBORG THEME — dark terminal / futuristic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap" rel="stylesheet">
<style>
/* ── Base ──────────────────────────────────────────────────────── */
*, *::before, *::after {
    font-family: 'Share Tech Mono', 'Courier New', monospace !important;
    box-sizing: border-box;
}
html, body, .stApp {
    background-color: #060a0e !important;
    color: #8aff80 !important;
}
/* scanline overlay */
.stApp::after {
    content: '';
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 3px,
        rgba(0,255,65,0.018) 3px,
        rgba(0,255,65,0.018) 4px
    );
    pointer-events: none;
    z-index: 9000;
}

/* ── Sidebar ────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #080d11 !important;
    border-right: 1px solid #00ff4133 !important;
}
section[data-testid="stSidebar"] * { color: #8aff80 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stNumberInput label { color: #00e5ff !important; }
section[data-testid="stSidebar"] [data-baseweb="select"] {
    background: #0a1018 !important;
    border: 1px solid #00ff4155 !important;
}
section[data-testid="stSidebar"] input {
    background: #0a1018 !important;
    border: 1px solid #00ff4155 !important;
    color: #8aff80 !important;
}

/* ── Buttons ─────────────────────────────────────────────────────── */
.stButton > button {
    background: transparent !important;
    border: 1px solid #00ff4166 !important;
    color: #00ff41 !important;
    font-family: 'Share Tech Mono', monospace !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    background: #00ff4114 !important;
    border-color: #00ff41 !important;
    box-shadow: 0 0 10px #00ff4166;
}
.stButton > button[kind="primary"] {
    border-color: #00e5ff !important;
    color: #00e5ff !important;
}
.stButton > button[kind="primary"]:hover {
    background: #00e5ff14 !important;
    box-shadow: 0 0 10px #00e5ff66;
}

/* ── Metrics ─────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: #0a1018 !important;
    border: 1px solid #00ff4122 !important;
    border-left: 3px solid #00ff41 !important;
    border-radius: 0 !important;
    padding: 12px 16px !important;
}
[data-testid="metric-container"] label {
    color: #00e5ff !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #8aff80 !important;
    font-size: 1.3rem !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    color: #b0bec5 !important;
}

/* ── Divider ─────────────────────────────────────────────────────── */
hr { border-color: #00ff4133 !important; }

/* ── Headers ─────────────────────────────────────────────────────── */
h1, h2, h3 {
    color: #00e5ff !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border-bottom: 1px solid #00ff4122;
    padding-bottom: 4px;
}

/* ── Dataframe ───────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #00ff4133 !important;
}
[data-testid="stDataFrame"] th {
    background: #0a1018 !important;
    color: #00e5ff !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
[data-testid="stDataFrame"] td { color: #8aff80 !important; }

/* ── Info / Caption ──────────────────────────────────────────────── */
.stInfo { background: #0a1018 !important; border-color: #00e5ff44 !important; color: #b0bec5 !important; }
.stCaption, .stCaption p { color: #445566 !important; font-size: 0.7rem !important; }

/* ── Expander ─────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #0a1018 !important;
    border: 1px solid #00ff4133 !important;
}
[data-testid="stExpander"] summary { color: #00e5ff !important; }

/* ── JSON ─────────────────────────────────────────────────────────── */
[data-testid="stJson"] { background: #050810 !important; color: #8aff80 !important; }

/* ── Toast ────────────────────────────────────────────────────────── */
.stToast { background: #0a1018 !important; border: 1px solid #00ff41 !important; color: #8aff80 !important; }

/* ── Spinner ──────────────────────────────────────────────────────── */
.stSpinner > div { border-color: #00ff41 transparent transparent transparent !important; }

/* ── Live dots ────────────────────────────────────────────────────── */
.sig-dot {
    display: inline-block; width: 8px; height: 8px;
    border-radius: 50%; margin-right: 6px;
    animation: pulse 1.4s ease-in-out infinite;
}
.sig-dot.green  { background: #00ff41; box-shadow: 0 0 6px #00ff41; }
.sig-dot.red    { background: #ff1744; box-shadow: 0 0 6px #ff1744; }
.sig-dot.yellow { background: #ffea00; box-shadow: 0 0 6px #ffea00; }
.sig-dot.cyan   { background: #00e5ff; box-shadow: 0 0 6px #00e5ff; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.25} }

/* ── Position card ────────────────────────────────────────────────── */
.pos-card {
    background: #0a1018;
    border: 1px solid #2979ff44;
    border-left: 3px solid #2979ff;
    padding: 16px 20px;
    margin: 8px 0;
}
.pos-card.long  { border-left-color: #00ff41; }
.pos-card.short { border-left-color: #ff1744; }

/* ── Warmup bar ───────────────────────────────────────────────────── */
.warmup-bar-bg {
    background: #0d1620; border: 1px solid #00ff4133; height: 6px;
    border-radius: 0; margin: 4px 0 12px;
}
.warmup-bar-fg {
    background: linear-gradient(90deg, #00ff41, #00e5ff);
    height: 100%;
    box-shadow: 0 0 6px #00ff41;
    transition: width 0.5s ease;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════
def _init_state():
    defaults = {
        'strategy_running': False,
        'connected': False,
        'paper_trades': [],
        'paper_capital': 1500.0,
        'current_position': None,
        'price_history': [],
        'last_refresh': time.time(),
        'live_price': None,
        'price_source': 'static',
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ═══════════════════════════════════════════════════════════════════════════
# PRICE FEED HELPER
# ═══════════════════════════════════════════════════════════════════════════

def get_live_price(coin: str) -> tuple:
    """Fetch real price; returns (price, source_label)."""
    try:
        from exchanges.clients import _fetch_binance_price, _FALLBACK_PRICES
        p = _fetch_binance_price(coin)
        if p is not None:
            return p, 'BINANCE_REST'
        return _FALLBACK_PRICES.get(coin, 73000.0), 'STATIC_FALLBACK'
    except Exception:
        return 73000.0, 'STATIC_FALLBACK'


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR — Connection & Controls
# ═══════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("## ⬡ AGARTHAI // CONTROLS")
st.sidebar.divider()

# Exchange connection
st.sidebar.markdown("##### // EXCHANGE LINK")
exchange = st.sidebar.selectbox(
    "Exchange node", ["Hyperliquid", "Bitget Futures"],
    key="exchange_select", label_visibility="collapsed"
)

EXCHANGE_COINS = {
    "Hyperliquid": ["BTC", "ETH", "SOL", "DOGE", "ARB", "OP", "AVAX", "LINK"],
    "Bitget Futures": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "ARBUSDT"],
}

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("CONNECT", use_container_width=True, type="primary"):
        with st.spinner("Handshaking..."):
            time.sleep(0.6)
            st.session_state.connected = True
            st.toast(f"NODE ONLINE — {exchange} (paper)")

with col2:
    if st.button("SEVER", use_container_width=True):
        st.session_state.connected = False
        st.session_state.strategy_running = False
        st.toast("NODE SEVERED")

if st.session_state.connected:
    st.sidebar.markdown(
        '<span class="sig-dot green"></span> **CONNECTED** — paper mode',
        unsafe_allow_html=True
    )
else:
    st.sidebar.markdown(
        '<span class="sig-dot red"></span> **DISCONNECTED**',
        unsafe_allow_html=True
    )

st.sidebar.divider()

# Coin selection
st.sidebar.markdown("##### // ASSET VECTOR")
available_coins = EXCHANGE_COINS.get(exchange, ["BTC"])
coin = st.sidebar.selectbox(
    "Trading pair", available_coins,
    disabled=not st.session_state.connected,
    label_visibility="collapsed"
)

# Capital & position sizing
st.sidebar.markdown("##### // CAPITAL PARAMETERS")
capital = st.sidebar.number_input("Allocated capital ($)", value=1500, step=100,
                                   label_visibility="visible")
leverage = st.sidebar.slider("Leverage", 1, 10, 3)
risk_pct = st.sidebar.slider("Risk / trade (%)", 1, 5, 2) / 100

# Strategy params
st.sidebar.markdown("##### // TIER THRESHOLDS")
tp = st.sidebar.number_input("TP (%)", value=0.75, step=0.25, format="%.2f") / 100
sl = st.sidebar.number_input("SL (%)", value=0.35, step=0.05, format="%.2f") / 100

st.sidebar.divider()

# Strategy control
st.sidebar.markdown("##### // STRATEGY CONTROL")
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("RUN", use_container_width=True, type="primary",
                  disabled=not st.session_state.connected or st.session_state.strategy_running):
        st.session_state.strategy_running = True
        st.toast("STRATEGY ONLINE")
        st.rerun()

with col2:
    if st.button("HALT", use_container_width=True,
                  disabled=not st.session_state.strategy_running):
        st.session_state.strategy_running = False
        st.toast("STRATEGY HALTED")
        st.rerun()

if st.sidebar.button("!! EMERGENCY STOP — FLATTEN ALL !!", use_container_width=True):
    st.session_state.strategy_running = False
    st.session_state.current_position = None
    st.toast("EMERGENCY STOP — positions flattened")
    st.rerun()

if st.session_state.strategy_running:
    st.sidebar.markdown(
        '<span class="sig-dot cyan"></span> **STRATEGY ACTIVE**',
        unsafe_allow_html=True
    )
else:
    st.sidebar.markdown(
        '<span class="sig-dot yellow"></span> **STRATEGY IDLE**',
        unsafe_allow_html=True
    )


# ═══════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([3, 2, 1])
with col1:
    if st.session_state.strategy_running:
        st.markdown(
            '# <span class="sig-dot green"></span>AGARTHAI // LIVE',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '# <span class="sig-dot yellow"></span>AGARTHAI // STANDBY',
            unsafe_allow_html=True
        )
with col2:
    st.markdown(f"### {exchange.upper().replace(' ', '_')} | {coin}")
with col3:
    st.markdown(f"`{datetime.now().strftime('%H:%M:%S')}`")

st.divider()

# ── Live price fetch ───────────────────────────────────────────────────────
live_price, price_source = get_live_price(coin)
# Keep a rolling buffer for the chart
if 'price_history' not in st.session_state:
    st.session_state.price_history = []
st.session_state.price_history.append(live_price)
# Keep last 3600 points (1h at 1s refresh)
if len(st.session_state.price_history) > 3600:
    st.session_state.price_history = st.session_state.price_history[-3600:]

# Price change vs previous tick
prev_price = st.session_state.price_history[-2] if len(st.session_state.price_history) > 1 else live_price
tick_change_pct = (live_price / prev_price - 1) * 100 if prev_price else 0

# ── Top metrics row ────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "LAST PRICE",
    f"${live_price:,.2f}",
    f"{tick_change_pct:+.3f}%"
)

unrealized = 0.0
if st.session_state.current_position:
    pos = st.session_state.current_position
    # Use realistic entry price (post-slippage) for PnL
    entry = pos.get('exec_price', pos.get('entry_price', live_price))
    unrealized = (live_price / entry - 1) * pos['direction'] * pos['size_usd']

c2.metric("UNREALIZED PNL", f"${unrealized:+.2f}")

realized = sum(t.get('pnl_usd', 0) for t in st.session_state.paper_trades)
c3.metric("REALIZED PNL", f"${realized:+.2f}")
c4.metric("EXECUTIONS", str(len(st.session_state.paper_trades)))
c5.metric("NET CAPITAL", f"${capital + realized + unrealized:,.0f}")

st.markdown(
    f'<div style="text-align:right;color:#445566;font-size:0.65rem;margin-top:-8px;">'
    f'price source: {price_source}</div>',
    unsafe_allow_html=True
)

# ── Live Chart ─────────────────────────────────────────────────────────────
st.markdown("### // PRICE VECTOR")

ph = st.session_state.price_history
n_pts = len(ph)
import pandas as pd
from datetime import timedelta
t_axis = pd.date_range(end=datetime.now(), periods=n_pts, freq='1s')

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=t_axis, y=ph, mode='lines',
    line=dict(color='#00e5ff', width=1.2),
    fill='tozeroy', fillcolor='rgba(0,229,255,0.04)',
    name='price',
))

# Mark paper trades on chart
for trade in st.session_state.paper_trades[-20:]:
    color = '#00ff41' if trade.get('pnl_usd', 0) >= 0 else '#ff1744'
    symbol = 'triangle-up' if trade.get('direction', 1) == 1 else 'triangle-down'
    entry_t = trade.get('time', datetime.now())
    ep = trade.get('entry_price', live_price)
    fig.add_trace(go.Scatter(
        x=[entry_t], y=[ep], mode='markers',
        marker=dict(color=color, size=10, symbol=symbol),
        showlegend=False,
    ))

# Current position lines
if st.session_state.current_position:
    pos = st.session_state.current_position
    ep = pos.get('exec_price', pos.get('entry_price', live_price))
    fig.add_hline(y=ep, line_dash='dash', line_color='#ffea00', line_width=1,
                  annotation_text=f"ENTRY ${ep:,.0f}",
                  annotation_font=dict(color='#ffea00', size=10))
    tp_lvl = ep * (1 + tp * pos['direction'])
    sl_lvl = ep * (1 - sl * pos['direction'])
    fig.add_hline(y=tp_lvl, line_dash='dot', line_color='#00ff41', line_width=1,
                  annotation_text="TP", annotation_font=dict(color='#00ff41', size=10))
    fig.add_hline(y=sl_lvl, line_dash='dot', line_color='#ff1744', line_width=1,
                  annotation_text="SL", annotation_font=dict(color='#ff1744', size=10))

fig.update_layout(
    template='plotly_dark',
    height=380,
    paper_bgcolor='#060a0e',
    plot_bgcolor='#060a0e',
    font=dict(family='Share Tech Mono, Courier New', color='#8aff80', size=11),
    xaxis=dict(gridcolor='#0d1a0d', showgrid=True, title=''),
    yaxis=dict(gridcolor='#0d1a0d', showgrid=True, title='PRICE (USD)',
               tickformat='$,.0f'),
    margin=dict(l=70, r=20, t=10, b=30),
    showlegend=False,
)
st.plotly_chart(fig, use_container_width=True)

# ── Warmup progress bars ───────────────────────────────────────────────────
with st.expander("// STRATEGY WARMUP STATUS", expanded=st.session_state.strategy_running):
    try:
        from live.engine import LiveEngine
        if 'live_engine' not in st.session_state:
            st.session_state.live_engine = LiveEngine()
        engine = st.session_state.live_engine
        engine.update_runtime()
        for strat_name, rt in engine.strategy_runtime.items():
            required = int(rt['warmup_required_sec'])
            buffered = int(rt['buffered_sec']) if rt['active'] else 0
            pct = min(1.0, buffered / required) if required > 0 else 0
            status_label = "ACTIVE" if rt['active'] else "OFFLINE"
            h_buf = buffered // 3600
            m_buf = (buffered % 3600) // 60
            h_req = required // 3600
            m_req = (required % 3600) // 60
            st.markdown(
                f'`{strat_name.upper()}` — '
                f'`{h_buf:02d}h{m_buf:02d}m` / `{h_req:02d}h{m_req:02d}m` — `{status_label}` — `{pct*100:.1f}%`'
            )
            bar_w = int(pct * 100)
            st.markdown(
                f'<div class="warmup-bar-bg">'
                f'<div class="warmup-bar-fg" style="width:{bar_w}%"></div>'
                f'</div>',
                unsafe_allow_html=True
            )
    except Exception as e:
        st.caption(f"Engine not loaded — {e}")

# ── Current Position ───────────────────────────────────────────────────────
st.markdown("### // ACTIVE POSITION")

if st.session_state.current_position:
    pos = st.session_state.current_position
    side_class = 'long' if pos['direction'] == 1 else 'short'
    side_label = 'LONG' if pos['direction'] == 1 else 'SHORT'
    ep = pos.get('exec_price', pos.get('entry_price', live_price))

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("VECTOR", side_label)
    col2.metric("EXEC PRICE", f"${ep:,.2f}")
    col3.metric("FILL PRICE", f"${pos.get('entry_price', ep):,.2f}")
    col4.metric("EXPOSURE", f"${pos['size_usd']:,.0f}")
    col5.metric("UNREALIZED", f"${unrealized:+.2f}")

    dur_sec = int(time.time() - pos.get('entry_time_ts', time.time()))
    h, m = dur_sec // 3600, (dur_sec % 3600) // 60
    slippage_entry = pos.get('slippage_usd', 0.0)
    fee_entry = pos.get('fee_usd', 0.0)
    st.markdown(
        f'<div class="pos-card {side_class}">'
        f'Duration: <b>{h:02d}h {m:02d}m</b> &nbsp;|&nbsp; '
        f'Slippage (entry): <b>${slippage_entry:.3f}</b> &nbsp;|&nbsp; '
        f'Fee (entry): <b>${fee_entry:.3f}</b>'
        f'</div>',
        unsafe_allow_html=True
    )

    if st.button("CLOSE POSITION [MANUAL]", type="secondary"):
        # Apply exit slippage (sell side)
        exit_side = 'sell' if pos['direction'] == 1 else 'buy'
        try:
            from exchanges.clients import _apply_paper_slippage
            exec_exit, slip_exit = _apply_paper_slippage(live_price, exit_side, pos['size_usd'])
        except Exception:
            exec_exit, slip_exit = live_price, 0.0

        # PnL uses actual execution prices (post-slippage)
        entry_exec = pos.get('exec_price', pos.get('entry_price', live_price))
        pnl = (exec_exit / entry_exec - 1) * pos['direction'] * pos['size_usd']
        # Deduct exit fee
        exit_fee = pos['size_usd'] * 0.00035  # taker
        net_pnl = pnl - exit_fee - slip_exit

        st.session_state.paper_trades.append({
            'time': datetime.now(),
            'entry_price': pos['entry_price'],
            'exec_price': entry_exec,
            'exit_price': exec_exit,
            'direction': pos['direction'],
            'size_usd': pos['size_usd'],
            'pnl_usd': net_pnl,
            'slippage_usd': slippage_entry + slip_exit,
            'fee_usd': fee_entry + exit_fee,
            'reason': 'MANUAL',
        })
        st.session_state.current_position = None
        st.toast(f"POSITION CLOSED — NET PNL ${net_pnl:+.2f}")
        st.rerun()

else:
    msg = ("SCANNING — awaiting signal threshold breach..."
           if st.session_state.strategy_running
           else "IDLE — engage RUN to begin signal scan.")
    st.markdown(
        f'<div style="color:#445566;padding:12px;border:1px solid #00ff4122;'
        f'background:#060a0e;">{msg}</div>',
        unsafe_allow_html=True
    )

# ── Trade History ──────────────────────────────────────────────────────────
st.markdown("### // EXECUTION LOG")

if st.session_state.paper_trades:
    hist_df = pd.DataFrame(st.session_state.paper_trades)
    if 'direction' in hist_df.columns:
        hist_df['vector'] = hist_df['direction'].map({1: 'LONG', -1: 'SHORT'})
    # Format numeric columns
    for col in ['pnl_usd', 'slippage_usd', 'fee_usd', 'size_usd', 'entry_price', 'exit_price']:
        if col in hist_df.columns:
            hist_df[col] = hist_df[col].map(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x)
    st.dataframe(hist_df, use_container_width=True, height=200)

    # Equity curve
    pnls_raw = [t.get('pnl_usd', 0) for t in st.session_state.paper_trades]
    pnls_num = []
    for v in pnls_raw:
        try:
            pnls_num.append(float(v))
        except Exception:
            pnls_num.append(0.0)
    eq = capital + np.cumsum(pnls_num)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        y=np.concatenate([[capital], eq]),
        mode='lines+markers',
        line=dict(color='#00ff41', width=2),
        marker=dict(size=5, color='#00ff41'),
        fill='tozeroy', fillcolor='rgba(0,255,65,0.04)',
    ))
    fig2.add_hline(y=capital, line_dash='dash', line_color='#445566', line_width=1)
    fig2.update_layout(
        template='plotly_dark',
        paper_bgcolor='#060a0e', plot_bgcolor='#060a0e',
        height=180,
        font=dict(family='Share Tech Mono, Courier New', color='#8aff80', size=11),
        xaxis=dict(gridcolor='#0d1a0d', title=''),
        yaxis=dict(gridcolor='#0d1a0d', title='CAPITAL ($)', tickformat='$,.0f'),
        margin=dict(l=70, r=20, t=8, b=30),
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.markdown(
        '<div style="color:#445566;padding:12px;border:1px solid #0d1a0d;">'
        'NO EXECUTIONS ON RECORD.</div>',
        unsafe_allow_html=True
    )

# ── Strategy State Monitor ─────────────────────────────────────────────────
with st.expander("// SYSTEM TELEMETRY"):
    c1, c2, c3 = st.columns(3)
    c1.metric("STATE MACHINE", "SCANNING" if st.session_state.strategy_running else "IDLE")
    c2.metric("EXECUTIONS", str(len(st.session_state.paper_trades)))
    c3.metric("NODE STATUS", "ONLINE" if st.session_state.connected else "OFFLINE")

    st.json({
        "exchange": exchange,
        "asset": coin,
        "live_price": live_price,
        "price_source": price_source,
        "strategy_running": st.session_state.strategy_running,
        "connected": st.session_state.connected,
        "capital_usd": capital,
        "leverage": leverage,
        "tp_pct": tp,
        "sl_pct": sl,
        "risk_pct": risk_pct,
        "n_executions": len(st.session_state.paper_trades),
        "position_open": st.session_state.current_position is not None,
        "realized_pnl": round(realized, 4),
        "unrealized_pnl": round(unrealized, 4),
    })

# ── Footer ─────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<div style="color:#2a3a2a;font-size:0.65rem;text-align:center;">'
    'PAPER TRADING ONLY — NO REAL ORDERS — '
    'Slippage: Almgren-Chriss | Fees: HL maker 0.01% taker 0.035% — '
    f'Refresh: {datetime.now().strftime("%H:%M:%S")}'
    '</div>',
    unsafe_allow_html=True
)

# ── Auto-refresh every 5s when active ─────────────────────────────────────
if st.session_state.strategy_running:
    time.sleep(5)
    st.rerun()
