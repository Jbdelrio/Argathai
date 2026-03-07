"""
Agarthai — Live Paper Trading GUI
====================================
streamlit run gui/live_app.py

Features:
  - Connect to Hyperliquid / Bitget (paper mode)
  - Real-time price feed
  - Start/Stop strategy button
  - Coin selection from exchange
  - Live PnL tracking
  - Unrealized PnL display
  - Position monitor
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time, json, sys
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, List

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Page Config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agarthai — Live Trading",
    page_icon="🔴",
    layout="wide",
)

st.markdown("""
<style>
    .stApp { background-color: #0a0e14; }
    .live-dot { display:inline-block; width:10px; height:10px; border-radius:50%;
                margin-right:8px; animation: blink 1s infinite; }
    .live-dot.green { background:#00e676; }
    .live-dot.red { background:#ff1744; }
    .live-dot.yellow { background:#ffea00; }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }
    .position-card {
        background: #1a1d23; border-radius: 10px; padding: 20px;
        border-left: 4px solid #2979ff;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════

if 'strategy_running' not in st.session_state:
    st.session_state.strategy_running = False
if 'connected' not in st.session_state:
    st.session_state.connected = False
if 'paper_trades' not in st.session_state:
    st.session_state.paper_trades = []
if 'paper_capital' not in st.session_state:
    st.session_state.paper_capital = 1500.0
if 'current_position' not in st.session_state:
    st.session_state.current_position = None
if 'price_history' not in st.session_state:
    st.session_state.price_history = []
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR — Connection & Controls
# ═══════════════════════════════════════════════════════════════════════════

st.sidebar.title("🔧 Live Controls")

# Exchange connection
st.sidebar.subheader("🔗 Exchange Connection")
exchange = st.sidebar.selectbox("Exchange", ["Hyperliquid", "Bitget Futures"],
                                 key="exchange_select")

EXCHANGE_COINS = {
    "Hyperliquid": ["BTC", "ETH", "SOL", "DOGE", "ARB", "OP", "AVAX", "LINK"],
    "Bitget Futures": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "ARBUSDT"],
}

# Connect button
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("🔌 Connect", use_container_width=True, type="primary"):
        with st.spinner("Connecting..."):
            time.sleep(1)  # Simulate connection
            st.session_state.connected = True
            st.toast(f"✅ Connected to {exchange} (paper mode)")

with col2:
    if st.button("⛔ Disconnect", use_container_width=True):
        st.session_state.connected = False
        st.session_state.strategy_running = False
        st.toast("🔌 Disconnected")

# Connection status indicator
if st.session_state.connected:
    st.sidebar.markdown('<span class="live-dot green"></span> **Connected** (Paper Mode)',
                        unsafe_allow_html=True)
else:
    st.sidebar.markdown('<span class="live-dot red"></span> **Disconnected**',
                        unsafe_allow_html=True)

# Coin selection
st.sidebar.subheader("🪙 Coin Selection")
available_coins = EXCHANGE_COINS.get(exchange, ["BTC"])
coin = st.sidebar.selectbox("Trading Pair", available_coins,
                             disabled=not st.session_state.connected)

# Capital & position sizing
st.sidebar.subheader("💰 Capital")
capital = st.sidebar.number_input("Capital ($)", value=1500, step=100)
leverage = st.sidebar.slider("Leverage", 1, 10, 3)
risk_pct = st.sidebar.slider("Risk/trade (%)", 1, 5, 2) / 100

# Strategy params
st.sidebar.subheader("🎯 TIER Parameters")
tp = st.sidebar.number_input("TP (%)", value=0.75, step=0.25, format="%.2f") / 100
sl = st.sidebar.number_input("SL (%)", value=0.35, step=0.05, format="%.2f") / 100

# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY CONTROL — START / STOP
# ═══════════════════════════════════════════════════════════════════════════

st.sidebar.divider()
st.sidebar.subheader("⚡ Strategy Control")

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("▶️ START", use_container_width=True, type="primary",
                  disabled=not st.session_state.connected or st.session_state.strategy_running):
        st.session_state.strategy_running = True
        st.toast("🚀 Strategy STARTED")
        st.rerun()

with col2:
    if st.button("⏹️ STOP", use_container_width=True,
                  disabled=not st.session_state.strategy_running):
        st.session_state.strategy_running = False
        st.toast("⏹️ Strategy STOPPED")
        st.rerun()

# Emergency stop
if st.sidebar.button("🚨 EMERGENCY STOP — Close All", use_container_width=True):
    st.session_state.strategy_running = False
    st.session_state.current_position = None
    st.toast("🚨 All positions closed")
    st.rerun()


# Status
if st.session_state.strategy_running:
    st.sidebar.markdown('<span class="live-dot green"></span> **Strategy ACTIVE**',
                        unsafe_allow_html=True)
else:
    st.sidebar.markdown('<span class="live-dot yellow"></span> **Strategy IDLE**',
                        unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

# Header
col1, col2, col3 = st.columns([3, 2, 1])
with col1:
    if st.session_state.strategy_running:
        st.markdown("# 🟢 Agarthai — LIVE")
    else:
        st.markdown("# ⚪ Agarthai — STANDBY")

with col2:
    st.markdown(f"### {exchange} | {coin}")

with col3:
    st.markdown(f"**{datetime.now().strftime('%H:%M:%S')}**")

st.divider()

# ── Simulate live price data (replace with real websocket in production) ──
# In production: use ccxt or exchange websocket
simulated_price = 73000 + np.random.randn() * 100
simulated_change = np.random.randn() * 0.5

# Top metrics
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Price", f"${simulated_price:,.2f}", f"{simulated_change:+.2f}%")

unrealized = 0
if st.session_state.current_position:
    pos = st.session_state.current_position
    unrealized = (simulated_price / pos['entry_price'] - 1) * pos['direction'] * pos['size_usd']

c2.metric("Unrealized PnL", f"${unrealized:+.2f}",
          delta=f"{'🟢' if unrealized > 0 else '🔴'}")

realized = sum(t.get('pnl_usd', 0) for t in st.session_state.paper_trades)
c3.metric("Realized PnL", f"${realized:+.2f}")
c4.metric("# Trades", len(st.session_state.paper_trades))
c5.metric("Capital", f"${capital + realized + unrealized:,.0f}")

# ── Live Chart ───────────────────────────────────────────────────────────
st.subheader("📈 Live Price Chart")

# Generate simulated 1h history
np.random.seed(int(time.time()) % 1000)
n_points = 3600
t_axis = pd.date_range(end=datetime.now(), periods=n_points, freq='1s')
sim_prices = simulated_price + np.cumsum(np.random.randn(n_points) * 5)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=t_axis, y=sim_prices, mode='lines',
    line=dict(color='#00e5ff', width=1.5),
    fill='tozeroy', fillcolor='rgba(0,229,255,0.05)',
))

# Mark trades on chart
for trade in st.session_state.paper_trades[-10:]:  # last 10
    color = '#00e676' if trade.get('pnl_usd', 0) > 0 else '#ff1744'
    fig.add_annotation(
        x=trade.get('time', datetime.now()),
        y=trade.get('entry_price', simulated_price),
        text=f"{'▲' if trade.get('direction', 1) == 1 else '▼'}",
        font=dict(size=16, color=color),
        showarrow=False,
    )

# Current position marker
if st.session_state.current_position:
    pos = st.session_state.current_position
    fig.add_hline(y=pos['entry_price'], line_dash='dash',
                  line_color='#ffea00', line_width=1,
                  annotation_text=f"Entry: ${pos['entry_price']:,.0f}")
    fig.add_hline(y=pos['entry_price'] * (1 + tp * pos['direction']),
                  line_dash='dot', line_color='#00e676', line_width=1,
                  annotation_text="TP")
    fig.add_hline(y=pos['entry_price'] * (1 - sl * pos['direction']),
                  line_dash='dot', line_color='#ff1744', line_width=1,
                  annotation_text="SL")

fig.update_layout(
    template='plotly_dark', height=400,
    xaxis_title='', yaxis_title='Price ($)',
    margin=dict(l=60, r=20, t=10, b=30),
)
st.plotly_chart(fig, use_container_width=True)

# ── Current Position ─────────────────────────────────────────────────────
st.subheader("📍 Current Position")

if st.session_state.current_position:
    pos = st.session_state.current_position
    col1, col2, col3, col4, col5 = st.columns(5)
    dir_emoji = "🟢 LONG" if pos['direction'] == 1 else "🔴 SHORT"
    col1.metric("Direction", dir_emoji)
    col2.metric("Entry", f"${pos['entry_price']:,.2f}")
    col3.metric("Size", f"${pos['size_usd']:,.0f}")
    col4.metric("Unrealized", f"${unrealized:+.2f}")
    col5.metric("Duration", f"{(time.time() - pos.get('entry_time_ts', time.time()))/60:.0f} min")

    # Close position button
    if st.button("🔒 Close Position Manually", type="secondary"):
        pnl = unrealized
        st.session_state.paper_trades.append({
            'time': datetime.now(),
            'entry_price': pos['entry_price'],
            'exit_price': simulated_price,
            'direction': pos['direction'],
            'size_usd': pos['size_usd'],
            'pnl_usd': pnl,
            'reason': 'MANUAL',
        })
        st.session_state.current_position = None
        st.toast(f"Position closed: ${pnl:+.2f}")
        st.rerun()
else:
    st.info("No active position. " +
            ("Strategy is running — waiting for signal..." if st.session_state.strategy_running
             else "Start the strategy to begin scanning."))

# ── Trade History ────────────────────────────────────────────────────────
st.subheader("📋 Trade History (Paper)")

if st.session_state.paper_trades:
    hist_df = pd.DataFrame(st.session_state.paper_trades)
    if 'direction' in hist_df:
        hist_df['side'] = hist_df['direction'].map({1: '🟢 LONG', -1: '🔴 SHORT'})
    st.dataframe(hist_df, use_container_width=True, height=200)

    # Equity curve of paper trades
    pnls = [t.get('pnl_usd', 0) for t in st.session_state.paper_trades]
    eq = capital + np.cumsum(pnls)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=np.concatenate([[capital], eq]),
                              mode='lines+markers',
                              line=dict(color='#00e676', width=2),
                              marker=dict(size=6)))
    fig.add_hline(y=capital, line_dash='dash', line_color='#616161')
    fig.update_layout(template='plotly_dark', height=200,
                      yaxis_title='Paper Capital ($)',
                      margin=dict(l=60, r=20, t=10, b=30))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.caption("No trades yet.")

# ── Strategy State Monitor ───────────────────────────────────────────────
with st.expander("🔬 Strategy State Monitor"):
    c1, c2, c3 = st.columns(3)
    c1.metric("State Machine", "SCANNING" if st.session_state.strategy_running else "IDLE")
    c2.metric("Setups Since Start", len(st.session_state.paper_trades))
    c3.metric("Connection", "🟢 OK" if st.session_state.connected else "🔴 DOWN")

    st.json({
        "exchange": exchange,
        "coin": coin,
        "strategy_running": st.session_state.strategy_running,
        "connected": st.session_state.connected,
        "capital": capital,
        "leverage": leverage,
        "tp": tp, "sl": sl,
        "n_paper_trades": len(st.session_state.paper_trades),
        "current_position": st.session_state.current_position is not None,
    })

# ── Footer ───────────────────────────────────────────────────────────────
st.divider()
st.caption("⚠️ **Paper trading only** — No real orders are placed. "
           "Connect your API keys in config/ for live execution.")

# Auto-refresh every 5 seconds when strategy is running
if st.session_state.strategy_running:
    time.sleep(5)
    st.rerun()
