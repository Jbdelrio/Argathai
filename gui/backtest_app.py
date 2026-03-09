"""
Agarthai — Backtest Analytics GUI
===================================
streamlit run gui/backtest_app.py

Features:
  - Load BTC/ETH 1s data
  - Run TIER-Q6h-D backtest with exchange-specific fees
  - Full risk metrics dashboard (VaR, Sharpe, Sortino, MaxDD...)
  - Position history table
  - Equity curve + drawdown plots
  - TP/SL sensitivity heatmap
  - Almgren-Chriss slippage breakdown
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys, time
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agarthai — Backtest",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark theme CSS
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .metric-card {
        background: #1a1d23; border-radius: 8px; padding: 16px;
        border: 1px solid #2d3139; margin: 4px 0;
    }
    .metric-value { font-size: 24px; font-weight: bold; color: #00e676; }
    .metric-label { font-size: 12px; color: #8b8d93; }
    .metric-negative { color: #ff1744; }
    h1, h2, h3 { color: #ffffff !important; }
    .diag-box {
        background: #0c1014; border: 1px solid #2a9fd640;
        border-radius: 4px; padding: 12px; font-size: 0.8rem;
        font-family: monospace; color: #7fbfdf; margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Configuration")

# Data source
st.sidebar.subheader("📁 Data")
coin = st.sidebar.selectbox("Coin", ["BTC", "ETH"])
data_source = st.sidebar.radio("Source", ["Local CSV", "Synthetic (demo)"])

# Exchange
st.sidebar.subheader("🏦 Exchange")
exchange = st.sidebar.selectbox("Exchange", ["Hyperliquid", "Bitget Futures"])

FEES = {
    "Hyperliquid": {"maker": 0.0001, "taker": 0.00035, "slip": 0.0001},
    "Bitget Futures": {"maker": 0.0002, "taker": 0.0006, "slip": 0.0002},
}
fee_cfg = FEES[exchange]
st.sidebar.caption(f"Maker: {fee_cfg['maker']*100:.3f}% | Taker: {fee_cfg['taker']*100:.3f}%")

# Capital
st.sidebar.subheader("💰 Capital")
capital = st.sidebar.number_input("Initial ($)", value=1500, step=100)
leverage = st.sidebar.slider("Leverage", 1, 10, 3)
risk_per_trade = st.sidebar.slider("Risk/trade (%)", 1, 5, 2) / 100

# Strategy params
st.sidebar.subheader("🎯 Strategy (TIER-Q6h-D)")
tp_pct = st.sidebar.slider("Take Profit (%)", 0.25, 2.0, 0.75, 0.25) / 100
sl_pct = st.sidebar.slider("Stop Loss (%)", 0.15, 1.0, 0.35, 0.05) / 100
horizon_h = st.sidebar.slider("Horizon (h)", 2, 12, 6)

run_backtest = st.sidebar.button("🚀 Run Backtest", type="primary", use_container_width=True)


# ── Main ─────────────────────────────────────────────────────────────────
st.title("📊 Agarthai — Backtest Analytics")

if not run_backtest:
    st.info("Configure parameters in the sidebar and click **Run Backtest**.")

    # Show quick viability check
    c_rt = 2 * fee_cfg['maker'] + fee_cfg['slip']
    p_min = (sl_pct + c_rt) / (tp_pct + sl_pct)
    pos_usd = min(capital * risk_per_trade / sl_pct, capital * leverage * 0.30)
    profit_tp = tp_pct * pos_usd
    loss_sl = (sl_pct + c_rt) * pos_usd
    fee_cost = c_rt * pos_usd

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Position size", f"${pos_usd:,.0f}")
    col2.metric("Profit if TP", f"${profit_tp:.2f}", delta=f"+{profit_tp:.2f}")
    col3.metric("Loss if SL", f"-${loss_sl:.2f}", delta=f"-{loss_sl:.2f}")
    col4.metric("Breakeven WR", f"{p_min:.1%}")

    st.stop()

# ── Run Backtest ─────────────────────────────────────────────────────────
progress = st.progress(0, "Loading data...")

# ── Diagnostic container (always visible, appended as we go) ─────────────
diag_box = st.empty()
diag_lines: list[str] = []

def _diag(msg: str):
    """Append a line to the terminal-style diagnostic box and print to terminal."""
    print(f"[backtest] {msg}")
    diag_lines.append(msg)
    diag_box.markdown(
        '<div class="diag-box">' +
        '<br>'.join(diag_lines[-30:]) +   # last 30 lines
        '</div>',
        unsafe_allow_html=True,
    )

_diag(f"Starting backtest — coin={coin}, exchange={exchange}, source={data_source}")

try:
    if data_source == "Local CSV":
        _diag("Loading CSV data…")
        from data.loader import load_1s_data
        df = load_1s_data(coin.lower())
        _diag(f"Loaded {len(df):,} rows | {len(df)/86400:.1f} days")
    else:
        # Generate synthetic
        st.warning("Using synthetic data (demo). Load real CSV for actual results.")
        _diag("Generating synthetic data (20 days, 1s)…")
        rng = np.random.default_rng(42)
        N = 20 * 86400
        r = rng.standard_normal(N) * 0.0001
        log_p = np.log(78000) + np.cumsum(r)
        price = np.exp(log_p)
        qty = rng.exponential(0.12, N)
        bf = np.clip(0.5 + 0.3*np.tanh(r*1e4) + rng.uniform(-0.05,0.05,N), 0.05, 0.95)
        df = pd.DataFrame({
            'timestamp': pd.date_range('2026-02-01', periods=N, freq='1s'),
            'last': price, 'vwap': price, 'qty': qty,
            'buy_qty': qty*bf, 'sell_qty': qty*(1-bf),
            'n_trades': rng.poisson(10, N),
            'ofi_proxy': qty*bf - qty*(1-bf),
            'symbol': f'{coin}USDT',
            'ret_1s': np.concatenate([[0], np.diff(log_p)]),
            'log_price': log_p,
        })
        _diag(f"Synthetic data ready: {N:,} rows")

    progress.progress(20, "Computing features...")
    n_rows = len(df)
    n_days = n_rows / 86400

    st.success(f"Loaded {n_rows:,} rows ({n_days:.1f} days) | "
               f"Price: {df['last'].iloc[0]:,.0f} → {df['last'].iloc[-1]:,.0f}")

except Exception as e:
    _diag(f"ERROR loading data: {e}")
    st.error(f"Error loading data: {e}")
    st.info("Place btc_1s.csv.gz / eth_1s.csv.gz in `data/binance_spot/` or set AGARTHAI_DATA_DIRS.")
    st.stop()

# ── Compute features (simplified inline for GUI responsiveness) ──────────
progress.progress(40, "Computing TIER features...")
_diag("Computing TIER features (I, S, E*)…")

r = df['ret_1s'].fillna(0).values
v = df['qty'].fillna(0).values
q = df['ofi_proxy'].fillna(0).values
prices = df['last'].values

# Rolling sums
def rsum(x, w):
    cs = np.cumsum(np.concatenate([[0], x]))
    out = np.full(len(x), np.nan); out[w-1:] = cs[w:] - cs[:len(x)-w+1]; return out
def rsumsq(x, w):
    cs = np.cumsum(np.concatenate([[0], x**2]))
    out = np.full(len(x), np.nan); out[w-1:] = cs[w:] - cs[:len(x)-w+1]; return out

ws, wm, wl = 60, 600, 3600
R_ws = rsum(r, ws); V_ws = rsum(v, ws); Q_ws = rsum(q, ws)
sigma_ws = np.sqrt(np.maximum(rsumsq(r, ws), 0))
sigma_wm = np.sqrt(np.maximum(rsumsq(r, wm), 0))
sigma_wl = np.sqrt(np.maximum(rsumsq(r, wl), 0))

med_V = pd.Series(V_ws).rolling(wl, min_periods=wl//2).median().values
eps = 1e-12
I = (np.abs(R_ws)/(sigma_wl+eps)) * (V_ws/(med_V+eps))
d = np.sign(R_ws)
S = sigma_ws / (sigma_wm + eps)

# Thresholds — use 7-day calibration window
cb = 7 * 86400
tau_I      = pd.Series(I).rolling(cb, min_periods=cb//4).quantile(0.95).values
tau_I_down = pd.Series(I).rolling(cb, min_periods=cb//4).quantile(0.80).values
tau_S      = pd.Series(S).rolling(cb, min_periods=cb//4).quantile(0.35).values

# Warmup check: need at least cb//4 rows before tau_I is valid
warmup_rows = max(wl, cb // 4)          # 151 200 rows ≈ 42h
valid_from  = np.argmax(~np.isnan(tau_I))  # first non-NaN index
_diag(f"Feature warmup: need {warmup_rows:,} rows ({warmup_rows/3600:.0f}h) | "
      f"first valid tau_I at row {valid_from:,} ({valid_from/3600:.1f}h)")

progress.progress(60, "Running state machine...")
_diag("Running TIER state machine…")

# Simple E* (combined score)
Lambda = np.abs(R_ws) / (np.abs(Q_ws) + eps)
A = np.full(len(r), np.nan); A[180:] = Lambda[:-180] - Lambda[180:]
C = np.abs(Q_ws) / (V_ws + eps)

def rz(x, w):
    s = pd.Series(x); med = s.rolling(w, min_periods=w//4).median()
    mad = (s-med).abs().rolling(w, min_periods=w//4).median() * 1.4826
    return ((s-med)/(mad+eps)).values

E_star = rz(I, wl) + rz(C, wl) + rz(A, wl) + rz(-S, wl)
tau_E  = pd.Series(E_star).rolling(cb, min_periods=cb//4).quantile(0.92).values

# ── State Machine ────────────────────────────────────────────────────────
step = 600; cooldown = 1800; persist_k = 2
state = 'IDLE'; imp_dir = 0; imp_start = 0; consec = 0; last_entry = -cooldown-1
setups = []
state_counts = {'IDLE': 0, 'IMPULSE': 0, 'STAB': 0}

loop_start = warmup_rows if valid_from == 0 else valid_from
for t in range(loop_start, len(r), step):
    if t - last_entry < cooldown: continue
    if np.isnan(I[t]) or np.isnan(tau_I[t]): continue

    state_counts[state] = state_counts.get(state, 0) + 1

    if state == 'IDLE':
        if I[t] > tau_I[t]:
            state = 'IMPULSE'; imp_dir = int(d[t]) if not np.isnan(d[t]) else 0
            imp_start = t; consec = 0
    elif state == 'IMPULSE':
        if I[t] < tau_I_down[t] and S[t] < tau_S[t]:
            state = 'STAB'; consec = 0
        elif t - imp_start > 7200:
            state = 'IDLE'
    elif state == 'STAB':
        if I[t] > tau_I[t]:
            state = 'IMPULSE'; imp_dir = int(d[t]) if not np.isnan(d[t]) else 0
            imp_start = t; consec = 0; continue
        if not np.isnan(E_star[t]) and not np.isnan(tau_E[t]) and E_star[t] > tau_E[t]:
            consec += 1
        else:
            consec = 0
        if consec >= persist_k and imp_dir != 0:
            setups.append({
                'idx': t, 'dir': -imp_dir, 'E': E_star[t],
                'price': prices[t], 'time': df['timestamp'].iloc[t] if 'timestamp' in df else t,
            })
            last_entry = t; state = 'IDLE'; consec = 0
        elif t - imp_start > 10800:
            state = 'IDLE'; consec = 0

_diag(f"State machine done: {len(setups)} setups | "
      f"states visited — IDLE:{state_counts.get('IDLE',0)}, "
      f"IMPULSE:{state_counts.get('IMPULSE',0)}, "
      f"STAB:{state_counts.get('STAB',0)}")

progress.progress(80, f"Labeling {len(setups)} setups...")
_diag(f"Labeling {len(setups)} setups with TP/SL/TIME…")

# ── Label trades with position sizing & Almgren-Chriss slippage ──────────
c_rt = 2 * fee_cfg['maker'] + fee_cfg['slip']
horizon = horizon_h * 3600
trades = []
cap = capital

for s in setups:
    t0 = s['idx']; p0 = s['price']; direction = s['dir']
    if np.isnan(p0) or p0 <= 0: continue

    # Position sizing (risk-based)
    pos = min(cap * risk_per_trade / sl_pct, cap * leverage * 0.30)
    if pos < 10: continue

    # Almgren-Chriss slippage
    sigma_local = sigma_wm[t0] if not np.isnan(sigma_wm[t0]) else 0.0001
    ac_slip = 0.5 * 1/10000 + 0.0002 * np.sqrt(sigma_local)  # simplified
    total_cost = c_rt + ac_slip

    end = min(t0 + horizon, len(prices) - 1)
    exit_idx = end; exit_reason = 'TIME'; label = 0
    mfe = 0; mae = 0

    for i in range(t0+1, end+1):
        pi = prices[i]
        if np.isnan(pi): continue
        ret = (pi/p0 - 1) * direction
        mfe = max(mfe, ret); mae = min(mae, ret)
        if ret >= tp_pct:
            exit_idx = i; exit_reason = 'TP'; label = 1; break
        elif ret <= -sl_pct:
            exit_idx = i; exit_reason = 'SL'; break

    raw_pnl = (prices[exit_idx]/p0 - 1) * direction
    net_pnl = raw_pnl - total_cost
    pnl_usd = net_pnl * pos
    cap += pnl_usd

    trades.append({
        'entry_idx': t0, 'entry_time': s['time'], 'entry_price': p0,
        'exit_idx': exit_idx, 'exit_price': prices[exit_idx],
        'direction': direction, 'E_star': s['E'],
        'pnl_pct': raw_pnl, 'pnl_net_pct': net_pnl, 'pnl_usd': pnl_usd,
        'exit_reason': exit_reason, 'label': label,
        'duration_sec': exit_idx - t0,
        'position_usd': pos, 'max_favorable': mfe, 'max_adverse': mae,
        'slippage_ac': ac_slip, 'fees_pct': c_rt,
    })

_diag(f"Labeling done: {len(trades)} trades out of {len(setups)} setups")

trades_df = pd.DataFrame(trades)
progress.progress(100, "Done!")
time.sleep(0.5)
progress.empty()

# ── No trades diagnostic ──────────────────────────────────────────────────
if trades_df.empty:
    _diag("WARNING: No trades generated — see diagnostics below")

    st.warning("⚠️ No trades generated.")

    # Diagnostic table
    rows_avail  = len(r)
    rows_warmup = warmup_rows
    rows_active = max(0, rows_avail - rows_warmup)
    n_loops     = rows_active // step if rows_active > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows disponibles", f"{rows_avail:,}", delta=f"{rows_avail/86400:.1f} jours")
    c2.metric("Warmup requis", f"{rows_warmup:,}", delta=f"{rows_warmup/3600:.0f}h")
    c3.metric("Rows actifs (post-warmup)", f"{rows_active:,}", delta=f"{rows_active/86400:.1f} jours")
    c4.metric("Itérations state machine", f"{n_loops:,}")

    if rows_avail < rows_warmup:
        st.error(
            f"❌ Données insuffisantes : {rows_avail:,} rows < warmup requis {rows_warmup:,} rows "
            f"({rows_warmup/3600:.0f}h). Le state machine n'a jamais tourné."
        )
    elif len(setups) == 0:
        st.info(
            f"Le state machine a itéré {n_loops:,} fois mais aucun setup n'a été détecté. "
            "Les seuils quantiles sont peut-être trop stricts sur cette période."
        )
        st.info(
            f"Essaie : source = **Synthetic (demo)** pour valider le pipeline, "
            "ou augmente le TP et réduis le SL pour assouplir les conditions."
        )
    else:
        st.info(f"{len(setups)} setups détectés mais 0 trades valides après filtrage.")

    st.stop()

# ══════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════

_diag(f"Rendering dashboard — {len(trades_df)} trades")

# ── Metrics Cards ────────────────────────────────────────────────────────
from backtest.metrics import compute_all_metrics

metrics = compute_all_metrics(trades_df, capital)

st.subheader("📈 Performance Overview")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Trades", metrics['n_trades'])
c2.metric("Win Rate", f"{metrics['win_rate']:.1%}")
c3.metric("Total PnL", f"${metrics['total_pnl_usd']:.2f}",
          delta=f"{metrics['return_pct']:.1f}%")
c4.metric("Sharpe", f"{metrics['sharpe']:.2f}")
c5.metric("Sortino", f"{metrics['sortino']:.2f}")
c6.metric("Max DD", f"${metrics['max_dd_usd']:.2f}",
          delta=f"{metrics['max_dd_pct']:.1f}%")

st.subheader("🛡️ Risk Metrics")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("VaR 95%", f"${metrics['var_95_usd']:.2f}")
c2.metric("CVaR 95%", f"${metrics['cvar_95_usd']:.2f}")
c3.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
c4.metric("Calmar", f"{metrics['calmar']:.2f}")
c5.metric("Kelly f*", f"{metrics['kelly_f']:.2f}")
c6.metric("Final Capital", f"${metrics['final_capital']:.0f}")

# ── Equity Curve ─────────────────────────────────────────────────────────
st.subheader("📉 Equity Curve & Drawdown")

eq   = np.array(metrics['equity_curve'])
peak = np.maximum.accumulate(eq)
dd   = eq - peak

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                    vertical_spacing=0.05)

fig.add_trace(go.Scatter(y=eq, mode='lines', name='Equity',
                          line=dict(color='#00e676', width=2),
                          fill='tozeroy', fillcolor='rgba(0,230,118,0.1)'), row=1, col=1)
fig.add_trace(go.Scatter(y=peak, mode='lines', name='Peak',
                          line=dict(color='#616161', width=1, dash='dash')), row=1, col=1)
fig.add_trace(go.Scatter(y=dd, mode='lines', name='Drawdown',
                          line=dict(color='#ff1744', width=1.5),
                          fill='tozeroy', fillcolor='rgba(255,23,68,0.15)'), row=2, col=1)

fig.update_layout(template='plotly_dark', height=500, showlegend=True,
                  margin=dict(l=60, r=20, t=30, b=30))
fig.update_yaxes(title_text="Capital ($)", row=1, col=1)
fig.update_yaxes(title_text="DD ($)", row=2, col=1)
st.plotly_chart(fig, use_container_width=True)

# ── PnL Distribution ────────────────────────────────────────────────────
st.subheader("📊 PnL Distribution")
col1, col2 = st.columns(2)

with col1:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=trades_df['pnl_usd'], nbinsx=20,
                                marker_color='#2979ff', opacity=0.7))
    fig.add_vline(x=0, line_color='#ff1744', line_width=1)
    fig.add_vline(x=trades_df['pnl_usd'].mean(), line_color='#ffea00',
                  line_width=1.5, line_dash='dash')
    fig.update_layout(template='plotly_dark', height=300,
                      xaxis_title='PnL ($)', yaxis_title='Count')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    exits = trades_df['exit_reason'].value_counts()
    colors = {'TP': '#00e676', 'SL': '#ff1744', 'TIME': '#ff9100'}
    fig = go.Figure(data=[go.Pie(
        labels=exits.index, values=exits.values,
        marker_colors=[colors.get(k, '#616161') for k in exits.index],
        hole=0.4,
    )])
    fig.update_layout(template='plotly_dark', height=300)
    st.plotly_chart(fig, use_container_width=True)

# ── Position History ─────────────────────────────────────────────────────
st.subheader("📋 Position History")
display_cols = ['entry_time', 'direction', 'entry_price', 'exit_price',
                'pnl_usd', 'pnl_net_pct', 'exit_reason', 'duration_sec',
                'position_usd', 'slippage_ac', 'fees_pct']
available = [c for c in display_cols if c in trades_df.columns]

styled = trades_df[available].copy()
if 'pnl_usd' in styled:
    styled['pnl_usd'] = styled['pnl_usd'].round(2)
if 'pnl_net_pct' in styled:
    styled['pnl_net_pct'] = (styled['pnl_net_pct'] * 100).round(4)
if 'slippage_ac' in styled:
    styled['slippage_ac'] = (styled['slippage_ac'] * 100).round(4)
if 'direction' in styled:
    styled['direction'] = styled['direction'].map({1: '🟢 LONG', -1: '🔴 SHORT'})

st.dataframe(styled, use_container_width=True, height=300)

# ── Slippage Breakdown (Almgren-Chriss) ──────────────────────────────────
st.subheader("🔬 Almgren-Chriss Slippage Analysis")
if 'slippage_ac' in trades_df:
    avg_slip  = trades_df['slippage_ac'].mean()
    avg_fees  = trades_df['fees_pct'].mean()
    total_cost = avg_slip + avg_fees
    avg_pos   = trades_df['position_usd'].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Slippage (AC)", f"{avg_slip*100:.4f}%")
    c2.metric("Avg Exchange Fees", f"{avg_fees*100:.3f}%")
    c3.metric("Total Cost/Trade", f"${total_cost*avg_pos:.2f}")
    c4.metric("Cost as % of TP", f"{total_cost/tp_pct*100:.1f}%")

st.caption(f"Exchange: {exchange} | Capital: ${capital} × {leverage}x | "
           f"TP={tp_pct*100:.2f}% SL={sl_pct*100:.2f}% | "
           f"AC slippage model applied")
