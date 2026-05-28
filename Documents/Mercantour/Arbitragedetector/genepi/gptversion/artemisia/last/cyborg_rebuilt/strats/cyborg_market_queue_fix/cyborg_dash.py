#!/usr/bin/env python3
"""
CYBORG – rebuilt interactive GUI

What this version fixes:
- controls are mounted once and never rerendered by the timer
- no fake trades or fake markets are loaded before Start
- Start/Stop/Pause act on a real background loop
- capital + allocation are configurable from the GUI
- Coin5min has explicit warmup / running state
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import logging
from typing import Dict, List

import dash
from dash import Dash, Input, Output, State, dcc, html, dash_table, ctx
import plotly.graph_objects as go
import pandas as pd

from data_manager import ST, get_pnl_curve, start_workers, _normalize_book_side

logging.getLogger("werkzeug").setLevel(logging.ERROR)
logger = logging.getLogger("cyborg.gui")

# ============================================================
# Theme helpers
# ============================================================

T = {
    "bg": "#060a10",
    "card": "#0c1018",
    "brd": "#1a2535",
    "cyan": "#00e5ff",
    "grn": "#00ff88",
    "red": "#ff4d6d",
    "amb": "#ffb000",
    "txt": "#d7e3f4",
    "dim": "#70839f",
    "wht": "#f3f7ff",
    "pur": "#9b6bff",
}
M = "'JetBrains Mono','Fira Code','Consolas',monospace"


def rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"


def card(children, **style):
    base = {
        "backgroundColor": T["card"],
        "border": f"1px solid {T['brd']}",
        "borderRadius": "8px",
        "padding": "14px",
        "marginBottom": "12px",
    }
    base.update(style)
    return html.Div(children, style=base)


def kpi(label: str, value: str, sub: str = "", color: str | None = None):
    return card(
        [
            html.Div(label.upper(), style={"fontFamily": M, "fontSize": "10px", "color": T["dim"], "marginBottom": "6px"}),
            html.Div(value, style={"fontFamily": M, "fontSize": "24px", "fontWeight": "700", "color": color or T["cyan"]}),
            html.Div(sub, style={"fontFamily": M, "fontSize": "10px", "color": T["dim"]}) if sub else None,
        ],
        flex="1",
        minWidth="150px",
        textAlign="center",
    )


def button(text: str, id_: str, color: str | None = None, danger: bool = False):
    c = T["red"] if danger else (color or T["cyan"])
    return html.Button(
        text,
        id=id_,
        n_clicks=0,
        style={
            "fontFamily": M,
            "fontSize": "12px",
            "fontWeight": "700",
            "borderRadius": "6px",
            "padding": "9px 14px",
            "cursor": "pointer",
            "border": f"1px solid rgba({rgb(c)},0.35)",
            "backgroundColor": "rgba(0,0,0,0)",
            "color": c,
        },
    )


def badge(text: str, color: str):
    return html.Span(
        text,
        style={
            "display": "inline-block",
            "padding": "2px 8px",
            "borderRadius": "999px",
            "fontFamily": M,
            "fontSize": "10px",
            "fontWeight": "700",
            "color": color,
            "border": f"1px solid rgba({rgb(color)},0.28)",
            "backgroundColor": f"rgba({rgb(color)},0.08)",
        },
    )


def status_color(status: str) -> str:
    return {
        "running": T["grn"],
        "connected": T["grn"],
        "warming": T["amb"],
        "connecting": T["amb"],
        "paused": T["pur"],
        "disconnected": T["red"],
        "stopped": T["red"],
        "error": T["red"],
    }.get((status or "").lower(), T["dim"])


def slider_block(id_: str, label: str, min_: float, max_: float, step: float, value: float, marks: Dict, unit: str = "%"):
    return html.Div(
        [
            html.Div(
                [
                    html.Span(label, style={"fontFamily": M, "fontSize": "11px", "color": T["dim"]}),
                    html.Span(id=f"{id_}-val", style={"fontFamily": M, "fontSize": "12px", "fontWeight": "700", "color": T["cyan"]}),
                ],
                style={"display": "flex", "justifyContent": "space-between", "marginBottom": "4px"},
            ),
            dcc.Slider(id=id_, min=min_, max=max_, step=step, value=value, marks=marks, persistence=True, persistence_type="session"),
        ],
        style={"marginBottom": "14px"},
    )


def table_component(id_: str):
    return dash_table.DataTable(
        id=id_,
        columns=[],
        data=[],
        style_header={
            "backgroundColor": T["bg"],
            "color": T["dim"],
            "fontWeight": "700",
            "fontFamily": M,
            "fontSize": "10px",
            "border": f"1px solid {T['brd']}",
        },
        style_cell={
            "backgroundColor": T["card"],
            "color": T["txt"],
            "fontFamily": M,
            "fontSize": "11px",
            "border": f"1px solid {T['brd']}",
            "padding": "6px 8px",
            "textAlign": "left",
            "whiteSpace": "normal",
            "height": "auto",
        },
        style_table={"overflowX": "auto"},
        page_size=12,
    )


# ============================================================
# Figure helpers
# ============================================================

def base_fig(title: str, height: int = 260) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=T["card"],
        plot_bgcolor=T["card"],
        font={"family": M, "color": T["txt"], "size": 11},
        margin={"l": 40, "r": 16, "t": 34, "b": 28},
        title={"text": title, "font": {"size": 11, "color": T["dim"]}},
        height=height,
        xaxis={"gridcolor": T["brd"], "zerolinecolor": T["brd"]},
        yaxis={"gridcolor": T["brd"], "zerolinecolor": T["brd"]},
        legend={"orientation": "h", "y": 1.1, "bgcolor": "rgba(0,0,0,0)"},
    )
    return fig


def pnl_figure() -> go.Figure:
    curve = get_pnl_curve()
    fig = base_fig("PNL CUMULÉ")
    if curve:
        df = pd.DataFrame(curve)
        fig.add_trace(go.Scatter(x=list(range(len(df))), y=df["pnl"], mode="lines", name="PnL"))
    return fig


def coin_chart_spot(snapshot: Dict, asset: str) -> go.Figure:
    fig = base_fig(f"SPOT / REF – {asset}", height=240)
    hist = snapshot["coin5min"].get("market_history", {}).get(asset, [])
    if hist:
        xs = list(range(len(hist)))
        spot = [r.get("spot") for r in hist]
        ref = [r.get("ref_px") for r in hist]
        fig.add_trace(go.Scatter(x=xs, y=spot, mode="lines", name="Spot sous-jacent"))
        fig.add_trace(go.Scatter(x=xs, y=ref, mode="lines", name="Prix à battre / ref"))
    return fig


def coin_chart_yesno(snapshot: Dict, asset: str) -> go.Figure:
    fig = base_fig(f"YES / NO – {asset}", height=240)
    hist = snapshot["coin5min"].get("market_history", {}).get(asset, [])
    if hist:
        xs = list(range(len(hist)))
        fig.add_trace(go.Scatter(x=xs, y=[r.get("yes_mark") for r in hist], mode="lines", name="YES mark"))
        fig.add_trace(go.Scatter(x=xs, y=[r.get("no_mark") for r in hist], mode="lines", name="NO mark"))
        fig.add_trace(go.Scatter(x=xs, y=[r.get("yes_ask") for r in hist], mode="lines", name="YES ask"))
        fig.add_trace(go.Scatter(x=xs, y=[r.get("no_ask") for r in hist], mode="lines", name="NO ask"))
        fig.update_yaxes(range=[0, 1.05])
    return fig


def coin_chart_sum(snapshot: Dict, asset: str) -> go.Figure:
    fig = base_fig(f"YES+NO – {asset}", height=240)
    hist = snapshot["coin5min"].get("market_history", {}).get(asset, [])
    if hist:
        xs = list(range(len(hist)))
        sums = [r.get("exec_sum_yes_no", r.get("sum_yes_no")) for r in hist]
        fig.add_trace(go.Scatter(x=xs, y=sums, mode="lines", name="YES+NO ask"))
        fig.add_hline(y=1.00, line_dash="dash", line_color="#ff4d6d")
        fig.add_hline(y=0.99, line_dash="dot", line_color="#ffb000")
        fig.add_hline(y=0.95, line_dash="dot", line_color="#00ff88")
        fig.update_yaxes(range=[0.85, 2.05])
    return fig


def coin_chart_score(snapshot: Dict, asset: str) -> go.Figure:
    fig = base_fig(f"OPPORTUNITY SCORE – {asset}", height=220)
    hist = snapshot["coin5min"].get("market_history", {}).get(asset, [])
    sig = snapshot["coin5min"].get("signals", {}).get(asset, {})
    if hist:
        xs = list(range(len(hist)))
        # We only have current score in snapshot, so show a flat line until a proper score history is stored
        score = float(sig.get("opp_score", sig.get("opportunity_score", 0.0)) or 0.0)
        fig.add_trace(go.Scatter(x=xs, y=[score for _ in xs], mode="lines", name="Opp score"))
        fig.add_hline(y=0.70, line_dash="dash", line_color="#00ff88")
        fig.add_hline(y=0.50, line_dash="dot", line_color="#ffb000")
        fig.update_yaxes(range=[0, 1.0])
    return fig


def render_score_bar(score):
    if score is None:
        return html.Div("n/a", style={"fontFamily": M, "fontSize": "10px", "color": T["dim"]})
    score = float(score)
    pct = int(max(0, min(100, score * 100)))
    if score >= 0.7:
        color = T["grn"]
    elif score >= 0.5:
        color = T["amb"]
    else:
        color = T["red"]
    return html.Div([html.Div(style={"width": f"{pct}%", "height": "100%", "backgroundColor": color, "borderRadius": "4px"})], style={"width": "100px", "height": "8px", "backgroundColor": T["bg"], "border": f"1px solid {T['brd']}", "borderRadius": "4px"})


def decision_color(decision):
    if decision == "TRADE":
        return T["grn"]
    if decision == "WAIT":
        return T["amb"]
    return T["red"]


def build_opportunity_ranking(signals):
    rows = []
    for asset, s in (signals or {}).items():
        rows.append({
            "asset": asset,
            "score": float(s.get("opportunity_score", 0.0) or 0.0),
            "decision": s.get("decision", "SKIP"),
            "sum": s.get("exec_sum_yes_no", s.get("sum_yes_no")),
            "time_left": s.get("time_left_sec"),
            "trend": s.get("leg1_bias") or s.get("direction"),
        })
    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows


def ranking_table(signals):
    ranking = build_opportunity_ranking(signals)
    header = html.Tr([html.Th(c, style={"fontFamily": M, "fontSize": "10px", "color": T["dim"], "textAlign": "left", "padding": "6px"}) for c in ["Asset", "Score", "Heat", "Decision", "YES+NO", "Time Left", "Bias"]])
    body = []
    for r in ranking:
        body.append(html.Tr([
            html.Td(r["asset"], style={"padding": "6px", "fontFamily": M}),
            html.Td(f"{r['score']:.3f}", style={"padding": "6px", "fontFamily": M}),
            html.Td(render_score_bar(r["score"]), style={"padding": "6px"}),
            html.Td(r["decision"], style={"padding": "6px", "fontFamily": M, "fontWeight": "700", "color": decision_color(r["decision"])}),
            html.Td(f"{r['sum']:.3f}" if r["sum"] is not None else "n/a", style={"padding": "6px", "fontFamily": M}),
            html.Td(f"{r['time_left']:.1f}s" if r["time_left"] is not None else "n/a", style={"padding": "6px", "fontFamily": M}),
            html.Td(str(r["trend"]), style={"padding": "6px", "fontFamily": M}),
        ], style={"borderBottom": f"1px solid {T['brd']}"}))
    if not body:
        body = [html.Tr([html.Td("Aucune opportunité", colSpan=7, style={"padding": "8px", "fontFamily": M, "color": T["dim"]})])]
    return html.Table([html.Thead(header), html.Tbody(body)], style={"width": "100%", "borderCollapse": "collapse"})

# ============================================================
# Layout blocks
# ============================================================

def dashboard_page():
    return html.Div([
        card([
            html.Div("MODE D'EXÉCUTION", style={"fontFamily": M, "fontSize": "11px", "color": T["dim"], "marginBottom": "12px"}),
            html.Div([
                html.Label(["Mode : ", dcc.RadioItems(id="exec-mode", options=[{"label": "Paper", "value": "paper"}, {"label": "Live", "value": "live"}], value="paper", inline=True, persistence=True, persistence_type="session", style={"marginLeft": "8px"})], style={"fontFamily": M, "fontSize": "12px", "color": T["txt"], "marginRight": "18px"}),
                html.Label(["max_order_usd : ", dcc.Input(id="exec-max-order", type="number", min=1, max=50, step=1, value=3, style={"width": "60px", "marginLeft": "8px"})], style={"fontFamily": M, "fontSize": "12px", "color": T["txt"], "marginRight": "18px"}),
                html.Label(["Private Key : ", dcc.Input(id="exec-privkey", type="password", style={"width": "180px", "marginLeft": "8px"})], style={"fontFamily": M, "fontSize": "12px", "color": T["txt"], "marginRight": "18px"}),
                html.Label(["Funder : ", dcc.Input(id="exec-funder", type="text", style={"width": "140px", "marginLeft": "8px"})], style={"fontFamily": M, "fontSize": "12px", "color": T["txt"], "marginRight": "18px"}),
                html.Label(["Signature Type : ", dcc.Input(id="exec-sigtype", type="number", min=0, max=2, step=1, value=2, style={"width": "40px", "marginLeft": "8px"})], style={"fontFamily": M, "fontSize": "12px", "color": T["txt"], "marginRight": "18px"}),
                html.Label(["Chain ID : ", dcc.Input(id="exec-chainid", type="number", min=1, max=9999, step=1, value=137, style={"width": "60px", "marginLeft": "8px"})], style={"fontFamily": M, "fontSize": "12px", "color": T["txt"]}),
                button("APPLIQUER MODE", "exec-apply", T["cyan"]),
                html.Div(id="exec-msg", style={"fontFamily": M, "fontSize": "11px", "color": T["dim"], "marginLeft": "10px"}),
            ], style={"display": "flex", "flexWrap": "wrap", "gap": "8px", "alignItems": "center", "marginBottom": "10px"}),
        ]),
        card([
            html.Div("SESSION PAPER", style={"fontFamily": M, "fontSize": "11px", "color": T["dim"], "marginBottom": "12px"}),
            html.Div([
                html.Div([
                    html.Div("Capital total ($)", style={"fontFamily": M, "fontSize": "10px", "color": T["dim"], "marginBottom": "4px"}),
                    dcc.Input(id="cfg-capital", type="number", min=100, step=100, value=10000, persistence=True, persistence_type="session",
                              style={"width": "100%", "padding": "9px", "backgroundColor": T["bg"], "color": T["txt"], "border": f"1px solid {T['brd']}", "borderRadius": "6px", "fontFamily": M}),
                ], style={"flex": "1"}),
                html.Div([
                    html.Div("Allocation BetMiss (%)", style={"fontFamily": M, "fontSize": "10px", "color": T["dim"], "marginBottom": "4px"}),
                    dcc.Input(id="cfg-bm-alloc", type="number", min=0, max=100, step=1, value=50, persistence=True, persistence_type="session",
                              style={"width": "100%", "padding": "9px", "backgroundColor": T["bg"], "color": T["txt"], "border": f"1px solid {T['brd']}", "borderRadius": "6px", "fontFamily": M}),
                ], style={"flex": "1"}),
                html.Div([
                    html.Div("Allocation Coin5min (%)", style={"fontFamily": M, "fontSize": "10px", "color": T["dim"], "marginBottom": "4px"}),
                    dcc.Input(id="cfg-c5-alloc", type="number", min=0, max=100, step=1, value=50, persistence=True, persistence_type="session",
                              style={"width": "100%", "padding": "9px", "backgroundColor": T["bg"], "color": T["txt"], "border": f"1px solid {T['brd']}", "borderRadius": "6px", "fontFamily": M}),
                ], style={"flex": "1"}),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "14px"}),
            html.Div([button("APPLIQUER & RESET PAPER", "cfg-apply", T["cyan"]), html.Div(id="cfg-msg", style={"fontFamily": M, "fontSize": "11px", "color": T["dim"], "marginLeft": "10px"})], style={"display": "flex", "alignItems": "center", "gap": "10px"}),
        ]),
        html.Div(id="dash-kpis", style={"display": "flex", "gap": "10px", "flexWrap": "wrap"}),
        card(dcc.Graph(id="dash-pnl-fig", config={"displayModeBar": False})),
        card([html.Div("TRADES RÉCENTS", style={"fontFamily": M, "fontSize": "11px", "color": T["dim"], "marginBottom": "8px"}), table_component("dash-trades-table")]),
    ])


def betmiss_page():
    return html.Div(
        [
            html.Div(
                [
                    card(
                        [
                            html.Div("CONTRÔLES BETMISS", style={"fontFamily": M, "fontSize": "11px", "color": T["dim"], "marginBottom": "12px"}),
                            html.Div(id="bm-status-line", style={"marginBottom": "12px"}),
                            slider_block("bm-edge", "Seuil edge min", 0.5, 10, 0.5, 2.0, {1: "1%", 2: "2%", 5: "5%", 10: "10%"}),
                            slider_block("bm-size", "Taille/trade", 0.5, 10, 0.5, 2.0, {1: "1%", 2: "2%", 5: "5%", 10: "10%"}),
                            html.Div([button("START", "bm-start", T["grn"]), button("PAUSE", "bm-pause", T["amb"]), button("STOP", "bm-stop", T["red"])], style={"display": "flex", "gap": "8px"}),
                            html.Div(id="bm-action-msg", style={"fontFamily": M, "fontSize": "11px", "color": T["dim"], "marginTop": "10px"}),
                        ],
                        width="280px",
                        flexShrink="0",
                    ),
                    html.Div(
                        [
                            html.Div(id="bm-kpis", style={"display": "flex", "gap": "10px", "flexWrap": "wrap"}),
                            card([html.Div("OPPORTUNITÉS LIVE", style={"fontFamily": M, "fontSize": "11px", "color": T["dim"], "marginBottom": "8px"}), table_component("bm-opp-table")]),
                            card([html.Div("TRADES BETMISS", style={"fontFamily": M, "fontSize": "11px", "color": T["dim"], "marginBottom": "8px"}), table_component("bm-trades-table")]),
                        ],
                        style={"flex": "1", "minWidth": "0"},
                    ),
                ],
                style={"display": "flex", "gap": "14px", "alignItems": "flex-start"},
            )
        ]
    )


def coin5min_page():
    return html.Div(
        [
            html.Div(
                [
                    card(
                        [
                            html.Div("CONTRÔLES COIN5MIN", style={"fontFamily": M, "fontSize": "11px", "color": T["dim"], "marginBottom": "12px"}),
                            html.Div(id="c5-status-line", style={"marginBottom": "12px"}),
                            slider_block("c5-edge", "Seuil edge min (YES+NO)", 0.5, 3.0, 0.1, 1.0, {0.5: "0.5%", 1: "1%", 2: "2%", 3: "3%"}),
                            slider_block("c5-size", "Taille/trade", 1, 10, 0.5, 5.0, {1: "1%", 5: "5%", 10: "10%"}),
                            html.Div("Intervalle du marché Up/Down", style={"fontFamily": M, "fontSize": "10px", "color": T["dim"], "marginBottom": "4px"}),
                            dcc.Dropdown(id="c5-int", options=[{"label": f"{x} min", "value": x} for x in [5, 15]], value=5, clearable=False, persistence=True, persistence_type="session", style={"marginBottom": "12px", "fontFamily": M}),
                            html.Div("Actifs", style={"fontFamily": M, "fontSize": "10px", "color": T["dim"], "marginBottom": "4px"}),
                            dcc.Checklist(id="c5-assets", options=[{"label": f" {a}", "value": a} for a in ["BTC", "ETH", "SOL", "XRP", "DOGE"]], value=["BTC", "ETH"], persistence=True, persistence_type="session", inputStyle={"marginRight": "6px"}, labelStyle={"display": "block", "marginBottom": "4px", "fontFamily": M, "fontSize": "12px", "color": T["txt"]}),
                            html.Hr(style={"borderColor": T["brd"], "margin": "14px 0"}),
                            html.Div("PANNEAU LEGGED ARB", style={"fontFamily": M, "fontSize": "11px", "color": T["dim"], "marginBottom": "10px"}),
                            html.Div("Source spot / ref proxy", style={"fontFamily": M, "fontSize": "10px", "color": T["dim"], "marginBottom": "4px"}),
                            dcc.Dropdown(id="c5-spot-source", options=[{"label": s.capitalize(), "value": s} for s in ["binance", "coinbase"]], value="binance", clearable=False, persistence=True, persistence_type="session", style={"marginBottom": "12px", "fontFamily": M}),
                            html.Div("Délai max entre jambe 1 et 2 (s)", style={"fontFamily": M, "fontSize": "10px", "color": T["dim"], "marginBottom": "4px"}),
                            dcc.Input(id="c5-max-leg-hold", type="number", min=15, max=240, step=5, value=180, persistence=True, persistence_type="session", style={"width": "100%", "padding": "9px", "backgroundColor": T["bg"], "color": T["txt"], "border": f"1px solid {T['brd']}", "borderRadius": "6px", "fontFamily": M, "marginBottom": "12px"}),
                            html.Div("Prix max de la première jambe", style={"fontFamily": M, "fontSize": "10px", "color": T["dim"], "marginBottom": "4px"}),
                            dcc.Input(id="c5-max-leg1", type="number", min=0.1, max=0.99, step=0.01, value=0.80, persistence=True, persistence_type="session", style={"width": "100%", "padding": "9px", "backgroundColor": T["bg"], "color": T["txt"], "border": f"1px solid {T['brd']}", "borderRadius": "6px", "fontFamily": M, "marginBottom": "12px"}),
                            html.Div("Seuil max de somme YES+NO", style={"fontFamily": M, "fontSize": "10px", "color": T["dim"], "marginBottom": "4px"}),
                            dcc.Input(id="c5-max-sum", type="number", min=0.85, max=0.999, step=0.01, value=0.99, persistence=True, persistence_type="session", style={"width": "100%", "padding": "9px", "backgroundColor": T["bg"], "color": T["txt"], "border": f"1px solid {T['brd']}", "borderRadius": "6px", "fontFamily": M, "marginBottom": "12px"}),
                            html.Div("$ max par trade (cap de risque)", style={"fontFamily": M, "fontSize": "10px", "color": T["dim"], "marginBottom": "4px"}),
                            dcc.Input(id="c5-max-trade-usd", type="number", min=1, max=1000, step=1, value=50, persistence=True, persistence_type="session", style={"width": "100%", "padding": "9px", "backgroundColor": T["bg"], "color": T["txt"], "border": f"1px solid {T['brd']}", "borderRadius": "6px", "fontFamily": M, "marginBottom": "12px"}),
                            dcc.Checklist(id="c5-directional-leg1", options=[{"label": " Jambe 1 directionnelle (sinon prendre la jambe la moins chère)", "value": "on"}], value=["on"], persistence=True, persistence_type="session", inputStyle={"marginRight": "6px"}, labelStyle={"display": "block", "marginBottom": "10px", "fontFamily": M, "fontSize": "12px", "color": T["txt"]}),
                            dcc.Checklist(id="c5-realistic-fill", options=[{"label": " Contrôle de fill réaliste (refetch book avant chaque ordre)", "value": "on"}], value=["on"], persistence=True, persistence_type="session", inputStyle={"marginRight": "6px"}, labelStyle={"display": "block", "marginBottom": "10px", "fontFamily": M, "fontSize": "12px", "color": T["txt"]}),
                            dcc.Checklist(id="c5-analysis-log", options=[{"label": " Enregistrer les trades dans coin5min_analysis.jsonl", "value": "on"}], value=[], persistence=True, persistence_type="session", inputStyle={"marginRight": "6px"}, labelStyle={"display": "block", "marginBottom": "10px", "fontFamily": M, "fontSize": "12px", "color": T["txt"]}),
                            html.Div(id="c5-analysis-status", style={"fontFamily": M, "fontSize": "10px", "color": T["dim"], "marginBottom": "8px"}),
                            html.Div([button("START", "c5-start", T["grn"]), button("PAUSE", "c5-pause", T["amb"]), button("STOP", "c5-stop", T["red"])], style={"display": "flex", "gap": "8px", "marginTop": "10px"}),
                            html.Div(id="c5-action-msg", style={"fontFamily": M, "fontSize": "11px", "color": T["dim"], "marginTop": "10px"}),
                            html.Hr(style={"borderColor": T["brd"], "margin": "14px 0"}),
                            html.Div("Actif affiché sur le chart", style={"fontFamily": M, "fontSize": "10px", "color": T["dim"], "marginBottom": "4px"}),
                            dcc.Dropdown(id="c5-chart-asset", options=[{"label": a, "value": a} for a in ["BTC", "ETH", "SOL", "XRP", "DOGE"]], value="BTC", clearable=False, persistence=True, persistence_type="session", style={"fontFamily": M}),
                        ], width="300px", flexShrink="0"
                    ),
                    html.Div(
                        [
                            html.Div(id="c5-kpis", style={"display": "flex", "gap": "10px", "flexWrap": "wrap"}),
                            card(html.Div(id="c5-warmup-block")),
                            card([html.Div("MARCHÉ / EXÉCUTION LIVE", style={"fontFamily": M, "fontSize": "11px", "color": T["dim"], "marginBottom": "8px"}), html.Div(id="c5-market-live-block")]),
                            card([html.Div("CARNET D'ORDRES POLYMARKET (L2)", style={"fontFamily": M, "fontSize": "11px", "color": T["dim"], "marginBottom": "8px"}), html.Div(id="c5-orderbook-block")]),
                            card([html.Div("🔥 Opportunity Ranking (exec_sum focalisé)", style={"fontFamily": M, "fontSize": "11px", "color": T["dim"], "marginBottom": "8px"}), html.Div(id="c5-ranking-block")]),
                            card(dcc.Graph(id="c5-chart-spot", config={"displayModeBar": False})),
                            card(dcc.Graph(id="c5-chart-yesno", config={"displayModeBar": False})),
                            card(dcc.Graph(id="c5-chart-sum", config={"displayModeBar": False})),
                            card(dcc.Graph(id="c5-chart-score", config={"displayModeBar": False})),
                            card([html.Div("SIGNAUX / SPOT / CARNET", style={"fontFamily": M, "fontSize": "11px", "color": T["dim"], "marginBottom": "8px"}), table_component("c5-signals-table")]),
                            card([html.Div("TRADES COIN5MIN", style={"fontFamily": M, "fontSize": "11px", "color": T["dim"], "marginBottom": "8px"}), table_component("c5-trades-table")]),
                        ], style={"flex": "1", "minWidth": "0"}
                    ),
                ], style={"display": "flex", "gap": "14px", "alignItems": "flex-start"}
            )
        ]
    )


def alerts_page():
    return card([html.Div("JOURNAL SYSTÈME", style={"fontFamily": M, "fontSize": "11px", "color": T["dim"], "marginBottom": "8px"}), html.Div(id="alerts-block")])


# ============================================================
# App layout
# ============================================================

app = Dash(__name__, suppress_callback_exceptions=True, title="SAXIFRAGE")

app.layout = html.Div(
    [
        dcc.Store(id="ui-refresh", data=0),
        html.Div(
            [
                html.Div([
                    html.Span(
                        [
                            html.Img(
                                src="data:image/svg+xml;utf8,<svg width='28' height='28' viewBox='0 0 32 32' fill='none' xmlns='http://www.w3.org/2000/svg'><circle cx='16' cy='16' r='15' fill='%23e0f7fa' stroke='%2300b894' stroke-width='2'/><path d='M16 6 L18 16 L16 26 L14 16 Z' fill='%2300b894'/><circle cx='16' cy='16' r='4' fill='%23fffde7' stroke='%23b2a300' stroke-width='1.5'/></svg>",
                                style={"height": "28px", "verticalAlign": "middle", "marginRight": "10px"},
                            ),
                            html.Span("SAXIFRAGE", style={"fontFamily": M, "fontSize": "20px", "fontWeight": "700", "color": T["cyan"]}),
                        ],
                        style={"display": "flex", "alignItems": "center"},
                    ),
                    badge("PAPER LIVE", T["grn"]),
                ], style={"display": "flex", "alignItems": "center", "gap": "10px"}),
                html.Div(id="header-status", style={"fontFamily": M, "fontSize": "12px", "color": T["txt"]}),
            ],
            style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "padding": "10px 18px", "borderBottom": f"1px solid {T['brd']}", "backgroundColor": T["card"]},
        ),
        dcc.Tabs(
            id="tabs",
            value="dashboard",
            children=[
                dcc.Tab(label="Dashboard", value="dashboard"),
                dcc.Tab(label="BetMiss", value="betmiss"),
                dcc.Tab(label="Coin5min", value="coin5min"),
                dcc.Tab(label="Alertes", value="alerts"),
            ],
            colors={"border": T["brd"], "primary": T["cyan"], "background": T["bg"]},
            style={"fontFamily": M},
        ),
        html.Div(
            [
                html.Div(id="page-dashboard", children=dashboard_page()),
                html.Div(id="page-betmiss", children=betmiss_page(), style={"display": "none"}),
                html.Div(id="page-coin5min", children=coin5min_page(), style={"display": "none"}),
                html.Div(id="page-alerts", children=alerts_page(), style={"display": "none"}),
            ],
            style={"padding": "16px", "minHeight": "calc(100vh - 132px)", "backgroundColor": T["bg"]},
        ),
        dcc.Interval(id="tick", interval=2000, n_intervals=0),
    ],
    style={"backgroundColor": T["bg"], "minHeight": "100vh", "color": T["txt"]},
)

app.index_string = f"""<!DOCTYPE html>
<html>
<head>
{{%metas%}}
<title>SAXIFRAGE</title>
{{%favicon%}}
{{%css%}}
<link href='https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap' rel='stylesheet'>
<style>
body {{ margin:0; background:{T['bg']}; }}
.tab {{ background:{T['bg']}!important; color:{T['dim']}!important; border:none!important; font-family:{M}!important; }}
.tab--selected {{ background:{T['card']}!important; color:{T['cyan']}!important; border-bottom:2px solid {T['cyan']}!important; }}
input[type=checkbox],input[type=radio],input[type=range] {{ accent-color: {T['cyan']}; }}
::-webkit-scrollbar {{ width:6px; }}
::-webkit-scrollbar-thumb {{ background:{T['brd']}; border-radius:3px; }}
.Select-control,.Select-menu-outer,.VirtualizedSelectOption {{ background:{T['bg']}!important; color:{T['txt']}!important; border-color:{T['brd']}!important; }}
</style>
</head>
<body>
{{%app_entry%}}
<footer>
{{%config%}}
{{%scripts%}}
{{%renderer%}}
</footer>
</body>
</html>"""


# ============================================================
# Simple visibility + live control value text
# ============================================================

@app.callback(
    Output("page-dashboard", "style"),
    Output("page-betmiss", "style"),
    Output("page-coin5min", "style"),
    Output("page-alerts", "style"),
    Input("tabs", "value"),
)
def switch_tab(tab):
    show = {"display": "block"}
    hide = {"display": "none"}
    return (
        show if tab == "dashboard" else hide,
        show if tab == "betmiss" else hide,
        show if tab == "coin5min" else hide,
        show if tab == "alerts" else hide,
    )


@app.callback(Output("bm-edge-val", "children"), Input("bm-edge", "value"))
def bm_edge_value(v):
    return f"{v:.1f}%"


@app.callback(Output("bm-size-val", "children"), Input("bm-size", "value"))
def bm_size_value(v):
    return f"{v:.1f}%"


@app.callback(Output("c5-edge-val", "children"), Input("c5-edge", "value"))
def c5_edge_value(v):
    return f"{v:.1f}%"


@app.callback(Output("c5-size-val", "children"), Input("c5-size", "value"))
def c5_size_value(v):
    return f"{v:.1f}%"


# ============================================================
# Config + action callbacks

# Callback pour appliquer la config live/paper
from dash.exceptions import PreventUpdate

@app.callback(
    Output("exec-msg", "children"),
    Input("exec-apply", "n_clicks"),
    State("exec-mode", "value"),
    State("exec-max-order", "value"),
    State("exec-privkey", "value"),
    State("exec-funder", "value"),
    State("exec-sigtype", "value"),
    State("exec-chainid", "value"),
    prevent_initial_call=True,
)
def apply_exec_config(_n, mode, max_order, privkey, funder, sigtype, chainid):
    if not mode:
        raise PreventUpdate
    ok, msg = ST.set_execution_config(
        mode=mode,
        max_order_usd=max_order or 3.0,
        private_key=privkey or "",
        funder=funder or "",
        signature_type=sigtype or 2,
        chain_id=chainid or 137,
    )
    # Affiche le solde si live
    if ok and mode == "live":
        bal = None
        try:
            bal = ST.execution.get_balance() if ST.execution else None
        except Exception:
            bal = None
        if bal is not None:
            msg += f" | Solde USDC: ${bal:.2f}"
    return msg
# ============================================================

@app.callback(
    Output("cfg-msg", "children"),
    Input("cfg-apply", "n_clicks"),
    State("cfg-capital", "value"),
    State("cfg-bm-alloc", "value"),
    State("cfg-c5-alloc", "value"),
    prevent_initial_call=True,
)
def apply_config(_n, capital, bm_alloc, c5_alloc):
    ok, msg = ST.reset_session(capital=capital, bm_alloc=bm_alloc, c5_alloc=c5_alloc)
    return msg


@app.callback(
    Output("bm-action-msg", "children"),
    Input("bm-start", "n_clicks"),
    Input("bm-pause", "n_clicks"),
    Input("bm-stop", "n_clicks"),
    prevent_initial_call=True,
)
def bm_actions(_start, _pause, _stop):
    trig = ctx.triggered_id
    if trig == "bm-start":
        ST.start_strategy("betmiss")
        return "BetMiss démarrée."
    if trig == "bm-pause":
        ST.pause_strategy("betmiss")
        return ST.snap()["betmiss"]["last_action"]
    if trig == "bm-stop":
        ST.stop_strategy("betmiss")
        return "BetMiss arrêtée."
    return ""


@app.callback(
    Output("c5-analysis-status", "children"),
    Input("tick", "n_intervals"),
    Input("c5-realistic-fill", "value"),
    Input("c5-analysis-log", "value"),
)
def c5_toggle_realism(_tick, realistic_fill_v, analysis_log_v):
    realistic = bool(realistic_fill_v and "on" in realistic_fill_v)
    analysis = bool(analysis_log_v and "on" in analysis_log_v)
    ST.update_coin5min_params(
        realistic_fill_enabled=realistic,
        analysis_log_enabled=analysis,
    )
    s = ST.snap()
    c5 = s["coin5min"]
    n_events = c5.get("analysis_events_count", 0)
    n_accept = c5.get("fill_accepted_count", 0)
    n_reject = c5.get("fill_rejected_count", 0)
    log_path = c5.get("analysis_log_path", "coin5min_analysis.jsonl")
    parts = []
    parts.append(f"Fill check: {'ON' if realistic else 'OFF'} (accept {n_accept} / reject {n_reject})")
    if analysis:
        parts.append(f"Log → {log_path} ({n_events} events)")
    else:
        parts.append("Log: OFF")
    return " | ".join(parts)


@app.callback(
    Output("c5-action-msg", "children"),
    Input("c5-start", "n_clicks"),
    Input("c5-pause", "n_clicks"),
    Input("c5-stop", "n_clicks"),
    prevent_initial_call=True,
)
def c5_actions(_start, _pause, _stop):
    trig = ctx.triggered_id
    if trig == "c5-start":
        ST.start_strategy("coin5min")
        return "Coin5min démarrée. Warmup en cours jusqu'à 100 points par actif."
    if trig == "c5-pause":
        ST.pause_strategy("coin5min")
        return ST.snap()["coin5min"]["last_action"]
    if trig == "c5-stop":
        ST.stop_strategy("coin5min")
        return "Coin5min arrêtée."
    return ""


@app.callback(
    Output("header-status", "children"),
    Output("dash-kpis", "children"),
    Output("dash-pnl-fig", "figure"),
    Output("dash-trades-table", "data"),
    Output("dash-trades-table", "columns"),
    Input("tick", "n_intervals"),
)
def refresh_dashboard(_tick):
    ST.auto_alerts()
    s = ST.snap()
    pf = s["portfolio"]
    header = html.Span(
        [
            html.Span("Capital ", style={"color": T["dim"]}),
            html.Span(f"${pf['total_value']:,.2f}", style={"fontWeight": "700", "color": T["wht"]}),
            html.Span(f"   PnL {pf['total_pnl']:+,.2f}", style={"color": T['grn'] if pf['total_pnl'] >= 0 else T['red']}),
        ]
    )
    trades = s["recent_trades"]
    n_coin = sum(1 for t in trades if t.get("strategy") == "coin5min")
    n_bm = sum(1 for t in trades if t.get("strategy") == "betmiss")
    wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
    kpis = [
        kpi("Valeur totale", f"${pf['total_value']:,.0f}"),
        kpi("PnL total", f"${pf['total_pnl']:+,.0f}", color=T["grn"] if pf["total_pnl"] >= 0 else T["red"]),
        kpi("Trades", str(len(trades)), f"BM {n_bm} | C5 {n_coin}"),
        kpi("Win rate", f"{(100*wins/max(len(trades),1)):.0f}%"),
    ]
    rows = [
        {
            "time": t.get("time"),
            "strategy": t.get("strategy"),
            "asset": t.get("asset"),
            "side": t.get("side"),
            "price": t.get("price"),
            "size": t.get("size"),
            "status": t.get("status"),
            "pnl": f"{t.get('pnl',0):+.2f}",
            "note": t.get("note", ""),
        }
        for t in trades[:20]
    ]
    cols = [{"name": c.upper(), "id": c} for c in ["time", "strategy", "asset", "side", "price", "size", "status", "pnl", "note"]]
    return header, kpis, pnl_figure(), rows, cols


@app.callback(
    Output("bm-status-line", "children"),
    Output("bm-kpis", "children"),
    Output("bm-opp-table", "data"),
    Output("bm-opp-table", "columns"),
    Output("bm-trades-table", "data"),
    Output("bm-trades-table", "columns"),
    Input("tick", "n_intervals"),
    Input("bm-edge", "value"),
    Input("bm-size", "value"),
)
def refresh_betmiss(_tick, edge, size):
    ST.update_betmiss_params(min_edge_pct=edge, trade_size_pct=size)
    s = ST.snap()
    bm = s["betmiss"]
    status = html.Div([
        badge((bm.get("status") or "stopped").upper(), status_color(bm.get("status", "stopped"))),
        html.Span(f"  {bm.get('last_action','')}", style={"fontFamily": M, "fontSize": "11px", "color": T["dim"], "marginLeft": "8px"}),
    ])
    bm_trades = [t for t in s["recent_trades"] if t.get("strategy") == "betmiss"]
    pnl = sum(float(t.get("pnl", 0)) for t in bm_trades)
    kpis = [
        kpi("Status", (bm.get("status") or "stopped").upper(), color=status_color(bm.get("status", "stopped"))),
        kpi("Allocation", f"{s['allocations']['betmiss']:.0f}%"),
        kpi("PnL BM", f"${pnl:+.0f}", color=T["grn"] if pnl >= 0 else T["red"]),
        kpi("Opps", str(len(bm.get("opportunities", [])))),
    ]
    opp_rows = bm.get("opportunities", []) or []
    opp_cols = [{"name": c.upper(), "id": c} for c in ["asset", "event", "buy_yes_on", "buy_no_on", "yes_price", "no_price", "edge"]]
    trade_rows = [
        {"time": t.get("time"), "asset": t.get("asset"), "side": t.get("side"), "size": t.get("size"), "pnl": f"{t.get('pnl',0):+.2f}", "status": t.get("status"), "note": t.get("note", "")}
        for t in bm_trades[:20]
    ]
    trade_cols = [{"name": c.upper(), "id": c} for c in ["time", "asset", "side", "size", "pnl", "status", "note"]]
    return status, kpis, opp_rows, opp_cols, trade_rows, trade_cols


@app.callback(
    Output("c5-status-line", "children"),
    Output("c5-kpis", "children"),
    Output("c5-warmup-block", "children"),
    Output("c5-ranking-block", "children"),
    Output("c5-market-live-block", "children"),
    Output("c5-chart-spot", "figure"),
    Output("c5-chart-yesno", "figure"),
    Output("c5-chart-sum", "figure"),
    Output("c5-chart-score", "figure"),
    Output("c5-signals-table", "data"),
    Output("c5-signals-table", "columns"),
    Output("c5-trades-table", "data"),
    Output("c5-trades-table", "columns"),
    Input("tick", "n_intervals"),
    Input("c5-edge", "value"),
    Input("c5-size", "value"),
    Input("c5-int", "value"),
    Input("c5-assets", "value"),
    Input("c5-chart-asset", "value"),
    Input("c5-max-leg-hold", "value"),
    Input("c5-max-leg1", "value"),
    Input("c5-max-sum", "value"),
    Input("c5-max-trade-usd", "value"),
    Input("c5-directional-leg1", "value"),
    Input("c5-spot-source", "value"),
)
def refresh_coin5min(_tick, edge, size, interval_min, assets, chart_asset, max_leg_hold, max_leg1, max_sum, max_trade_usd, directional_leg1, spot_source):
    s = ST.snap()
    c5 = s["coin5min"]
    status = html.Div([
        badge((c5.get("status") or "stopped").upper(), status_color(c5.get("status", "stopped"))),
        html.Span(f"  {c5.get('last_action','')}", style={"fontFamily": M, "fontSize": "11px", "color": T["dim"], "marginLeft": "8px"}),
    ])
    c5_trades = [t for t in s["recent_trades"] if t.get("strategy") == "coin5min"]
    pnl = sum(float(t.get("pnl", 0)) for t in c5_trades)
    signals = c5.get("signals", {})
    best_sum = None
    for sig in signals.values():
        val = sig.get("exec_sum_yes_no", sig.get("sum_yes_no"))
        if val is not None:
            best_sum = val if best_sum is None else min(best_sum, val)
    kpis = [
        kpi("Status", (c5.get("status") or "stopped").upper(), color=status_color(c5.get("status", "stopped"))),
        kpi("Allocation", f"{s['allocations']['coin5min']:.0f}%"),
        kpi("PnL C5", f"${pnl:+.0f}", color=T["grn"] if pnl >= 0 else T["red"]),
        kpi("Meilleur YES+NO", f"{best_sum:.4f}" if best_sum is not None else "n/a", sub=f"edge {((1-best_sum)*100):+.2f}%" if best_sum is not None else "", color=T["cyan"]),
    ]

    warmup_points = c5.get("warmup_points", {})
    warmup_items = [html.Div(f"Warmup Coin5min: {c5.get('warmup_elapsed_sec',0)}s écoulées | historique chargé", style={"fontFamily": M, "fontSize": "11px", "color": T["dim"], "marginBottom": "10px"})]
    for asset in assets or []:
        pts = warmup_points.get(asset, 0)
        ratio = min(1.0, pts / max(c5.get("warmup_required", 8), 1))
        warmup_items.append(html.Div([
            html.Div([html.Span(asset, style={"fontFamily": M, "fontSize": "12px", "color": T["wht"]}), html.Span(f"{pts}/{c5.get('warmup_required',8)}", style={"fontFamily": M, "fontSize": "11px", "color": T["dim"]})], style={"display": "flex", "justifyContent": "space-between"}),
            html.Div(style={"height": "8px", "backgroundColor": T["bg"], "border": f"1px solid {T['brd']}", "borderRadius": "999px", "overflow": "hidden", "marginTop": "4px"}, children=[html.Div(style={"height": "100%", "width": f"{100*ratio:.1f}%", "backgroundColor": status_color("running" if ratio >= 1 else "warming")})]),
        ], style={"marginBottom": "8px"}))

    signal_rows = []
    for asset in assets or []:
        sig = signals.get(asset, {})
        spot = c5.get("spot_rows", {}).get(asset, {})
        meta = c5.get("market_meta", {}).get(asset, {})
        signal_rows.append({
            "asset": asset,
            "warmup": warmup_points.get(asset, 0),
            "interval": sig.get("interval_min") or interval_min,
            "ref_px": meta.get("ref_px"),
            "spot": round(spot.get("price", 0), 4) if spot else None,
            "spot_src": spot.get("source") if spot else None,
            "time_left_s": sig.get("time_left_sec"),
            "yes_mark": sig.get("yes_mark_price"),
            "no_mark": sig.get("no_mark_price"),
            "yes_exec": sig.get("yes_exec_buy", sig.get("yes_ask")),
            "no_exec": sig.get("no_exec_buy", sig.get("no_ask")),
            "mark_sum": sig.get("mark_sum_yes_no"),
            "exec_sum": sig.get("exec_sum_yes_no", sig.get("sum_yes_no")),
            "target_sum": sig.get("target_sum_yes_no", c5.get("target_sum_yes_no")),
            "edge_exec": f"{100*sig.get('edge_exec',0):+.2f}%" if sig.get("edge_exec") is not None else "",
            "state": sig.get("direction"),
            "leg_state": sig.get("leg_state"),
            "hedge_at": sig.get("hedge_trigger"),
            "budget_pct": sig.get("budget_pct"),
            "budget_cap_usd": sig.get("budget_cap_usd"),
            "budget_effective_usd": sig.get("budget_effective_usd"),
            "trend_bias_used": sig.get("trend_bias_used"),
            "opp_score": sig.get("opp_score", sig.get("opportunity_score")),
            "sum_score": sig.get("sum_score"),
            "trend_score": sig.get("trend_score_component"),
            "spread_score": sig.get("spread_score"),
            "time_score": sig.get("time_score"),
            "decision": sig.get("decision"),
            "decision_reason": sig.get("decision_reason"),
            "prob_score": sig.get("prob_score"),
            "p_up": sig.get("p_up"),
            "p_down": sig.get("p_down"),
            "risk_ok": sig.get("risk_ok"),
            "risk_reason": sig.get("risk_reason"),
        })
    signal_cols = [{"name": c.upper(), "id": c} for c in [
        "asset", "warmup", "interval", "ref_px", "spot", "spot_src", "time_left_s",
        "yes_mark", "no_mark", "yes_exec", "no_exec", "mark_sum", "exec_sum",
        "target_sum", "edge_exec", "state", "leg_state", "hedge_at", "budget_pct",
        "budget_cap_usd", "budget_effective_usd", "trend_bias_used", "opp_score",
        "sum_score", "trend_score", "spread_score", "time_score", "decision", "decision_reason"
    ]]

    trade_rows = [{"time": t.get("time"), "asset": t.get("asset"), "side": t.get("side"), "size": t.get("size"), "status": t.get("status"), "pnl": f"{t.get('pnl',0):+.2f}", "note": t.get("note", "")} for t in c5_trades[:20]]
    live_asset = chart_asset if chart_asset in (assets or []) else ((assets or ["BTC"])[0])
    live_sig = signals.get(live_asset, {})
    live_meta = c5.get("market_meta", {}).get(live_asset, {})
    market_live = html.Div([
        html.Div(f"{live_meta.get('question','Aucun marché détecté')}", style={"fontFamily": M, "fontSize": "14px", "fontWeight": "700", "color": T["wht"], "marginBottom": "10px"}),
        html.Div([
            html.Div([html.Div("Asset", style={"fontFamily": M, "fontSize": "10px", "color": T["dim"]}), html.Div(str(live_asset), style={"fontFamily": M, "fontSize": "16px", "fontWeight": "700", "color": T["cyan"]})], style={"minWidth": "120px"}),
            html.Div([html.Div("Prix à battre", style={"fontFamily": M, "fontSize": "10px", "color": T["dim"]}), html.Div(str(live_meta.get("ref_px","n/a")), style={"fontFamily": M, "fontSize": "16px", "fontWeight": "700", "color": T["wht"]})], style={"minWidth": "140px"}),
            html.Div([html.Div("Spot live", style={"fontFamily": M, "fontSize": "10px", "color": T["dim"]}), html.Div(str(round((c5.get("spot_rows", {}).get(live_asset, {}) or {}).get("price", 0.0), 4)), style={"fontFamily": M, "fontSize": "16px", "fontWeight": "700", "color": T["wht"]})], style={"minWidth": "140px"}),
            html.Div([html.Div("Temps restant", style={"fontFamily": M, "fontSize": "10px", "color": T["dim"]}), html.Div(f"{live_sig.get('time_left_sec','n/a')}s", style={"fontFamily": M, "fontSize": "16px", "fontWeight": "700", "color": T["amb"]})], style={"minWidth": "140px"}),
        ], style={"display": "flex", "gap": "24px", "flexWrap": "wrap", "marginBottom": "10px"}),
        html.Div(
            f"YES mark {live_sig.get('yes_mark_price','n/a')} | NO mark {live_sig.get('no_mark_price','n/a')} | "
            f"YES exec {live_sig.get('yes_exec_buy', live_sig.get('yes_ask','n/a'))} | NO exec {live_sig.get('no_exec_buy', live_sig.get('no_ask','n/a'))} | "
            f"mark_sum {live_sig.get('mark_sum_yes_no','n/a')} | exec_sum {live_sig.get('exec_sum_yes_no', live_sig.get('sum_yes_no','n/a'))} | "
            f"target_sum {live_sig.get('target_sum_yes_no','n/a')} | p_up {live_sig.get('p_up','n/a')} | p_down {live_sig.get('p_down','n/a')} | "
            f"decision {live_sig.get('decision','n/a')} | reason {live_sig.get('decision_reason','n/a')} | risk {live_sig.get('risk_ok','n/a')}/{live_sig.get('risk_reason','')}",
            style={"fontFamily": M, "fontSize": "12px", "color": T["txt"]}
        )
    ])
    trade_cols = [{"name": c.upper(), "id": c} for c in ["time", "asset", "side", "size", "status", "pnl", "note"]]

    return status, kpis, warmup_items, ranking_table(signals), market_live, coin_chart_spot(s, chart_asset), coin_chart_yesno(s, chart_asset), coin_chart_sum(s, chart_asset), coin_chart_score(s, chart_asset), signal_rows, signal_cols, trade_rows, trade_cols


def _orderbook_ladder(side_name: str, asks: List[Dict], bids: List[Dict], bot_orders: List[Dict]):
    """Render one side (YES or NO) as a vertical L2 ladder.

    asks: list of {price, size} sorted ascending (best ask first)
    bids: list of {price, size} sorted descending (best bid first)
    bot_orders: list of {price, size_usd, label, color} placed by the bot on this side
    """
    # build display order: asks high → low, then bids high → low
    asks_disp = list(reversed(asks))  # so the highest ask appears at the top
    rows = []

    def _format_size(sz: float) -> str:
        if sz is None:
            return "—"
        if sz >= 1000:
            return f"{sz/1000:.1f}k"
        return f"{sz:.0f}"

    def _bot_marker(price: float):
        """Return marker for any bot order whose price matches this level (±0.005)."""
        matches = [o for o in bot_orders if abs(float(o.get("price", -1)) - float(price)) <= 0.005]
        if not matches:
            return None
        labels = " ".join(f"◀ {o['label']} ${o.get('size_usd', 0):.0f}" for o in matches)
        return html.Span(labels, style={"fontFamily": M, "fontSize": "10px", "color": T["amb"], "marginLeft": "6px", "fontWeight": "700"})

    # Header
    rows.append(html.Div([
        html.Div(f"{side_name} BOOK", style={"fontFamily": M, "fontSize": "11px", "fontWeight": "700", "color": T["wht"], "flex": "1"}),
        html.Div("Price", style={"fontFamily": M, "fontSize": "10px", "color": T["dim"], "width": "55px", "textAlign": "right"}),
        html.Div("Size", style={"fontFamily": M, "fontSize": "10px", "color": T["dim"], "width": "55px", "textAlign": "right"}),
        html.Div("", style={"width": "150px"}),
    ], style={"display": "flex", "gap": "10px", "padding": "4px 0", "borderBottom": f"1px solid {T['brd']}"}))

    # Ask rows (red)
    for lvl in asks_disp:
        p = float(lvl["price"])
        s = float(lvl["size"])
        marker = _bot_marker(p)
        rows.append(html.Div([
            html.Div("ASK", style={"fontFamily": M, "fontSize": "10px", "color": T["red"], "width": "40px"}),
            html.Div("", style={"flex": "1"}),
            html.Div(f"{p:.4f}", style={"fontFamily": M, "fontSize": "12px", "color": T["red"], "width": "55px", "textAlign": "right"}),
            html.Div(_format_size(s), style={"fontFamily": M, "fontSize": "11px", "color": T["dim"], "width": "55px", "textAlign": "right"}),
            html.Div(marker or "", style={"width": "150px"}),
        ], style={"display": "flex", "gap": "10px", "padding": "2px 0"}))

    # Spread separator
    if asks and bids:
        spread = float(asks[0]["price"]) - float(bids[0]["price"])
        rows.append(html.Div([
            html.Div(f"── spread {spread:+.4f} ──", style={"fontFamily": M, "fontSize": "10px", "color": T["amb"], "textAlign": "center", "flex": "1"}),
        ], style={"display": "flex", "padding": "3px 0", "borderTop": f"1px dashed {T['brd']}", "borderBottom": f"1px dashed {T['brd']}", "margin": "2px 0"}))
    elif not asks and not bids:
        rows.append(html.Div("Carnet vide ou indisponible", style={"fontFamily": M, "fontSize": "11px", "color": T["dim"], "padding": "8px 0"}))

    # Bid rows (green)
    for lvl in bids:
        p = float(lvl["price"])
        s = float(lvl["size"])
        marker = _bot_marker(p)
        rows.append(html.Div([
            html.Div("BID", style={"fontFamily": M, "fontSize": "10px", "color": T["grn"], "width": "40px"}),
            html.Div("", style={"flex": "1"}),
            html.Div(f"{p:.4f}", style={"fontFamily": M, "fontSize": "12px", "color": T["grn"], "width": "55px", "textAlign": "right"}),
            html.Div(_format_size(s), style={"fontFamily": M, "fontSize": "11px", "color": T["dim"], "width": "55px", "textAlign": "right"}),
            html.Div(marker or "", style={"width": "150px"}),
        ], style={"display": "flex", "gap": "10px", "padding": "2px 0"}))

    return html.Div(rows, style={"flex": "1", "minWidth": "0", "padding": "8px", "backgroundColor": T["bg"], "border": f"1px solid {T['brd']}", "borderRadius": "6px"})


@app.callback(
    Output("c5-orderbook-block", "children"),
    Input("tick", "n_intervals"),
    Input("c5-chart-asset", "value"),
)
def refresh_orderbook(_tick, chart_asset):
    import time as _t
    s = ST.snap()
    c5 = s["coin5min"]
    asset = chart_asset or "BTC"
    cached_ob = (c5.get("orderbooks") or {}).get(asset) or {}
    open_meta = (c5.get("open_trade_meta") or {}).get(asset) or {}
    partial = (c5.get("partial_legs") or {}).get(asset) or {}
    meta = (c5.get("market_meta") or {}).get(asset) or {}

    # Pull a *fresh* L2 book directly from Polymarket, so the display is not
    # bound to the Coin5min cycle period. Falls back to the cached book on error.
    fresh_yes_bids: List[Dict] = []
    fresh_yes_asks: List[Dict] = []
    fresh_no_bids: List[Dict] = []
    fresh_no_asks: List[Dict] = []
    fetched_ok = False
    fetch_ts = None
    token_yes = meta.get("token_yes")
    token_no = meta.get("token_no")
    try:
        if token_yes:
            book = ST.feeds.get_polymarket_orderbook(token_yes)
            fresh_yes_bids = _normalize_book_side(book.get("bids", []), descending=True)[:8]
            fresh_yes_asks = _normalize_book_side(book.get("asks", []), descending=False)[:8]
            fetched_ok = True
        if token_no:
            book = ST.feeds.get_polymarket_orderbook(token_no)
            fresh_no_bids = _normalize_book_side(book.get("bids", []), descending=True)[:8]
            fresh_no_asks = _normalize_book_side(book.get("asks", []), descending=False)[:8]
            fetched_ok = True
        if fetched_ok:
            fetch_ts = _t.time()
    except Exception:
        fetched_ok = False

    if fetched_ok:
        yes_bid_levels = fresh_yes_bids
        yes_ask_levels = fresh_yes_asks
        no_bid_levels = fresh_no_bids
        no_ask_levels = fresh_no_asks
        ob_source = "live"
        ob_age_sec = 0.0
    else:
        yes_bid_levels = cached_ob.get("yes_bid_levels") or []
        yes_ask_levels = cached_ob.get("yes_ask_levels") or []
        no_bid_levels = cached_ob.get("no_bid_levels") or []
        no_ask_levels = cached_ob.get("no_ask_levels") or []
        ob_source = "cached"
        snap_ts = cached_ob.get("snap_ts")
        ob_age_sec = (_t.time() - float(snap_ts)) if snap_ts else None

    ob = cached_ob  # for sum/depth display (still consistent enough)

    if not yes_bid_levels and not no_bid_levels and not yes_ask_levels and not no_ask_levels:
        return html.Div([
            html.Div(f"Aucun carnet disponible pour {asset}.", style={"fontFamily": M, "fontSize": "12px", "color": T["dim"]}),
            html.Div(f"Status: {meta.get('status','—')} | {meta.get('message','')}", style={"fontFamily": M, "fontSize": "11px", "color": T["dim"], "marginTop": "4px"}),
        ])

    # Build bot order markers
    yes_bot: List[Dict] = []
    no_bot: List[Dict] = []
    if open_meta.get("mode") == "simultaneous":
        # extract entry prices from signal-side meta if available; fall back to current exec
        yes_p = open_meta.get("yes_entry_price") or ob.get("yes_exec_buy") or ob.get("yes_best_ask")
        no_p = open_meta.get("no_entry_price") or ob.get("no_exec_buy") or ob.get("no_best_ask")
        if yes_p is not None:
            yes_bot.append({"price": yes_p, "size_usd": open_meta.get("yes_size_usd", 0.0), "label": "YES LIVE"})
        if no_p is not None:
            no_bot.append({"price": no_p, "size_usd": open_meta.get("no_size_usd", 0.0), "label": "NO LIVE"})
    if partial:
        first_side = partial.get("first_side")
        first_price = partial.get("first_price")
        hedge_side = partial.get("hedge_side")
        hedge_trigger = partial.get("hedge_trigger")
        if first_side == "yes" and first_price is not None:
            yes_bot.append({"price": first_price, "size_usd": partial.get("first_size_usd", 0.0), "label": "LEG1 YES"})
        elif first_side == "no" and first_price is not None:
            no_bot.append({"price": first_price, "size_usd": partial.get("first_size_usd", 0.0), "label": "LEG1 NO"})
        if hedge_side == "yes" and hedge_trigger is not None:
            yes_bot.append({"price": hedge_trigger, "size_usd": 0.0, "label": f"HEDGE@≤{hedge_trigger:.3f}"})
        elif hedge_side == "no" and hedge_trigger is not None:
            no_bot.append({"price": hedge_trigger, "size_usd": 0.0, "label": f"HEDGE@≤{hedge_trigger:.3f}"})

    yes_ladder = _orderbook_ladder("YES", yes_ask_levels, yes_bid_levels, yes_bot)
    no_ladder = _orderbook_ladder("NO", no_ask_levels, no_bid_levels, no_bot)

    # Use FRESH best bid/ask to compute live exec_sum (independent of the cached cycle).
    live_exec_sum = None
    if yes_ask_levels and no_ask_levels:
        live_exec_sum = float(yes_ask_levels[0]["price"]) + float(no_ask_levels[0]["price"])
    age_label = f"{ob_age_sec:.1f}s ago" if (ob_source == "cached" and ob_age_sec is not None) else "now"
    age_color = T["grn"] if ob_source == "live" else (T["amb"] if (ob_age_sec or 0) < 3 else T["red"])

    header = html.Div([
        html.Span(f"Asset: ", style={"fontFamily": M, "fontSize": "11px", "color": T["dim"]}),
        html.Span(f"{asset}", style={"fontFamily": M, "fontSize": "12px", "fontWeight": "700", "color": T["cyan"]}),
        html.Span(f"   exec_sum(live) = {live_exec_sum:.4f}" if live_exec_sum is not None else "   exec_sum(live) = —", style={"fontFamily": M, "fontSize": "11px", "color": T["txt"], "marginLeft": "12px"}),
        html.Span(f"   edge_live = {(1.0-live_exec_sum)*100:+.2f}%" if live_exec_sum is not None else "", style={"fontFamily": M, "fontSize": "11px", "color": T["cyan"] if (live_exec_sum and live_exec_sum < 1.0) else T["red"], "marginLeft": "8px"}),
        html.Span(f"   mark_sum(cached) = {ob.get('mark_sum_yes_no','—')}", style={"fontFamily": M, "fontSize": "11px", "color": T["dim"], "marginLeft": "12px"}),
        html.Span(f"   book = ", style={"fontFamily": M, "fontSize": "11px", "color": T["dim"], "marginLeft": "12px"}),
        html.Span(f"{ob_source.upper()} ({age_label})", style={"fontFamily": M, "fontSize": "11px", "color": age_color, "fontWeight": "700"}),
    ], style={"marginBottom": "8px"})

    legend = html.Div(
        "◀ marqueurs = ordres du bot sur ce niveau (LIVE = ordre actif, LEG1 = single-leg, HEDGE = niveau cible)",
        style={"fontFamily": M, "fontSize": "10px", "color": T["dim"], "marginTop": "6px", "fontStyle": "italic"},
    )

    return html.Div([
        header,
        html.Div([yes_ladder, no_ladder], style={"display": "flex", "gap": "14px", "alignItems": "stretch"}),
        legend,
    ])


@app.callback(Output("alerts-block", "children"), Input("tick", "n_intervals"))
def refresh_alerts(_tick):
    s = ST.snap()
    rows = []
    for a in s["alerts"][:60]:
        color = T["red"] if a["level"] == "CRIT" else T["amb"] if a["level"] == "WARN" else T["grn"]
        rows.append(
            html.Div(
                [
                    html.Span(a["time"], style={"width": "70px", "fontFamily": M, "fontSize": "11px", "color": T["dim"]}),
                    badge(a["level"], color),
                    badge(a["strat"], T["dim"]),
                    html.Span(a["msg"], style={"fontFamily": M, "fontSize": "11px", "color": T["txt"], "marginLeft": "8px"}),
                ],
                style={"display": "flex", "gap": "8px", "alignItems": "center", "padding": "7px 0", "borderBottom": f"1px solid {T['brd']}"},
            )
        )
    if not rows:
        rows = [html.Div("Aucune alerte.", style={"fontFamily": M, "fontSize": "11px", "color": T["dim"]})]
    return rows


if __name__ == "__main__":
    import os
    start_workers()
    port = int(os.environ.get("CYBORG_DASH_PORT", "8077"))
    print(f"CYBORG ready on http://localhost:{port}")
    app.run(debug=False, host="0.0.0.0", port=port)
