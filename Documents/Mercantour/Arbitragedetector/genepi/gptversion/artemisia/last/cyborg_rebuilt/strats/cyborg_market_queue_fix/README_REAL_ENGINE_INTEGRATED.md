# REAL ENGINE — Integrated switch

## What is included
- `execution_engine.py`
- `data_manager_real_live_switch_full.py`
- `cyborg_dash_real_live_switch_full.py`

## What this version does
- adds an execution mode switch: `paper / live`
- adds a `CONNECT` action for the execution engine
- shows execution alerts in the GUI
- mirrors strategy order intents to the execution engine
- refreshes live order states periodically

## Important limitation
This integration is a **phased rollout build**:
- strategy logic / pnl accounting is still driven by the existing paper engine
- in `live` mode, real orders can be routed through `execution_engine.py`
- use **tiny sizes first**: 1$ to 3$

## Why this is still useful
It lets you test:
- venue connectivity
- order posting
- pending / partial / filled / cancelled transitions
- visible alerts for real execution problems

## Required setup
1. install `py-clob-client`
2. fill `.env.polymarket.example`
3. set mode to `live`
4. press `CONNECT`
5. only then start with very small size

## Alerts
Visible in GUI:
- `ORDER_POSTED`
- `ORDER_PARTIAL`
- `ORDER_FILLED`
- `ORDER_CANCELLED`
- `ORDER_REJECTED`
- `ORDER_REFRESH_ERROR`

`sound_hint` is also surfaced so you can later attach browser audio.
