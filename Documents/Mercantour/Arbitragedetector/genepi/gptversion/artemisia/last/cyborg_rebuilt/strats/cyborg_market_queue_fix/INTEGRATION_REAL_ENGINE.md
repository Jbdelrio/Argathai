# Integration notes — real execution

## Files
- `execution_engine.py`
- `.env.polymarket.example`

## Environment variables
- `POLYMARKET_PRIVATE_KEY`
- `POLYMARKET_CHAIN_ID`
- `POLYMARKET_SIGNATURE_TYPE`
- `POLYMARKET_FUNDER`

## Minimum integration in `data_manager.py`

### 1. Build the engine
```python
from execution_engine import build_engine_from_env

ST.execution = build_engine_from_env(mode="paper")   # later: "live"
ST.execution.connect()
```

### 2. Replace paper post
```python
resp = ST.execution.place_limit_order(
    symbol=asset,
    token_id=token_id,
    side="BUY",
    price=target_price,
    size=size_usd,
    tags={"strategy": "coin5min", "leg": "leg1"},
)
```

### 3. Track refresh in runtime loop
```python
for order in ST.execution.list_orders():
    if order["state"] in ("OPEN", "PARTIAL", "POSTING"):
        ST.execution.refresh_order(order["client_order_id"])
```

### 4. Surface alerts in GUI
Call:
```python
alerts = ST.execution.get_alerts()
```

Each alert contains:
- `level`
- `code`
- `message`
- `order_id`
- `symbol`
- `sound_hint`

## GUI alerts

For visible alerts:
- banner red/orange/green
- flashing tag for `PARTIAL`, `REJECTED`, `ERROR`

For sound:
- map `sound_hint` to a short browser audio file
- examples:
  - `fill`
  - `partial`
  - `error`

## Safety rollout
1. `paper`
2. `paper + artificial latency`
3. `live` with 1$
4. scale gradually
