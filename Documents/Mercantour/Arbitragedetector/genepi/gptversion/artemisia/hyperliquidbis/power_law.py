"""Artemisia Glacialis v2 — BTC Power Law (Santostasi)"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Optional
from config import Config, BTC_GENESIS, PL_SLOPE, PL_INTERCEPT, PL_BAND


class PowerLaw:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def days(self, dt=None):
        dt = dt or datetime.now(timezone.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max((dt - BTC_GENESIS).total_seconds() / 86400, 1)

    def corridor(self, dt=None):
        d = self.days(dt)
        lm = PL_SLOPE * np.log10(d) + PL_INTERCEPT
        return {"lower": 10**(lm - PL_BAND), "median": 10**lm,
                "upper": 10**(lm + PL_BAND)}

    def deviation(self, price):
        c = self.corridor()
        return np.log10(max(price, 1)) - (PL_SLOPE * np.log10(self.days()) + PL_INTERCEPT)

    def bias(self, price) -> Dict:
        dev = self.deviation(price)
        c = self.corridor()
        cfg = self.cfg
        if dev < cfg.pl_long_zone:
            b, exp = "strong_long", 1.4
        elif dev < 0:
            b, exp = "long", 1.0
        elif dev < cfg.pl_neutral_zone:
            b, exp = "neutral", 0.7
        else:
            b, exp = "reduce", 0.4
        return {"bias": b, "deviation": round(dev, 4), "exposure": exp,
                "corridor": {k: round(v, 0) for k, v in c.items()}}

    def corridor_history(self, start_year=2023):
        dates = pd.date_range(datetime(start_year, 1, 1, tzinfo=timezone.utc),
                              datetime.now(timezone.utc), freq="D")
        rows = [{"date": d, **self.corridor(d)} for d in dates]
        return pd.DataFrame(rows)
