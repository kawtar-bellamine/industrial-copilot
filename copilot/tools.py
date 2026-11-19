"""Tools the copilot can call. Each returns plain data and the period it covers, so every figure in an answer can be
traced back to the historian. Questions outside the history, unknown assets or tags raise NoData."""
from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .historian import ASSETS, DB as _DB

DB = _DB


def set_db(path: Path):
    global DB
    DB = Path(path); _series.cache_clear(); _table.cache_clear()


class NoData(Exception):
    """Raised when the question asks for something the historian does not contain."""


@lru_cache(maxsize=32)
def _series(asset: str, tag: str) -> pd.Series:
    con = sqlite3.connect(DB)
    d = pd.read_sql("select ts, value from history where path = ? order by ts", con, params=(f"{asset}/{tag}",),
                    parse_dates=["ts"])
    con.close()
    return d.set_index("ts")["value"]


@lru_cache(maxsize=4)
def _table(name: str) -> pd.DataFrame:
    con = sqlite3.connect(DB)
    d = pd.read_sql(f"select * from {name}", con)
    con.close()
    for c in ("start", "end"):
        if c in d:
            d[c] = pd.to_datetime(d[c])
    return d


def data_range(asset: str):
    s = _series(asset, next(iter(ASSETS[asset])))
    return s.index.min(), s.index.max() + pd.Timedelta(seconds=1)


def _check(asset, tag, start, end):
    if asset not in ASSETS:
        raise NoData(f"no asset called '{asset}' (assets: {', '.join(ASSETS)})")
    if tag is not None and tag not in ASSETS[asset]:
        raise NoData(f"the {asset} has no '{tag}' tag (it has: {', '.join(ASSETS[asset])})")
    lo, hi = data_range(asset)
    if start >= hi or end <= lo:
        raise NoData(f"no {asset} data between {start:%Y-%m-%d %H:%M} and {end:%Y-%m-%d %H:%M} (history covers "
                     f"{lo:%Y-%m-%d %H:%M} to {hi:%Y-%m-%d %H:%M})")
    x = _series(asset, tag or next(iter(ASSETS[asset])))
    if not ((x.index >= start) & (x.index < end)).any():
        raise NoData(f"no {asset} data between {start:%Y-%m-%d %H:%M} and {end:%Y-%m-%d %H:%M}: the asset was "
                     "not recording then")


def tag_stats(asset: str, tag: str, start, end) -> dict:
    _check(asset, tag, start, end)
    x = _series(asset, tag); x = x[(x.index >= start) & (x.index < end)]
    unit = ASSETS[asset][tag]
    return dict(tool="tag_stats", asset=asset, tag=tag, unit=unit, start=str(start), end=str(end), samples=len(x),
                first=str(x.index.min()), last=str(x.index.max()), mean=round(float(x.mean()), 3),
                min=round(float(x.min()), 3), max=round(float(x.max()), 3), std=round(float(x.std()), 3))


def events(asset: str, start, end) -> dict:
    """Labelled faults / anomalies overlapping the period (the event log)."""
    if asset not in ASSETS:
        raise NoData(f"no asset called '{asset}'")
    e = _table("events")
    e = e[(e.asset == asset) & (e.start < end) & (e.end >= start)].sort_values("start")
    items = [dict(kind=r.kind, start=str(r.start), end=str(r.end),
                  minutes=round((r.end - r.start).total_seconds() / 60, 1), run=r.run) for r in e.itertuples()]
    by_kind = pd.Series([i["kind"] for i in items], dtype=object).value_counts().to_dict()
    return dict(tool="events", asset=asset, start=str(start), end=str(end), count=len(items), events=items,
                by_kind={k: int(v) for k, v in by_kind.items()},
                source=e.source.iloc[0] if len(e) else "")


def fault_effect(asset: str, start, end, before_s: int = 120) -> dict:
    """For the first labelled fault in the period: mean of every tag during the fault against the before_s seconds
    before it, sorted by the size of the change in standard deviations of the 'before' period."""
    ev = events(asset, start, end)
    if not ev["events"]:
        raise NoData(f"no labelled fault on the {asset} in that period")
    f = ev["events"][0]
    a, b = pd.Timestamp(f["start"]), pd.Timestamp(f["end"])
    rows = []
    for tag, unit in ASSETS[asset].items():
        x = _series(asset, tag)
        pre = x[(x.index >= a - pd.Timedelta(seconds=before_s)) & (x.index < a)]
        dur = x[(x.index >= a) & (x.index <= b)]
        if len(pre) < 10 or len(dur) < 10:
            continue
        z = (dur.mean() - pre.mean()) / (pre.std() + 1e-9)
        rows.append(dict(tag=tag, unit=unit, before=round(float(pre.mean()), 3), during=round(float(dur.mean()), 3),
                         change_sd=round(float(z), 1)))
    if not rows:
        raise NoData("not enough data around that fault")
    rows.sort(key=lambda r: -abs(r["change_sd"]))
    return dict(tool="fault_effect", asset=asset, fault=f, before_seconds=before_s, changes=rows)


# ---------- drift detection (Plant/Machine, hourly means) ----------
def detect(y: pd.Series, ref_share: float = 0.15, method: str = "ewma", lam: float = 0.2, L: float = 10.0,
           k: float = 0.5, h: float = 20.0, thr: float = 5.0):
    """Alarms on hourly means against a reference period, the first 15% of the series (NAB's probationary
    period). Returns the alarm mask (after the reference period) and the standardised series."""
    n0 = max(24, int(len(y) * ref_share))
    base = y.iloc[:n0]
    z = (y - base.mean()) / (base.std() + 1e-9)
    if method == "threshold":
        a = np.abs(z.values) > thr
    elif method == "ewma":
        e = z.ewm(alpha=lam, adjust=False).mean()
        a = np.abs(e.values) > L * np.sqrt(lam / (2 - lam))
    else:
        sp = sn = 0.0; a = np.zeros(len(z), bool)
        for i, v in enumerate(z.values):
            sp = max(0, sp + v - k); sn = max(0, sn - v - k)
            if sp > h or sn > h:
                a[i] = True; sp = sn = 0.0                   # restart after an alarm
    a[:n0] = False
    return a, z


def hourly(asset: str, tag: str) -> pd.Series:
    return _series(asset, tag).resample("1h").mean().dropna()


def drifts(asset: str, start, end, settings: dict | None = None) -> dict:
    if asset != "Plant/Machine":
        raise NoData("drift detection runs on the Plant/Machine hourly data; for the pump, ask about its faults")
    _check(asset, "temperature", start, end)
    s = settings or dict(method="ewma", L=10.0)        # chosen on the other NAB series (see the evaluation)
    y = hourly(asset, "temperature")
    a, z = detect(y, **s)
    al = y.index[a]
    starts = [t for i, t in enumerate(al) if i == 0 or (t - al[i - 1]) > pd.Timedelta(hours=1)]
    starts = [t for t in starts if start <= t < end]
    n0 = max(24, int(len(y) * 0.15))
    return dict(tool="drifts", asset=asset, tag="temperature", start=str(start), end=str(end),
                reference=f"{y.index[0]:%Y-%m-%d} to {y.index[n0 - 1]:%Y-%m-%d}",
                reference_mean=round(float(y.iloc[:n0].mean()), 2), count=len(starts), alarms=[str(t) for t in starts],
                detector=s)
