"""Build the historian (SQLite) from two real, labelled data sets, fetched at pinned commits.

- Testbed/Pump: SKAB, the Skoltech Anomaly Benchmark (github.com/waico/SKAB, GPL-3.0). A water loop with a
  centrifugal pump, 8 sensors at 1 Hz, 35 runs recorded on 8 Feb, 1 Mar and 9 Mar 2020, with labelled faults
  (valves closed, rotor imbalance, leaks, water level, cavitation). The labels become the event log.
- Plant/Machine: temperature of an internal component of a large industrial machine, every 5 minutes from
  2 Dec 2013 to 19 Feb 2014, with 4 labelled anomaly windows, one a planned shutdown and one a catastrophic failure
  (Numenta Anomaly Benchmark, realKnownCause, github.com/numenta/NAB, AGPL-3.0).

Tag paths follow an ISA-95 style: Site/Asset/tag. A few SKAB runs overlap in time by up to 13 minutes (the testbed's
clock): the historian keeps the run that started first for the overlapping seconds and reports how many were dropped.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
DB = ROOT / "data/historian.sqlite"
SOURCES = {"SKAB": ("https://github.com/waico/SKAB.git", "b2c0d46c2971dcbfe71e26087b6d231998bb91c2", ["data"]),
           "NAB": ("https://github.com/numenta/NAB.git", "ea702d75cc2258d9d7dd35ca8e5e2539d71f3140",
                   ["data/realKnownCause", "labels/combined_windows.json"])}
PUMP_TAGS = {"Accelerometer1RMS": ("vibration_1", "g"), "Accelerometer2RMS": ("vibration_2", "g"),
             "Current": ("current", "A"), "Pressure": ("pressure", "bar"), "Temperature": ("motor_temperature", "°C"),
             "Thermocouple": ("fluid_temperature", "°C"), "Voltage": ("voltage", "V"),
             "Volume Flow RateRMS": ("flow", "L/min")}
FAULTS = {"valve1": "inlet valve closed", "valve2": "outlet valve closed"}
OTHER = {1: "fluid leak or addition", 2: "fluid leak or addition", 3: "fluid leak or addition",
         4: "fluid leak or addition", 5: "rotor imbalance", 6: "rotor imbalance", 7: "rotor imbalance",
         8: "rotor imbalance", 9: "rotor imbalance", 10: "water level change", 11: "water level change",
         12: "water level change", 13: "cavitation", 14: "cavitation"}
NAB_EVENTS = ["planned shutdown", "anomaly before the failure", "catastrophic failure", "labelled anomaly"]
ASSETS = {"Testbed/Pump": {t: u for t, u in PUMP_TAGS.values()}, "Plant/Machine": {"temperature": "as recorded"}}


def fetch(name: str) -> Path:
    url, commit, paths = SOURCES[name]
    dest = RAW / name
    if not (dest / paths[0]).exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--no-checkout", url, str(dest)], check=True)
        subprocess.run(["git", "-C", str(dest), "checkout", "-q", commit, "--", *paths], check=True)
    return dest


def skab_runs():
    root = fetch("SKAB") / "data"
    runs = []
    for f in sorted(root.glob("*/*.csv")):
        group = f.parent.name
        d = pd.read_csv(f, sep=";", parse_dates=["datetime"])
        kind = None if group == "anomaly-free" else FAULTS.get(group) or OTHER[int(f.stem)]
        runs.append(dict(run=f"{group}/{f.stem}", kind=kind, df=d))
    return sorted(runs, key=lambda r: r["df"].datetime.iloc[0])


def nab_machine():
    root = fetch("NAB")
    d = pd.read_csv(root / "data/realKnownCause/machine_temperature_system_failure.csv", parse_dates=["timestamp"])
    w = json.loads((root / "labels/combined_windows.json").read_text())[
        "realKnownCause/machine_temperature_system_failure.csv"]
    return d, [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in w]


def fault_segments(d: pd.DataFrame):
    """(start, end) timestamps of each run of anomaly = 1."""
    a = d["anomaly"].values.astype(int)
    seg, start = [], None
    for i, v in enumerate(a):
        if v and start is None:
            start = i
        if not v and start is not None:
            seg.append((d.datetime.iloc[start], d.datetime.iloc[i - 1])); start = None
    if start is not None:
        seg.append((d.datetime.iloc[start], d.datetime.iloc[len(a) - 1]))
    return seg


def build(db: Path = DB) -> dict:
    db.parent.mkdir(parents=True, exist_ok=True)
    if db.exists():
        db.unlink()
    hist, events, runs_tab = [], [], []
    seen_until = pd.Timestamp.min
    dropped = 0
    for r in skab_runs():
        d = r["df"]
        runs_tab.append(dict(asset="Testbed/Pump", run=r["run"], start=str(d.datetime.iloc[0]),
                             end=str(d.datetime.iloc[-1]), kind=r["kind"] or "normal operation"))
        if r["kind"]:
            for a, b in fault_segments(d):
                events.append(dict(asset="Testbed/Pump", start=str(a), end=str(b), kind=r["kind"], run=r["run"],
                                   source="SKAB labels"))
        keep = d.datetime > seen_until
        dropped += int((~keep).sum())
        seen_until = max(seen_until, d.datetime.iloc[-1])
        d = d[keep]
        for col, (tag, _) in PUMP_TAGS.items():
            hist.append(pd.DataFrame(dict(path=f"Testbed/Pump/{tag}", ts=d.datetime.astype(str), value=d[col])))
    m, wins = nab_machine()
    hist.append(pd.DataFrame(dict(path="Plant/Machine/temperature", ts=m.timestamp.astype(str), value=m.value)))
    for (a, b), k in zip(wins, NAB_EVENTS):
        events.append(dict(asset="Plant/Machine", start=str(a), end=str(b), kind=k, run="", source="NAB labels"))
    con = sqlite3.connect(db)
    pd.concat(hist).to_sql("history", con, index=False)
    con.execute("create index ix on history(path, ts)")
    pd.DataFrame(events).to_sql("events", con, index=False)
    pd.DataFrame(runs_tab).to_sql("runs", con, index=False)
    pd.DataFrame([dict(asset=a, tag=t, unit=u) for a, tags in ASSETS.items() for t, u in tags.items()]).to_sql(
        "tags", con, index=False)
    con.commit(); con.close()
    return dict(rows=int(sum(len(h) for h in hist)), events=len(events), runs=len(runs_tab),
                overlapping_seconds_dropped=dropped)


if __name__ == "__main__":
    print(build())
