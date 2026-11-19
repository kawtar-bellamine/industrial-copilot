"""Evaluation on the real historian:
(1) questions answered, checked against figures recomputed independently from the raw CSV files;
(2) questions it must refuse; (3) the grounding check against corrupted answers; (4) questions written freely;
(5) the drift detectors on the 7 real series of NAB's realKnownCause set, with their labelled windows, settings
chosen by leaving one series out."""
from __future__ import annotations

import itertools
import json
import random
import re

import numpy as np
import pandas as pd

from . import tools
from .assistant import answer, grounded
from .historian import NAB_EVENTS, PUMP_TAGS, ROOT, fault_segments, fetch, nab_machine, skab_runs

TAG_WORDS = {"vibration_1": None, "vibration_2": None, "current": "current", "pressure": "pressure",
             "motor_temperature": "motor temperature", "fluid_temperature": "fluid temperature",
             "voltage": "voltage", "flow": "flow rate"}


def raw_pump():
    """The pump record rebuilt from the raw files, independently of the historian code: runs in time order, seconds
    already covered by an earlier run left out."""
    parts, until = [], pd.Timestamp.min
    for r in skab_runs():
        d = r["df"]
        parts.append(d[d.datetime > until]); until = max(until, d.datetime.max())
    return pd.concat(parts).set_index("datetime")


def questions(rng: random.Random):
    pump = raw_pump()
    runs = skab_runs()
    mach, wins = nab_machine()
    mach = mach.set_index("timestamp")["value"]
    qs = []
    days = sorted({t.normalize() for t in pump.index})
    for _ in range(24):                                     # pump sensor value over a time window
        day = rng.choice(days)
        sub = pump[pump.index.normalize() == day]
        h0 = rng.choice(sorted({t.hour for t in sub.index}))
        a = day + pd.Timedelta(hours=h0, minutes=rng.choice([0, 15, 30]))
        b = a + pd.Timedelta(minutes=rng.choice([15, 30, 60]))
        if not ((sub.index >= a) & (sub.index < b)).any():
            continue
        tag = rng.choice([t for t, w in TAG_WORDS.items() if w])
        col = next(c for c, (t, _) in PUMP_TAGS.items() if t == tag)
        x = pump.loc[(pump.index >= a) & (pump.index < b), col]
        dl = rng.choice([f"{day:%Y-%m-%d}", f"{day.day} {day:%B} {day.year}", f"{day:%B} {day.day}, {day.year}"])
        q = rng.choice([f"What was the average {TAG_WORDS[tag]} of the pump on {dl} between {a:%H:%M} and {b:%H:%M}?",
                        f"Mean {TAG_WORDS[tag]} on the testbed on {dl} from {a:%H:%M} to {b:%H:%M}?"])
        qs.append(dict(q=q, kind="pump value", check=("number", round(float(x.mean()), 3))))
    for _ in range(8):                                      # machine temperature on a day
        day = rng.choice(sorted({t.normalize() for t in mach.index})[1:-1])
        x = mach[mach.index.normalize() == day]
        dl = rng.choice([f"{day:%Y-%m-%d}", f"{day.day} {day:%b} {day.year}"])
        q = rng.choice([f"Average temperature of the industrial machine on {dl}?",
                        f"What was the machine temperature on {dl}? Give the max."])
        key = "max" if "max" in q else "mean"
        qs.append(dict(q=q, kind="machine value", check=("number", round(float(getattr(x, key)()), 3))))
    for day in days:                                        # faults on a day
        n = sum(1 for r in runs if r["kind"] for a, b in fault_segments(r["df"])
                if a.normalize() <= day and b >= day)
        qs.append(dict(q=f"How many faults were labelled on the pump on {day:%Y-%m-%d}?", kind="fault count",
                       check=("number", n)))
    for r in rng.sample([r for r in runs if r["kind"]], 8):  # what changed during a fault
        d = r["df"].set_index("datetime")
        a, b = fault_segments(r["df"])[0]
        pre = d[(d.index >= a - pd.Timedelta(seconds=120)) & (d.index < a)]
        dur = d[(d.index >= a) & (d.index <= b)]
        z = {PUMP_TAGS[c][0]: abs((dur[c].mean() - pre[c].mean()) / (pre[c].std() + 1e-9)) for c in PUMP_TAGS}
        qs.append(dict(q=f"What changed during the fault in run {r['run']}?", kind="fault signature",
                       check=("text", max(z, key=z.get).replace("_", " "))))
    qs.append(dict(q="What happened to the industrial machine?", kind="machine events",
                   check=("text", "catastrophic failure from 2014-01-27")))
    qs.append(dict(q="Was there a planned shutdown of the industrial machine?", kind="machine events",
                   check=("text", "planned shutdown from 2013-12-10")))
    refuse = ["What was the humidity of the pump on 9 March 2020?", "Average vibration of compressor C2 yesterday?",
              "What was the pump flow on 2021-05-01?", "Average current of the pump on 2019-03-09?",
              "What will the machine temperature be tomorrow?", "Forecast the pump pressure for next week",
              "What was the speed of the pump on 9 March 2020?", "Torque of the industrial machine on 2014-01-10?",
              "Average pressure of machine M7 on 2020-03-09?", "Write me a poem about pumps",
              "What was the pump pressure on 15 Dec 2013?", "Who was the operator on the testbed?"]
    qs += [dict(q=q, kind="must refuse", check=("refuse",)) for q in refuse]
    return qs


def check(r, c):
    if c[0] == "refuse":
        return r["refused"]
    if r["refused"]:
        return False
    if c[0] == "text":
        return c[1].lower() in r["answer"].lower()
    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", r["answer"])]
    return any(abs(v - c[1]) < 1e-6 for v in nums)


def corrupt(results, rng):
    """Change one number of each correct answer, as a language model might, and check the answer is withheld."""
    caught = n = 0
    for r in results:
        if r["refused"]:
            continue
        txt = re.sub(r"\d{4}-\d{2}-\d{2}( \d{2}:\d{2}(:\d{2})?)?", lambda m: m.group(0).replace("0", "\0"), r["answer"])
        nums = [m for m in re.finditer(r"(?<![\w/])\d+\.\d+", txt)]
        if not nums:
            continue
        m = rng.choice(nums)
        v = float(m.group(0)); new = f"{v * rng.choice([1.07, 0.93, 1.15]) + 0.01:.2f}"
        bad = (txt[:m.start()] + new + txt[m.end():]).replace("\0", "0")
        n += 1; caught += not grounded(bad, r["calls"])[0]
    return dict(corrupted=n, withheld=caught)


FREEFORM = [  # written freely after the rule set was frozen; the expected behaviour noted by hand
    ("How did the pump behave when the outlet valve was closed?", "needs a fault-type filter, not a date"),
    ("Was the motor hotter on 9 March than on 1 March?", "needs two periods and a comparison"),
    ("Show me the flow during run valve1/3", "answerable: flow over the run"),
    ("When did the machine fail?", "answerable: the event log"),
    ("Is the pump vibrating more than usual today?", "'today' is outside the history: must refuse"),
    ("What is the max pressure ever recorded on the pump?", "answerable: whole history"),
    ("Did the drift alarm see the machine failure coming?", "needs alarms and events together"),
    ("What's the typical current draw of the pump motor?", "answerable: whole-history mean"),
    ("List the cavitation tests", "answerable: events filtered by type"),
    ("Which run had the strongest vibration?", "needs a ranking over runs"),
]


# ---------- drift detectors on NAB realKnownCause ----------
def nab_series():
    root = fetch("NAB")
    w = json.loads((root / "labels/combined_windows.json").read_text())
    out = {}
    for f in sorted((root / "data/realKnownCause").glob("*.csv")):
        d = pd.read_csv(f, parse_dates=["timestamp"]).set_index("timestamp")["value"]
        y = d.resample("1h").mean().dropna()
        out[f.stem] = (y, [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in w[f"realKnownCause/{f.name}"]])
    return out


SETTINGS = ([dict(method="threshold", thr=t) for t in (3, 4, 5, 6, 8, 10)]
            + [dict(method="ewma", L=L) for L in (3, 4, 6, 8, 10, 14)]
            + [dict(method="cusum", h=h) for h in (5, 10, 20, 40, 80, 160)])


def score_series(y, wins, s):
    a, _ = tools.detect(y, **s)
    al = y.index[a]
    starts = [t for i, t in enumerate(al) if i == 0 or (t - al[i - 1]) > pd.Timedelta(hours=1)]
    n0 = max(24, int(len(y) * 0.15))
    t0 = y.index[n0]
    wins = [(x, b) for x, b in wins if b > t0]               # windows inside the reference period cannot count
    found, delays = 0, []
    for x, b in wins:
        hit = [t for t in starts if x <= t <= b]
        if hit:
            found += 1; delays.append((hit[0] - x).total_seconds() / 3600)
    fp = sum(1 for t in starts if not any(x <= t <= b for x, b in wins))
    days = (y.index[-1] - t0).total_seconds() / 86400
    return dict(windows=len(wins), found=found, delays_h=delays, false_alarms=fp, days=days)


def agg(rows):
    w = sum(r["windows"] for r in rows); f = sum(r["found"] for r in rows)
    fp = sum(r["false_alarms"] for r in rows); d = sum(r["days"] for r in rows)
    dl = [x for r in rows for x in r["delays_h"]]
    return dict(windows=w, found=f, false_alarms=fp, false_alarms_per_30_days=round(30 * fp / d, 2),
                median_hours_from_window_start=round(float(np.median(dl)), 1) if dl else None)


def drift_eval():
    S = nab_series()
    names = list(S)
    out = {}
    for method in ("threshold", "ewma", "cusum"):
        cand = [s for s in SETTINGS if s["method"] == method]
        rows, chosen = [], {}
        for test in names:                                   # leave one series out
            def utility(s):
                rr = [score_series(*S[n], s) for n in names if n != test]
                a = agg(rr)
                return a["found"] - 0.5 * a["false_alarms"]  # a missed window costs two false alarms
            best = max(cand, key=utility)
            chosen[test] = best
            rows.append(dict(series=test, **score_series(*S[test], best)))
        out[method] = dict(**agg(rows), settings=chosen, per_series=[
            {k: v for k, v in r.items() if k != "delays_h"} for r in rows])
    return out, {n: dict(hours=len(S[n][0]), windows=len(S[n][1])) for n in names}


def main():
    rng = random.Random(7)
    qs = questions(rng)
    results = [answer(q["q"]) for q in qs]
    ok = [check(r, q["check"]) for r, q in zip(results, qs)]
    by = {}
    for q, o in zip(qs, ok):
        by.setdefault(q["kind"], [0, 0]); by[q["kind"]][0] += o; by[q["kind"]][1] += 1
    fails = [dict(q=q["q"], expected=str(q["check"]), got=r["answer"][:300]) for q, r, o in zip(qs, results, ok) if not o]
    res = dict(questions=len(qs), correct=int(sum(ok)), by_kind={k: f"{a}/{b}" for k, (a, b) in by.items()},
               failures=fails, grounding=corrupt(results, rng))
    ff = [dict(question=q, note=n, answer=answer(q)["answer"]) for q, n in FREEFORM]
    res["drift_detectors"], res["nab_series"] = drift_eval()
    con_hist = tools._table("runs")
    res["historian"] = dict(pump_runs=len(con_hist), events=len(tools._table("events")))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/metrics.json").write_text(json.dumps(res, indent=1, default=str))
    (ROOT / "results/freeform_questions.json").write_text(json.dumps(ff, indent=1, ensure_ascii=False))
    (ROOT / "results/qa_samples.json").write_text(json.dumps(
        [dict(question=r["question"], answer=r["answer"], ok=o) for r, o in zip(results, ok)], indent=1,
        ensure_ascii=False))
    print(json.dumps({k: v for k, v in res.items() if k not in ("nab_series",)}, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
