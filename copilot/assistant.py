"""The copilot: understand the question, call the tools, answer with the figures they returned, cite them, and refuse
when the historian does not hold the answer.

Understanding is rule-based (assets and their aliases, tags, periods, intents), so every step can be tested. An LLM
can replace the planner and the writer (OpenAI-compatible endpoint through LLM_BASE_URL / LLM_API_KEY / LLM_MODEL);
the grounding check below stays in place either way: an answer is released only if every number it contains comes
from a tool output.
"""
from __future__ import annotations

import json
import os
import re

import pandas as pd

from . import tools
from .historian import ASSETS

ASSET_ALIASES = [(r"\b(pump|testbed|test bench|water loop|circulation loop)\b", "Testbed/Pump"),
                 (r"\b(industrial machine|the machine|machine's|machine temperature)\b", "Plant/Machine")]
TAG_ALIASES = [(r"\bvibrations?\b|\baccelerometer", ["vibration_1", "vibration_2"]),
               (r"\b(motor|engine) (body )?temperature", ["motor_temperature"]),
               (r"\b(fluid|water|loop) temperature|thermocouple", ["fluid_temperature"]),
               (r"\bcurrent|amperage|amps?\b", ["current"]), (r"\bpressure\b", ["pressure"]),
               (r"\bvoltage\b", ["voltage"]), (r"\bflow( rate)?\b", ["flow"]),
               (r"\btemperatures?\b", ["temperature"])]
UNKNOWN_TAGS = r"\b(humidity|rpm|speed|torque|oil|power factor|noise level)\b"
MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov",
                                      "dec"], start=1)}
MON = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"


def asset_of(q: str):
    ql = q.lower()
    for pat, a in ASSET_ALIASES:
        if re.search(pat, ql):
            return a
    m = re.search(r"\b(?:machine|station|asset|compressor|conveyor|robot|press)\s*([A-Z]{0,3}-?\d+)?", q, re.I)
    if m and re.search(r"compressor|conveyor|robot|press|station|asset|machine\s*[A-Z]{0,3}-?\d", m.group(0), re.I):
        raise tools.NoData(f"there is no '{m.group(0).strip()}' in the historian (assets: {', '.join(ASSETS)})")
    return None


def tags_of(q: str, asset: str | None):
    ql = q.lower()
    if re.search(UNKNOWN_TAGS, ql):
        raise tools.NoData(f"'{re.search(UNKNOWN_TAGS, ql).group(0)}' is not recorded (tags: "
                           + "; ".join(f"{a}: {', '.join(t)}" for a, t in ASSETS.items()) + ")")
    for pat, tags in TAG_ALIASES:
        if re.search(pat, ql):
            if tags == ["temperature"] and asset == "Testbed/Pump":
                return ["motor_temperature", "fluid_temperature"]
            return tags
    return []


def _year_for(asset, month):
    if asset == "Plant/Machine":
        return 2013 if month == 12 else 2014
    return 2020


def period(q: str, asset: str):
    """(start, end, label). Dates: 2020-03-09, 9 March 2020, March 9; times: 'between 10:00 and 11:00'; runs."""
    ql = q.lower()
    lo, hi = tools.data_range(asset)
    m = re.search(r"\b(valve1|valve2|other|anomaly-free)/(\d+)\b", ql)
    if m:
        r = tools._table("runs")
        r = r[r.run == f"{m.group(1)}/{m.group(2)}"]
        if r.empty:
            raise tools.NoData(f"no run {m.group(0)}")
        return r.start.iloc[0], r.end.iloc[0] + pd.Timedelta(seconds=1), f"during run {m.group(0)}"
    day = None
    if (m := re.search(r"(\d{4})-(\d{2})-(\d{2})", ql)):
        day = pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    elif (m := re.search(r"\b(\d{1,2})(?:st|nd|rd|th)? " + MON + r"(?:,? (\d{4}))?", ql)):
        mo = MONTHS[m.group(2)[:3]]
        day = pd.Timestamp(int(m.group(3)) if m.group(3) else _year_for(asset, mo), mo, int(m.group(1)))
    elif (m := re.search(r"\b" + MON + r" (\d{4})\b", ql)):
        a = pd.Timestamp(int(m.group(2)), MONTHS[m.group(1)[:3]], 1)
        return a, a + pd.offsets.MonthBegin(1), f"in {a:%B %Y}"
    elif (m := re.search(r"\b" + MON + r" (\d{1,2})(?!\d)(?:st|nd|rd|th)?(?:,? (\d{4}))?", ql)):
        mo = MONTHS[m.group(1)[:3]]
        day = pd.Timestamp(int(m.group(3)) if m.group(3) else _year_for(asset, mo), mo, int(m.group(2)))
    t = re.search(r"(?:between|from) (\d{1,2})[:h](\d{2}) (?:and|to) (\d{1,2})[:h](\d{2})", ql)
    if day is not None and t:
        a = day + pd.Timedelta(hours=int(t.group(1)), minutes=int(t.group(2)))
        b = day + pd.Timedelta(hours=int(t.group(3)), minutes=int(t.group(4)))
        return a, b, f"on {a:%d %b %Y} from {a:%H:%M} to {b:%H:%M}"
    if day is not None:
        return day, day + pd.Timedelta(days=1), f"on {day:%d %b %Y}"
    if re.search(r"\b(19|20)\d{2}\b", ql) and not re.search(r"\b(2013|2014|2020)\b", ql):
        y = re.search(r"\b(19|20)\d{2}\b", ql).group(0)
        raise tools.NoData(f"no {asset} data in {y} (history covers {lo:%Y-%m-%d} to {hi:%Y-%m-%d})")
    return lo, hi, "over the whole history"


def intent(q: str) -> str:
    ql = q.lower()
    if re.search(r"\b(tomorrow|next (week|month|run|day)|will (it|the)|forecast|predict)\b", ql):
        return "forecast"
    if re.search(r"what changed|which sensors?|effect of|signature|how did .* (change|react)|what did .* look like", ql):
        return "effect"
    if re.search(r"drift|abnormal trend|early warning|alarm", ql):
        return "drifts"
    if re.search(r"fault|fail|incident|what happened|anomal|event|shutdown|breakdown|problem|how many .*(stops|faults)",
                 ql):
        return "events"
    if re.search(r"average|mean|max|min|highest|lowest|value|level|reading|how (high|low|hot)|what was the", ql):
        return "stats"
    return "unknown"


def _fmt(v, unit):
    return f"{v} {unit}" if unit != "as recorded" else f"{v}"


def answer(q: str) -> dict:
    calls = []
    try:
        it = intent(q)
        if it == "forecast":
            raise tools.NoData("I report on recorded data only; I do not forecast")
        asset = asset_of(q)
        if asset is None and re.search(r"\b(valve1|valve2|other|anomaly-free)/\d+\b", q.lower()):
            asset = "Testbed/Pump"                          # runs are recordings of the pump testbed
        tags = tags_of(q, asset)
        if asset is None:
            if any(t in ASSETS["Testbed/Pump"] for t in tags) and tags != ["temperature"]:
                asset = "Testbed/Pump"
            elif it == "unknown":
                raise tools.NoData("I can answer on the pump testbed and the industrial machine: sensor values, "
                                   "labelled faults, what changed during a fault, and drift alarms")
            else:
                raise tools.NoData("which asset? the pump testbed or the industrial machine")
        if it == "unknown":
            raise tools.NoData("I can answer on sensor values, labelled faults, what changed during a fault, and "
                               "drift alarms")
        start, end, plabel = period(q, asset)
        if it == "stats":
            if not tags:
                raise tools.NoData(f"which sensor? the {asset} has: {', '.join(ASSETS[asset])}")
            if tags == ["temperature"] and asset == "Testbed/Pump":
                tags = ["motor_temperature", "fluid_temperature"]
            r = [tools.tag_stats(asset, t, start, end) for t in tags]; calls += r
            txt = " ".join(f"{plabel[0].upper() + plabel[1:]}, the {asset} {x['tag'].replace('_', ' ')} averaged {_fmt(x['mean'], x['unit'])} "
                           f"(min {x['min']}, max {x['max']}, {x['samples']} samples)." for x in r)
        elif it == "events":
            r = tools.events(asset, start, end); calls.append(r)
            if not r["events"]:
                txt = f"No labelled fault on the {asset} {plabel}."
            else:
                kinds = r["by_kind"]
                noun = "anomaly window" if asset == "Plant/Machine" else "fault"
                lst = "; ".join(f"{e['kind']} from {e['start'][:19]} to "
                                f"{e['end'][11:19] if e['end'][:10] == e['start'][:10] else e['end'][:19]} "
                                f"({e['minutes']} min)" for e in r["events"][:6])
                more = ", list truncated" if r["count"] > 6 else ""
                txt = (f"{r['count']} labelled {noun}{'s' if r['count'] > 1 else ''} on the {asset} {plabel} "
                       f"({', '.join(f'{v} {k}' for k, v in kinds.items())}): {lst}{more}. Source: {r['source']}.")
        elif it == "effect":
            r = tools.fault_effect(asset, start, end); calls.append(r)
            top = r["changes"][:3]
            txt = (f"During the {r['fault']['kind']} from {r['fault']['start'][:19]} ({r['fault']['minutes']} min), "
                   "the largest changes against the 120 seconds before were: " + "; ".join(
                       f"{c['tag'].replace('_', ' ')} {c['before']} to {c['during']} {c['unit'] if c['unit'] != 'as recorded' else ''}"
                       .rstrip() + f" ({c['change_sd']:+} standard deviations)" for c in top) + ".")
        else:
            r = tools.drifts(asset, start, end); calls.append(r)
            if r["alarms"]:
                txt = (f"{len(r['alarms'])} drift alarm{'s' if len(r['alarms']) > 1 else ''} on the {asset} temperature "
                       f"{plabel}, against the reference period {r['reference']} (mean {r['reference_mean']}): "
                       + ", ".join(a[:16] for a in r["alarms"][:8]) + ".")
            else:
                txt = f"No drift alarm on the {asset} temperature {plabel}."
    except tools.NoData as e:
        return dict(question=q, answer=f"I cannot answer from the historian: {e}.", refused=True, calls=calls,
                    grounded=True)
    ok, bad = grounded(txt, calls)
    if not ok:
        return dict(question=q, answer="Answer withheld: it contains figures not found in the data "
                                       f"({', '.join(bad)}).", refused=True, calls=calls, grounded=False)
    return dict(question=q, answer=txt, refused=False, calls=calls, grounded=True, intent=it)


NUM = re.compile(r"-?\d+(?:\.\d+)?")


def grounded(text: str, calls: list) -> tuple[bool, list]:
    """Every number of the answer must appear in a tool output (dates, times and run names excepted)."""
    blob = json.dumps(calls)
    allowed = set(NUM.findall(blob)) | {"120", "6"}
    clean = re.sub(r"\d{4}-\d{2}-\d{2}( \d{2}:\d{2}(:\d{2})?)?", "", text)
    clean = re.sub(r"\b\d{2}:\d{2}(:\d{2})?\b", "", clean)
    clean = re.sub(r"\b\d{1,2} (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{4}\b", "", clean)
    clean = re.sub(r"\b(valve1|valve2|other|anomaly-free)/\d+\b|vibration_\d", "", clean)
    nums = NUM.findall(clean)
    bad = [n for n in nums if n.lstrip("-+") not in allowed and n not in allowed
           and n.rstrip("0").rstrip(".") not in allowed]
    return not bad, bad


def llm_rewrite(result: dict) -> dict:
    """Optional: let an LLM reword the answer from the tool outputs; the grounding check decides if it is released."""
    base, key, model = os.getenv("LLM_BASE_URL"), os.getenv("LLM_API_KEY"), os.getenv("LLM_MODEL")
    if not (base and key and model) or result.get("refused"):
        return result
    import urllib.request
    body = {"model": model, "temperature": 0, "messages": [
        {"role": "system", "content": "Answer the engineer in two sentences using only the figures in the tool "
                                      "outputs. Do not compute new figures."},
        {"role": "user", "content": f"Question: {result['question']}\nTool outputs: {json.dumps(result['calls'])}"}]}
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            txt = json.loads(r.read())["choices"][0]["message"]["content"]
    except Exception:  # noqa: BLE001
        return result
    ok, _ = grounded(txt, result["calls"])
    return dict(result, answer=txt) if ok else result
