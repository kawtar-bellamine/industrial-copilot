import pandas as pd

from copilot import tools
from copilot.assistant import answer, grounded, intent, period
from copilot.historian import nab_machine


def test_historian_holds_both_real_assets():
    lo, hi = tools.data_range("Testbed/Pump")
    assert lo == pd.Timestamp("2020-02-08 13:30:47") and hi.date() == pd.Timestamp("2020-03-09").date()
    ev = tools.events("Plant/Machine", *tools.data_range("Plant/Machine"))
    assert ev["count"] == 4 and "catastrophic failure" in ev["by_kind"]
    assert tools.events("Testbed/Pump", *tools.data_range("Testbed/Pump"))["count"] == 34


def test_tag_stats_match_the_raw_file():
    raw, _ = nab_machine()
    day = raw[raw.timestamp.dt.date == pd.Timestamp("2014-01-10").date()].value
    r = tools.tag_stats("Plant/Machine", "temperature", pd.Timestamp("2014-01-10"), pd.Timestamp("2014-01-11"))
    assert r["mean"] == round(float(day.mean()), 3) and r["samples"] == len(day)


def test_intents_and_periods():
    assert intent("Which faults happened on the pump?") == "events"
    assert intent("What changed during the fault in run valve1/2?") == "effect"
    assert intent("Any drift alarm on the machine?") == "drifts"
    s, e, _ = period("flow on 9 March 2020 between 10:00 and 11:30", "Testbed/Pump")
    assert s == pd.Timestamp("2020-03-09 10:00") and e == pd.Timestamp("2020-03-09 11:30")
    s, e, _ = period("alarms in January 2014", "Plant/Machine")
    assert s == pd.Timestamp("2014-01-01") and e == pd.Timestamp("2014-02-01")


def test_refusals():
    for q in ["Average vibration of compressor C2?", "Humidity of the pump?", "Pump flow on 2021-05-01?",
              "What will the machine temperature be tomorrow?", "Tell me a joke", "Pump pressure on 15 Dec 2013?"]:
        assert answer(q)["refused"], q


def test_answer_and_grounding():
    r = answer("What was the average flow of the pump on 9 March 2020 between 10:00 and 11:00?")
    assert not r["refused"] and r["grounded"] and "L/min" in r["answer"]
    bad = r["answer"].replace(str(r["calls"][0]["mean"]), "33.33")
    assert not grounded(bad, r["calls"])[0]


def test_fault_signature_of_a_closed_outlet_valve_is_the_flow():
    r = answer("What changed during the fault in run valve2/1?")
    assert r["calls"][0]["changes"][0]["tag"] == "flow"
