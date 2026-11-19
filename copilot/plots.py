"""Figures for the README."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import tools  # noqa: E402
from .historian import NAB_EVENTS, nab_machine  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/figures"
BLUE, ORANGE, GREY, RED = "#1f5fa8", "#e69138", "#9aa7b5", "#c0392b"


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def machine():
    raw, wins = nab_machine()
    y = tools.hourly("Plant/Machine", "temperature")
    a, _ = tools.detect(y, method="ewma", L=10.0)
    n0 = max(24, int(len(y) * 0.15))
    fig, ax = plt.subplots(figsize=(13, 3.9), dpi=130)
    ax.plot(raw.timestamp, raw.value, color=GREY, lw=0.5, label="temperature, every 5 min")
    ax.plot(y.index, y.values, color=BLUE, lw=1.1, label="hourly mean")
    ax.axvspan(y.index[0], y.index[n0 - 1], color="#eef3f9", zorder=0, label="reference period (first 15%)")
    for i, ((s, e), k) in enumerate(zip(wins, NAB_EVENTS)):
        ax.axvspan(s, e, color=RED, alpha=0.15, lw=0)
        ax.text(s, 114 if i % 2 == 0 else 106, k, fontsize=8, color=RED)
    al = y.index[a]
    starts = [t for i, t in enumerate(al) if i == 0 or (t - al[i - 1]).total_seconds() > 3600]
    for t in starts:
        ax.axvline(t, color=ORANGE, lw=1.6)
    ax.plot([], [], color=ORANGE, lw=1.6, label="EWMA alarm (limit chosen on the other series)")
    ax.set_ylim(0, 120); ax.set_ylabel("temperature (as recorded)")
    ax.legend(fontsize=8, frameon=False, loc="lower left", ncol=2); clean(ax)
    ax.set_title("Real industrial machine (NAB): labelled anomaly windows in red", fontsize=10.5)
    fig.tight_layout(); fig.savefig(OUT / "fig1_machine.png"); plt.close(fig)


def detectors():
    r = json.loads((ROOT / "results/metrics.json").read_text())["drift_detectors"]
    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=130)
    for m, c in (("threshold", GREY), ("ewma", BLUE), ("cusum", ORANGE)):
        v = r[m]
        ax.scatter(v["false_alarms_per_30_days"], v["found"] / v["windows"], s=90, color=c, zorder=3)
        ax.annotate(f"{m.upper() if m != 'threshold' else 'fixed threshold'}: {v['found']}/{v['windows']} windows",
                    (v["false_alarms_per_30_days"], v["found"] / v["windows"]), textcoords="offset points",
                    xytext=(8, -3), fontsize=8.5)
    ax.set_xlabel("false alarms per 30 days"); ax.set_ylabel("labelled windows detected")
    ax.set_xlim(0, 2.6); ax.set_ylim(0, 0.6); clean(ax)
    ax.set_title("7 real NAB series (realKnownCause), settings chosen leaving one series out", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "fig2_detectors.png"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    machine(); detectors()


if __name__ == "__main__":
    main()
