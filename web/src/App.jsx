import React, { useEffect, useState } from "react";

const EXAMPLES = ["Which faults happened on the pump on 2020-02-08?", "What changed during the fault in run valve2/1?",
  "What was the average flow of the pump on 9 March 2020 between 10:00 and 11:00?",
  "Any drift alarm on the industrial machine?", "What was the humidity of the pump?"];

function Spark({ data, color = "#1f5fa8" }) {
  if (!data || !data.v.length) return null;
  const W = 560, H = 120, v = data.v, lo = Math.min(...v), hi = Math.max(...v);
  const pts = v.map((y, i) => `${(i / (v.length - 1)) * W},${H - ((y - lo) / (hi - lo || 1)) * (H - 10) - 5}`).join(" ");
  return <svg viewBox={`0 0 ${W} ${H}`} className="spark"><polyline points={pts} fill="none" stroke={color} strokeWidth="1.6" /></svg>;
}

export default function App() {
  const [q, setQ] = useState(EXAMPLES[0]);
  const [hist, setHist] = useState([]);
  const [ov, setOv] = useState(null);
  const [trend, setTrend] = useState(null);
  useEffect(() => {
    fetch("/api/overview").then((r) => r.json()).then(setOv);
    fetch("/api/trend").then((r) => r.json()).then(setTrend);
  }, []);
  async function ask(text) {
    const r = await (await fetch("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" },
                                               body: JSON.stringify({ question: text }) })).json();
    setHist((h) => [r, ...h]);
  }
  return (
    <div className="page">
      <header><div className="brand">Maintenance copilot · real historian</div>
        <div className="sub">Pump testbed (SKAB, 8 sensors at 1 Hz) and an industrial machine (NAB). Every figure comes from a tool call, shown under the answer.</div></header>
      {ov && <div className="tiles">{ov.assets.map((o) => (
        <div key={o.asset} className="tile"><span>{o.asset}</span><b>{o.events}</b>
          <small>labelled {o.asset.includes("Pump") ? "faults" : "anomaly windows"} · {o.start.slice(0, 10)} to {o.end.slice(0, 10)}</small></div>))}
        <div className={"tile " + (ov.drifts.length ? "warn" : "")}><span>Machine drift alarms</span><b>{ov.drifts.length}</b>
          <small>{ov.drifts.map((d) => d.slice(0, 16)).join(" · ") || "none"}</small></div></div>}
      <div className="grid">
        <main>
          <form className="ask" onSubmit={(e) => { e.preventDefault(); ask(q); }}>
            <input value={q} onChange={(e) => setQ(e.target.value)} /><button>Ask</button></form>
          <div className="examples">{EXAMPLES.map((e) => <button key={e} onClick={() => { setQ(e); ask(e); }}>{e}</button>)}</div>
          {hist.map((r, i) => (
            <div key={i} className={"msg " + (r.refused ? "refused" : "")}>
              <div className="qq">{r.question}</div>
              <div className="aa">{r.answer}</div>
              {r.calls.length > 0 && <details><summary>{r.calls.length} tool call{r.calls.length > 1 ? "s" : ""} · grounded {r.grounded ? "✓" : "✗"}</summary>
                <pre>{JSON.stringify(r.calls, null, 1)}</pre></details>}
            </div>))}
        </main>
        <aside><div className="card"><div className="lbl">Industrial machine temperature, hourly mean</div>
          <Spark data={trend} /><small>Dec 2013 to Feb 2014. Reference for drift alarms: the first 15% of the record.</small></div></aside>
      </div>
    </div>);
}
