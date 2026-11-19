"""HTTP API for the copilot interface."""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from copilot import tools
from copilot.assistant import answer, llm_rewrite

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="Industrial copilot")


class Q(BaseModel):
    question: str


@app.post("/api/ask")
def ask(q: Q):
    return llm_rewrite(answer(q.question))


@app.get("/api/overview")
def overview():
    out = []
    for asset in ("Testbed/Pump", "Plant/Machine"):
        lo, hi = tools.data_range(asset)
        ev = tools.events(asset, lo, hi)
        out.append(dict(asset=asset, start=str(lo)[:16], end=str(hi)[:16], events=ev["count"], by_kind=ev["by_kind"]))
    d = tools.drifts("Plant/Machine", *tools.data_range("Plant/Machine"))
    return dict(assets=out, drifts=d["alarms"])


@app.get("/api/trend")
def trend():
    y = tools.hourly("Plant/Machine", "temperature")
    return dict(t=[str(t) for t in y.index], v=[round(float(v), 2) for v in y.values])


dist = ROOT / "web" / "dist"
if dist.exists():
    app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/")
    def index():
        return FileResponse(dist / "index.html")
