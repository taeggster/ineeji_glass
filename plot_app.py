"""Interactive two-line viewer for data/20250710_0000-20260729_0000.csv.

One line is always TMS_nox, the other is picked from a dropdown. The whole
time frame is shown at once; panning/zooming refetches the visible window so
what you see is always drawn from the raw 30-second samples.

    python plot_app.py                 # then open http://127.0.0.1:8000
    python plot_app.py --port 8080 --csv data/other.csv

Startup streams the CSV into memory as float32 (~500 MB, takes a minute or so).
Only pandas/pyarrow/numpy are needed -- the server is stdlib http.server.
"""

from __future__ import annotations

import argparse
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv

ROOT = Path(__file__).parent
DEFAULT_CSV = ROOT / "data" / "20250710_0000-20260729_0000.csv"
NOX = "TMS_nox"
TIME_COL = "Date"
NON_NUMERIC = {"burner_cleaning"}  # bool-ish text columns, not plottable

# Filled by load().
TS: np.ndarray = np.empty(0, dtype=np.int64)  # epoch ms, ascending
COLS: dict[str, np.ndarray] = {}


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load(csv_path: Path) -> None:
    """Stream the CSV in, keeping Date as epoch ms and every numeric column as float32."""
    global TS

    with open(csv_path, "r", encoding="utf-8") as fh:
        header = next(fh).rstrip("\n").split(",")

    # Pin every column's type so blocks can't disagree and empty columns still parse.
    types = {}
    for name in header:
        if name == TIME_COL:
            types[name] = pa.timestamp("ms")
        elif name in NON_NUMERIC:
            types[name] = pa.string()
        else:
            types[name] = pa.float64()

    reader = pacsv.open_csv(
        csv_path,
        read_options=pacsv.ReadOptions(block_size=64 << 20),
        convert_options=pacsv.ConvertOptions(column_types=types, strings_can_be_null=True),
    )

    chunks: dict[str, list[np.ndarray]] = {n: [] for n in header}
    rows = 0
    for batch in reader:
        for name in header:
            arr = batch.column(name)
            if name == TIME_COL:
                chunks[name].append(arr.cast(pa.int64()).to_numpy(zero_copy_only=False))
            elif name in NON_NUMERIC:
                continue
            else:
                chunks[name].append(
                    arr.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
                )
        rows += batch.num_rows
        print(f"\r  {rows:,} rows", end="", flush=True)
    print()

    TS = np.concatenate(chunks[TIME_COL])
    order = np.argsort(TS, kind="stable")
    resorted = not np.array_equal(order, np.arange(TS.size))
    if resorted:
        TS = TS[order]

    for name in header:
        if name == TIME_COL or name in NON_NUMERIC or not chunks[name]:
            continue
        col = np.concatenate(chunks[name])
        if resorted:
            col = col[order]
        if np.isnan(col).all():  # column is entirely empty -- nothing to plot
            continue
        COLS[name] = col
        chunks[name].clear()

    if NOX not in COLS:
        raise SystemExit(f"{NOX!r} not found in {csv_path.name}")
    print(f"  {TS.size:,} rows, {len(COLS)} plottable columns")


# --------------------------------------------------------------------------- #
# decimation
# --------------------------------------------------------------------------- #
def decimate(t: np.ndarray, v: np.ndarray, target: int) -> tuple[np.ndarray, np.ndarray]:
    """Min/max decimate to about `target` points, preserving every spike.

    Each bucket contributes its minimum and its maximum, emitted in time order,
    so the drawn envelope is pixel-identical to plotting all raw samples.
    """
    n = v.size
    if n <= target:
        return t, v

    nb = max(1, target // 2)
    size = n // nb
    m = nb * size
    T = t[:m].reshape(nb, size)
    V = v[:m].reshape(nb, size)

    valid = ~np.isnan(V)
    empty = ~valid.any(axis=1)
    rows = np.arange(nb)

    imin = np.where(valid, V, np.inf).argmin(axis=1)
    imax = np.where(valid, V, -np.inf).argmax(axis=1)
    ta, va = T[rows, imin], V[rows, imin]
    tb, vb = T[rows, imax], V[rows, imax]

    swap = tb < ta
    t1 = np.where(swap, tb, ta)
    v1 = np.where(swap, vb, va)
    t2 = np.where(swap, ta, tb)
    v2 = np.where(swap, va, vb)

    # All-NaN buckets become a real gap in the line.
    t1 = np.where(empty, T[:, 0], t1)
    t2 = np.where(empty, T[:, -1], t2)
    v1 = np.where(empty, np.nan, v1)
    v2 = np.where(empty, np.nan, v2)

    out_t = np.empty(nb * 2, dtype=t.dtype)
    out_v = np.empty(nb * 2, dtype=np.float64)
    out_t[0::2], out_t[1::2] = t1, t2
    out_v[0::2], out_v[1::2] = v1, v2

    if m < n:  # leftover tail, at most nb-1 samples
        out_t = np.concatenate([out_t, t[m:]])
        out_v = np.concatenate([out_v, v[m:].astype(np.float64)])
    return out_t, out_v


def jsonable(t: np.ndarray, v: np.ndarray) -> dict:
    return {
        "x": t.tolist(),
        "y": [None if np.isnan(y) else round(float(y), 6) for y in v],
    }


def series(cols: list[str], start: float | None, end: float | None, width: int) -> dict:
    lo = 0 if start is None else int(np.searchsorted(TS, int(start), "left"))
    hi = TS.size if end is None else int(np.searchsorted(TS, int(end), "right"))
    lo = max(0, min(lo - 1, TS.size))  # one sample of overhang so lines reach the edge
    hi = max(lo, min(hi + 1, TS.size))

    # A one-day window is only ~2,880 samples, so the floor here keeps every
    # request at full resolution; decimate() is just a backstop.
    target = max(6_000, min(width * 2, 20_000))
    t = TS[lo:hi]
    out = {"n_raw": int(hi - lo), "target": target, "series": []}
    for name in [NOX, *cols]:
        dt, dv = decimate(t, COLS[name][lo:hi], target)
        out["series"].append({"name": name, **jsonable(dt, dv)})
    out["decimated"] = out["n_raw"] > target
    return out


# --------------------------------------------------------------------------- #
# server
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):  # keep the console quiet
        pass

    def _send(self, body: bytes, ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        q = parse_qs(url.query)

        if url.path in ("/", "/index.html"):
            self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return

        if url.path == "/api/meta":
            body = json.dumps(
                {
                    "columns": [c for c in COLS if c != NOX],
                    "nox": NOX,
                    "t0": int(TS[0]),
                    "t1": int(TS[-1]),
                    "rows": int(TS.size),
                }
            ).encode()
            self._send(body, "application/json")
            return

        if url.path == "/api/series":
            cols = [c for c in q.get("col", []) if c]
            unknown = [c for c in cols if c not in COLS]
            if unknown:
                self.send_error(400, f"unknown column(s): {unknown}")
                return
            start = q.get("start", [None])[0]
            end = q.get("end", [None])[0]
            width = int(float(q.get("width", ["1400"])[0]))
            payload = series(cols, float(start) if start else None,
                             float(end) if end else None, width)
            self._send(json.dumps(payload).encode(), "application/json")
            return

        self.send_error(404)


PAGE = r"""
<title>NOx viewer</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root { color-scheme: light dark; }
  body { margin: 0; font: 14px/1.4 system-ui, sans-serif; }
  header { display: flex; gap: 12px; align-items: center; padding: 10px 14px;
           border-bottom: 1px solid color-mix(in srgb, currentColor 18%, transparent); }
  select { font: inherit; padding: 4px 6px; max-width: 260px; }
  label { display: flex; gap: 5px; align-items: center; white-space: nowrap; }
  .swatch { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
  #status { margin-left: auto; opacity: .65; font-variant-numeric: tabular-nums; }
  #chart { width: 100%; height: calc(100vh - 52px); }
</style>

<header>
  <label><span class="swatch" style="background:#e2703a"></span><select id="col2"></select></label>
  <label><span class="swatch" style="background:#3fa66a"></span><select id="col3"></select></label>
  <label>Window <select id="span">
    <option value="900000">15 min</option>
    <option value="3600000">1 hour</option>
    <option value="10800000">3 hours</option>
    <option value="21600000">6 hours</option>
    <option value="86400000" selected>1 day</option>
  </select></label>
  <button id="first">⏮ start</button>
  <button id="last">end ⏭</button>
  <span id="status">loading…</span>
</header>
<div id="chart"></div>

<script>
const chart = document.getElementById('chart');
const pickers = [document.getElementById('col2'), document.getElementById('col3')];
const spanPick = document.getElementById('span');
const COLORS = ['#3f7ee8', '#e2703a', '#3fa66a'];  // nox, series 2, series 3
const status = document.getElementById('status');
const MIN_SPAN = 30 * 1000 * 20;   // never zoom in past ~20 samples wide
const MAX_SPAN = 24 * 3600 * 1000; // one day is as far out as it goes
let meta, drawing = false, timer = null, suppress = false, view = [null, null];

const fmt = n => n.toLocaleString();
const iso = ms => new Date(ms).toISOString().slice(0, 19).replace('T', ' ');

async function fetchSeries(start, end) {
  const p = new URLSearchParams({ width: Math.round(chart.clientWidth) });
  pickers.forEach(s => p.append('col', s.value));
  if (start != null) p.set('start', Math.round(start));
  if (end != null) p.set('end', Math.round(end));
  const r = await fetch('/api/series?' + p);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

function traces(d) {
  return d.series.map((s, i) => ({
    x: s.x.map(iso), y: s.y, name: s.name,
    type: 'scattergl', mode: 'lines', connectgaps: false,
    line: { width: 1.2, color: COLORS[i] },
    yaxis: i === 0 ? 'y' : 'y' + (i + 1),
    hovertemplate: '%{y}<extra>' + s.name + '</extra>',
  }));
}

const axis = (i, extra) => Object.assign({
  title: { text: i === 0 ? meta.nox : pickers[i - 1].value, standoff: 4 },
  color: COLORS[i], zeroline: false, showgrid: i === 0,
}, extra);

const layout = () => ({
  margin: { l: 62, r: 8, t: 10, b: 40 },
  dragmode: 'pan',
  hovermode: 'x unified',
  showlegend: true,
  legend: { orientation: 'h', y: 1.06, x: 0 },
  xaxis: { type: 'date', domain: [0, 0.9], range: [iso(view[0]), iso(view[1])] },
  yaxis: axis(0, { side: 'left' }),
  yaxis2: axis(1, { side: 'right', overlaying: 'y', anchor: 'x' }),
  yaxis3: axis(2, { side: 'right', overlaying: 'y', anchor: 'free', position: 0.965 }),
  uirevision: 'keep',
});

const CONFIG = { scrollZoom: false, responsive: true, displaylogo: false };

async function draw(relayoutOnly) {
  if (drawing) { clearTimeout(timer); timer = setTimeout(() => draw(true), 120); return; }
  drawing = true;
  status.textContent = 'loading…';
  try {
    const d = await fetchSeries(view[0], view[1]);
    const t = traces(d);
    suppress = true;
    if (relayoutOnly) {
      await Plotly.react(chart, t, layout(), CONFIG);
    } else {
      await Plotly.newPlot(chart, t, layout(), CONFIG);
      chart.on('plotly_relayout', onRelayout);
    }
    suppress = false;
    status.textContent = iso(view[0]) + ' → ' + iso(view[1]) + ' · ' + fmt(d.n_raw) + ' samples'
      + (d.decimated ? ' · min/max reduced to ' + fmt(d.target) + ' px-columns' : ' · full resolution');
  } catch (e) {
    suppress = false;
    status.textContent = 'error: ' + e.message;
  } finally {
    drawing = false;
  }
}

/** Clamp a candidate window to the data range, keeping its span where possible. */
function setView(lo, hi) {
  let span = Math.min(Math.max(hi - lo, MIN_SPAN), MAX_SPAN, meta.t1 - meta.t0);
  if (lo < meta.t0) lo = meta.t0;
  if (lo + span > meta.t1) lo = meta.t1 - span;
  view = [Math.round(lo), Math.round(lo + span)];
}

function schedule(instant) {
  suppress = true;
  Plotly.relayout(chart, { 'xaxis.range': [iso(view[0]), iso(view[1])] }).then(() => { suppress = false; });
  clearTimeout(timer);
  timer = setTimeout(() => draw(true), instant ? 0 : 200);
}

// Wheel / two-finger scroll pans through time. Ctrl (or ⌘) + wheel zooms.
chart.addEventListener('wheel', ev => {
  if (!meta) return;
  ev.preventDefault();
  const span = view[1] - view[0];
  if (ev.ctrlKey || ev.metaKey) {
    const box = chart.getBoundingClientRect();
    const frac = Math.min(Math.max((ev.clientX - box.left - 60) / (box.width - 120), 0), 1);
    const anchor = view[0] + frac * span;
    const next = span * (ev.deltaY > 0 ? 1.25 : 0.8);
    setView(anchor - frac * next, anchor + (1 - frac) * next);
  } else {
    const delta = Math.abs(ev.deltaX) > Math.abs(ev.deltaY) ? ev.deltaX : ev.deltaY;
    const step = span * 0.2 * Math.sign(delta);
    setView(view[0] + step, view[1] + step);
  }
  schedule(false);
}, { passive: false });

// Arrow keys page left/right.
window.addEventListener('keydown', ev => {
  if (!meta || ev.target.tagName === 'SELECT') return;
  const span = view[1] - view[0];
  const step = ev.key === 'ArrowLeft' ? -span * 0.5 : ev.key === 'ArrowRight' ? span * 0.5 : 0;
  if (!step) return;
  ev.preventDefault();
  setView(view[0] + step, view[1] + step);
  schedule(false);
});

function onRelayout(ev) {  // drag-pan and box-zoom from Plotly itself
  if (suppress) return;
  if (ev['xaxis.range[0]'] == null) return;
  const parse = s => (typeof s === 'number' ? s : Date.parse(s.replace(' ', 'T') + 'Z'));
  setView(parse(ev['xaxis.range[0]']), parse(ev['xaxis.range[1]']));
  clearTimeout(timer);
  timer = setTimeout(() => draw(true), 200);
}

function applySpan() {
  const want = Number(spanPick.value);
  const mid = (view[0] + view[1]) / 2;
  setView(mid - want / 2, mid + want / 2);
  schedule(true);
}

(async function init() {
  meta = await (await fetch('/api/meta')).json();
  const options = meta.columns.map(c => `<option>${c}</option>`).join('');
  const defaults = ['TMS_sox', 'OIL MAIN', 'ARCH #1', 'MELTER BT #11'].filter(c => meta.columns.includes(c));
  pickers.forEach((s, i) => {
    s.innerHTML = options;
    if (defaults[i]) s.value = defaults[i];
    s.onchange = () => draw(true);
  });

  view = [meta.t0, Math.min(meta.t0 + Number(spanPick.value), meta.t1)];
  spanPick.onchange = applySpan;
  document.getElementById('first').onclick = () => {
    setView(meta.t0, meta.t0 + (view[1] - view[0])); schedule(true);
  };
  document.getElementById('last').onclick = () => {
    setView(meta.t1 - (view[1] - view[0]), meta.t1); schedule(true);
  };
  await draw(false);
})();
</script>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"missing: {args.csv}")

    print(f"loading {args.csv}")
    load(args.csv)

    url = f"http://127.0.0.1:{args.port}"
    print(f"serving {url}  (ctrl-c to stop)")
    if not args.no_open:
        webbrowser.open(url)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
