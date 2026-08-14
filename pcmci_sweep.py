"""Run a sweep of PCMCI+ / LPCMCI configurations over the furnace data and cache them.

    python pcmci_sweep.py              # run everything not already cached
    python pcmci_sweep.py --force      # re-run even if cached
    python pcmci_sweep.py --only 2 5   # run only these SWEEP entries (by name prefix)
    python pcmci_sweep.py --report     # print the NOx tables for every cached run
    OMP_NUM_THREADS=1 python pcmci_sweep.py --jobs 4

    python pcmci_sweep.py --method lpcmci --only 12   # same sweep, latent-aware
    python pcmci_sweep.py --method lpcmci --report

--method picks the algorithm. pcmciplus (default) assumes causal sufficiency:
every common cause is in the frame, so an edge means a cause. lpcmci drops that
assumption and returns a PAG, where the edge mark says what was learnt:
'-->' cause, '<->' unmeasured common cause only, 'o->' one of the two, 'o-o'
adjacent but undetermined. It is a lot slower -- start at 30min/20min.

Results land in runs_simple/ (PCMCI+) or runs_simple_lpcmci/ (LPCMCI) as
<name>_<horizon>_<tag>.npz -- the tag is a hash of the config, so changing a
knob makes a new file instead of silently overwriting the old one.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.lpcmci import LPCMCI
from tigramite.pcmci import PCMCI

# =============================================================================
# CONFIG -- edit freely
# =============================================================================

DATA  = Path("data/20250710_0000-20260729_0000.parquet")

# one directory per method, so the two sets of results never mix
RUNS_BY_METHOD = {"pcmciplus": Path("runs_simple"),
                  "lpcmci":    Path("runs_simple_lpcmci")}
METHOD = "pcmciplus"          # default; override with --method lpcmci

START = "2025-07-10"
END   = "2025-10-12"          # auto-control tests begin after this -> manual only

# ---- the sweep --------------------------------------------------------------
# flows : "pos"  -> per-position, L/R averaged   (oil_1..4 + oxy_1..4 or ratio_1..4)
#         "main" -> OIL MAIN / OXY MAIN only     (oil_main + oxy_main or ratio_main)
#         "burner" -> per-burner, L and R kept apart (16 flow variables)
# ratio : False -> oil + oxy      True -> oil + ratio (oxy dropped)
SWEEP = [
    # name,             flows,  ratio, freq
    # ("1_pos_oxy_10",    "pos",  False, "10min"),
    # ("2_pos_ratio_10",  "pos",  True,  "10min"),
    ("3_main_oxy_10",   "main", False, "10min"),
    # ("4_main_ratio_10", "main", True,  "10min"),
    # ("5_pos_ratio_5",   "pos",  True,  "5min"),
    # ("6_main_ratio_5",  "main", True,  "5min"),
    # ("7_pos_ratio_15",  "pos",  True,  "15min"),
    # ("8_main_ratio_15", "main", True,  "15min"),
    ("9_main_oxy_5",  "main", False,  "5min"),
    ("10_main_oxy_15",  "main", False,  "15min"),
    ("11_main_oxy_20",  "main", False,  "20min"),
     ("12_main_oxy_30",  "main", False,  "30min")
]

HORIZON  = "3.5h"               # tau_max = HORIZON / freq, per run
PC_ALPHA = 0.01
MISSING  = 999.               # sentinel handed to tigramite

# ---- LPCMCI only ------------------------------------------------------------
# LPCMCI allows latent confounders, so it is much slower than PCMCI+ and its
# output is a PAG: edges carry marks ('-->' cause, '<->' latent confounder,
# 'o->' one of the two, 'o-o' undetermined). These caps keep the runtime sane;
# raise them for a more thorough (slower) search.
LPCMCI_KW = dict(
    n_preliminary_iterations=1,   # 0 = fastest, 4 = paper's "thorough" setting
    max_p_global=np.inf,          # e.g. 3 -> only test conditioning sets up to size 3
    max_q_global=np.inf,          # e.g. 20 -> at most 20 tests per size
    max_pds_set=np.inf,           # e.g. 20 -> cap the non-ancestral search set
)

# ---- target -----------------------------------------------------------------
TARGET        = "TMS_nox"
TARGET_CLEAN  = f"{TARGET}_clean"
NOX_MIN_VALID = 10.0          # <= this is an analyser dropout
NOX_PAD       = "5min"       # widen each dropout both ways (eats the ramp)

# ---- variables --------------------------------------------------------------
OIL_SUFFIX = "_cleaned"
BURNERS    = ["1L", "2L", "3L", "4L", "1R", "2R", "3R", "4R"]   # no 0R

OILC  = [f"OIL {b}{OIL_SUFFIX}" for b in BURNERS]
OXYC  = [f"OXY {b}" for b in BURNERS]
MAINC = ["OIL MAIN_cleaned", "OXY MAIN"]

# temperatures kept as their own variable (none -- all are composited below)
TEMPC   = ["ARCH #3", "MELTER BT #11"]

# temperatures averaged into one variable. Members sit 15-40 deg apart, so the
# zero-masking below runs FIRST and a composite is NaN unless every member is
# present -- averaging across a dropout would inject a fake step.
# dropped: ARCH #1 (reads 30-84, not a temperature), ARCH #2 (90% zeros),
#          TEMP #05 Throat
COMPOSITES = {
    # "arch_3_4":    ["ARCH #3", "ARCH #4"],
    # "bt_temp": ["TEMP #08", "TEMP #09", "TEMP #10", "MELTER BT #11"],
}
REQUIRE_ALL    = True        # NaN a composite unless every member is present
TEMP_DROP_ZERO = True         # exact 0 on a thermocouple is a dropout, not a reading

MISC    = ["pull"]
WEATHER = []

COMP_SRC = [c for cols in COMPOSITES.values() for c in cols]

FLOWC   = MAINC                          # >0 filter, then log
NONFLOW = TEMPC + MISC    # carried through as levels

WEATHER_EXOGENOUS = True             # forbid furnace -> weather links

# =============================================================================


def load_raw():
    """Read the parquet at native resolution, clean, and build TARGET_CLEAN."""
    want = (["Date", TARGET, "burner_cleaning"] + FLOWC + TEMPC + COMP_SRC
            + MISC + WEATHER)
    want = list(dict.fromkeys(want))
    have = set(pq.ParquetFile(DATA).schema_arrow.names)
    missing = [c for c in want if c not in have]
    if missing:
        print(f"NOT IN FILE (skipped): {missing}")
    cols = [c for c in want if c in have]

    d = pd.read_parquet(DATA, columns=cols)
    d["Date"] = pd.to_datetime(d["Date"])
    d = d.set_index("Date").sort_index().loc[START:]
    d = d[d.index < END]

    flows = [c for c in FLOWC if c in d.columns]
    d[flows] = d[flows].where(d[flows] > 0)              # impossible -> NaN

    temps = [c for c in TEMPC + COMP_SRC if c in d.columns]
    if TEMP_DROP_ZERO and temps:
        nz = (d[temps] == 0).sum()
        d[temps] = d[temps].where(d[temps] != 0)         # dropout -> NaN
        hit = {c: int(n) for c, n in nz.items() if n}
        if hit:
            print("temp zeros -> NaN: "
                  + ", ".join(f"{c} {n} ({n / len(d):.2%})" for c, n in hit.items()))

    for name, members in COMPOSITES.items():             # average AFTER zero-masking
        cols_m = [c for c in members if c in d.columns]
        if not cols_m:
            print(f"composite {name!r}: no members in file, skipped"); continue
        sub = d[cols_m]
        d[name] = (sub.mean(axis=1).where(sub.notna().all(axis=1))
                   if REQUIRE_ALL else sub.mean(axis=1))
        d = d.drop(columns=cols_m)
        print(f"composite {name!r} = mean({', '.join(cols_m)})"
              f" | NaN {d[name].isna().mean():.2%}")

    wx = [c for c in WEATHER if c in d.columns]
    d[wx] = d[wx].where(d[wx] > -900)                    # -999/-998 sentinels -> NaN

    step    = d.index.to_series().diff().median()
    k_pad   = int(round(pd.Timedelta(NOX_PAD) / step))
    nox_bad = d[TARGET].isna() | (d[TARGET] <= NOX_MIN_VALID)
    nox_bad = (nox_bad.rolling(2 * k_pad + 1, center=True, min_periods=1)
                      .max().astype(bool))
    d[TARGET_CLEAN] = d[TARGET].mask(nox_bad)

    print(f"raw {step} | {len(d)} rows | {START}..{END} "
          f"| NOx dropouts {nox_bad.mean():.2%}")
    return d, flows


def resample(d, flows, freq):
    """Resample to `freq`, blank any bin touched by burner cleaning, log the flows."""
    keep = flows + [c for c in NONFLOW if c in d.columns] + [TARGET_CLEAN]
    R = d[keep].resample(freq).mean()

    if "burner_cleaning" in d.columns:
        bc = (d["burner_cleaning"].astype(bool).resample(freq).max()
                .astype(bool).reindex(R.index, fill_value=False))
        R.loc[bc.to_numpy()] = np.nan

    R[flows] = np.log(R[flows])
    return R


def build_frame(R, flows_mode, use_ratio):
    """Assemble the analysis frame and difference it. Returns a DataFrame."""
    C = pd.DataFrame(index=R.index)

    if flows_mode == "main":
        units = [("main", ["OIL MAIN_cleaned"], ["OXY MAIN"])]
    elif flows_mode == "pos":
        units = [(str(i), [f"OIL {i}L{OIL_SUFFIX}", f"OIL {i}R{OIL_SUFFIX}"],
                          [f"OXY {i}L", f"OXY {i}R"]) for i in (1, 2, 3, 4)]
    elif flows_mode == "burner":
        units = [(b, [f"OIL {b}{OIL_SUFFIX}"], [f"OXY {b}"]) for b in BURNERS]
    else:
        raise ValueError(f"unknown flows mode: {flows_mode}")

    for name, oil_cols, oxy_cols in units:
        oil_cols = [c for c in oil_cols if c in R.columns]
        oxy_cols = [c for c in oxy_cols if c in R.columns]
        if not oil_cols or not oxy_cols:
            raise KeyError(f"missing flow columns for unit {name!r}")
        oil = R[oil_cols].mean(axis=1)
        oxy = R[oxy_cols].mean(axis=1)
        C[f"oil_{name}"] = oil
        if use_ratio:
            C[f"ratio_{name}"] = oxy - oil      # logs -> subtraction IS the ratio
        else:
            C[f"oxy_{name}"] = oxy

    for c in NONFLOW:
        if c in R.columns:
            C[c] = R[c]
    C[TARGET_CLEAN] = R[TARGET_CLEAN]

    return C.diff()


def link_assumptions_for(var_names, tau_max, method="pcmciplus"):
    """Allow everything except furnace -> weather.

    The two algorithms read the marks differently. For PCMCI+ '-?>' means "if
    this lagged link exists it points forward in time" -- a free statement. For
    LPCMCI '-?>' would *assert* ancestorship, so the no-claim mark there is
    'o?>' ("j is not an ancestor of i"), which LPCMCI imposes on lagged links
    anyway. Contemporaneously 'o?o' is the no-claim mark for both.
    """
    lagged = "o?>" if method == "lpcmci" else "-?>"
    n = len(var_names)
    wx = {k for k, nm in enumerate(var_names) if nm.startswith("WEATHER_")}
    la = {}
    for j in range(n):
        la[j] = {}
        for i in range(n):
            for tau in range(tau_max + 1):
                if i == j and tau == 0:
                    continue
                if WEATHER_EXOGENOUS and j in wx and i not in wx:
                    # lagged furnace -> weather: omitted == forbidden. At lag 0
                    # LPCMCI wants the pair mentioned symmetrically, so keep the
                    # edge but forbid the arrowhead into weather.
                    if tau == 0 and method == "lpcmci":
                        la[j][(i, 0)] = "<?o"     # i (furnace) is not an ancestor of j (wx)
                    continue
                if (WEATHER_EXOGENOUS and method == "lpcmci"
                        and tau == 0 and i in wx and j not in wx):
                    la[j][(i, 0)] = "o?>"         # same claim, other direction
                    continue
                la[j][(i, -tau)] = lagged if tau > 0 else "o?o"
    return la


def diagnostics(C):
    """Rows kept, effective rank and worst collinear pair, on the analysis frame."""
    x = C.replace([np.inf, -np.inf], np.nan).dropna()
    x = x.loc[:, x.std() > 1e-9]
    if len(x) < 100 or x.shape[1] < 2:
        return dict(rows=len(x), rank=None, max_corr=None, worst=None)
    z = (x - x.mean()) / x.std()
    ev = np.linalg.svd(z.values / np.sqrt(len(z)), compute_uv=False) ** 2
    ev /= ev.sum()
    cc = z.corr().abs()
    np.fill_diagonal(cc.values, 0)
    a, b = np.unravel_index(cc.values.argmax(), cc.shape)
    return dict(rows=len(x),
                rank=int((ev.cumsum() < 0.95).sum() + 1),
                max_corr=float(cc.values.max()),
                worst=f"{cc.index[a]} ~ {cc.columns[b]}")


def run_one(d, flows, name, flows_mode, use_ratio, freq, force=False,
            method=METHOD):
    tau_max = int(pd.Timedelta(HORIZON) / pd.Timedelta(freq))
    R = resample(d, flows, freq)
    C = build_frame(R, flows_mode, use_ratio)
    var_names = list(C.columns)

    cfg = dict(name=name, START=START, END=END, FREQ=freq, TAU_MAX=tau_max,
               HORIZON=HORIZON, PC_ALPHA=PC_ALPHA, NOX_PAD=NOX_PAD,
               NOX_MIN_VALID=NOX_MIN_VALID, TEMP_DROP_ZERO=TEMP_DROP_ZERO,
               flows=flows_mode, ratio=use_ratio, vars=var_names)
    if method != "pcmciplus":        # keeps existing PCMCI+ tags/caches valid
        cfg["method"] = method
        cfg["lpcmci_kw"] = {k: str(v) for k, v in LPCMCI_KW.items()}
    runs = RUNS_BY_METHOD[method]
    tag  = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    path = runs / f"{name}_{HORIZON}_{tag}.npz"

    dg = diagnostics(C)
    head = (f"[{name}] {method} {len(var_names)}v tau={tau_max} ({HORIZON} @ {freq}) "
            f"| {dg['rows']} complete rows")
    if dg["max_corr"] is not None:
        head += f" | rank {dg['rank']}/{len(var_names)} | max|corr| {dg['max_corr']:.3f}"
    print(head)
    if dg["max_corr"] is not None and dg["max_corr"] > 0.95:
        print(f"    WARNING near-duplicate pair: {dg['worst']} ({dg['max_corr']:.4f})")
    if dg["rows"] < 500:
        print(f"    WARNING only {dg['rows']} complete rows -- results unreliable")

    if path.exists() and not force:
        print(f"    cached -> {path.name}")
        return path

    X = C.to_numpy(float)
    Z = (X - np.nanmean(X, axis=0)) / np.nanstd(X, axis=0)
    Z[~np.isfinite(Z)] = MISSING

    frame = pp.DataFrame(Z, var_names=var_names,
                         datatime={0: np.arange(len(Z))},
                         missing_flag=MISSING)
    la = link_assumptions_for(var_names, tau_max, method)
    t0 = time.time()

    if method == "lpcmci":
        alg = LPCMCI(dataframe=frame,
                     cond_ind_test=ParCorr(significance="analytic"),
                     verbosity=0)
        res = alg.run_lpcmci(tau_min=0, tau_max=tau_max, pc_alpha=PC_ALPHA,
                             link_assumptions=la, **LPCMCI_KW)
    else:
        alg = PCMCI(dataframe=frame,
                    cond_ind_test=ParCorr(significance="analytic"),
                    verbosity=0)
        res = alg.run_pcmciplus(tau_min=0, tau_max=tau_max, pc_alpha=PC_ALPHA,
                                link_assumptions=la)

    # For LPCMCI p_matrix is the *maximal* p-value over the tested conditioning
    # sets and is 0 where no test ran, so FDR over it is only meaningful once
    # the graph has already selected the edges -- see report().
    q = alg.get_corrected_pvalues(p_matrix=res["p_matrix"], tau_max=tau_max,
                                  fdr_method="fdr_bh")

    runs.mkdir(exist_ok=True)
    np.savez_compressed(path, graph=res["graph"], val=res["val_matrix"],
                        p=res["p_matrix"], q=q,
                        var_names=np.array(var_names), config=json.dumps(cfg))
    print(f"    ran in {time.time() - t0:.0f}s -> {path.name}")
    return path


def _report_pcmciplus(vn, g, v, q, tau, mins, j, alpha):
    for label, idx in (("causes of dNOx", lambda i, t: (i, j, t)),
                       ("effects of dNOx", lambda i, t: (j, i, t))):
        rows = [(vn[i], t, v[idx(i, t)], q[idx(i, t)])
                for i in range(len(vn)) for t in range(tau + 1)
                if g[idx(i, t)] == "-->" and q[idx(i, t)] < alpha and i != j]
        print(f"  --- {label} (q < {alpha}) ---")
        if not rows:
            print("    (none)")
        for nm, t, val, qq in sorted(rows, key=lambda r: -abs(r[2])):
            print(f"    {nm:<22} lag {t:>3} ({t * mins:>4} min)  "
                  f"MCI {val:+.3f}  q {qq:.1e}")

    self_rows = [(t, v[j, j, t], q[j, j, t]) for t in range(1, tau + 1)
                 if g[j, j, t] != "" and q[j, j, t] < alpha]
    if self_rows:
        s = ", ".join(f"lag {t}: {val:+.3f}" for t, val, _ in self_rows)
        print(f"  --- NOx autodependency ---\n    {s}")


# LPCMCI returns a PAG, so an edge mark carries the confounding verdict too.
# Read the mark left-to-right as var -> NOx.
LP_INTO_NOX = {"-->": "cause",
               "o->": "cause OR latent confounder",
               "<->": "latent confounder only (not a cause)",
               "o-o": "adjacent, direction undetermined"}
LP_OUT_OF_NOX = {"<--": "effect", "<-o": "effect OR latent confounder"}


def _report_lpcmci(vn, g, v, p, tau, mins, j):
    """List every edge touching NOx, bucketed by what its PAG mark claims."""
    causes, effects, other = [], [], []
    for i in range(len(vn)):
        if i == j:
            continue
        for t in range(tau + 1):
            # t == 0 is one undirected-in-storage edge; for t > 0 the two
            # directions live in separate cells of the graph array.
            pairs = ([(g[i, j, 0], v[i, j, 0], p[i, j, 0], "into")] if t == 0
                     else [(g[i, j, t], v[i, j, t], p[i, j, t], "into"),
                           (g[j, i, t], v[j, i, t], p[j, i, t], "out")])
            for s, val, pv, side in pairs:
                if not s:
                    continue
                row = (vn[i], t, s, val, pv)
                if side == "into" and s in LP_INTO_NOX:
                    (causes if s != "<->" else other).append(row)
                elif side == "into" and s in LP_OUT_OF_NOX:
                    effects.append(row)
                elif side == "out" and s in ("-->", "o->"):
                    effects.append(row)
                else:
                    other.append(row)

    for label, rows in (("causes of dNOx", causes),
                        ("effects of dNOx", effects),
                        ("confounded / undetermined", other)):
        print(f"  --- {label} ---")
        if not rows:
            print("    (none)")
        for nm, t, s, val, pv in sorted(rows, key=lambda r: -abs(r[3])):
            meaning = LP_INTO_NOX.get(s) or LP_OUT_OF_NOX.get(s) or ""
            print(f"    {nm:<22} lag {t:>3} ({t * mins:>4} min)  {s}  "
                  f"val {val:+.3f}  pmax {pv:.1e}   {meaning}")

    self_rows = [(t, v[j, j, t], g[j, j, t]) for t in range(1, tau + 1)
                 if g[j, j, t] != ""]
    if self_rows:
        s = ", ".join(f"lag {t}: {val:+.3f}" for t, val, _ in self_rows)
        print(f"  --- NOx autodependency ---\n    {s}")


def report(alpha=PC_ALPHA, method=METHOD):
    """Print the NOx causes and effects for every cached run of `method`."""
    runs = RUNS_BY_METHOD[method]
    files = sorted(runs.glob("*.npz"))
    if not files:
        print(f"no runs found in {runs}/")
        return
    for f in files:
        z    = np.load(f, allow_pickle=False)
        cfg  = json.loads(str(z["config"]))
        vn   = [str(s) for s in z["var_names"]]
        g, v, p, q = z["graph"], z["val"], z["p"], z["q"]
        tau  = cfg["TAU_MAX"]
        mins = int(pd.Timedelta(cfg["FREQ"]) / pd.Timedelta("1min"))
        j    = vn.index(TARGET_CLEAN)
        m    = cfg.get("method", "pcmciplus")

        print(f"\n{'=' * 74}\n{cfg['name']}  |  {m}  {cfg['FREQ']}  tau={tau}  "
              f"flows={cfg['flows']}  ratio={cfg['ratio']}  |  {f.name}\n{'=' * 74}")

        if m == "lpcmci":
            _report_lpcmci(vn, g, v, p, tau, mins, j)
        else:
            _report_pcmciplus(vn, g, v, q, tau, mins, j, alpha)


_CACHE = {}


def _worker(job):
    """One sweep entry, in its own process. Loads the data once per worker."""
    name, flows_mode, use_ratio, freq, force, method = job
    if "d" not in _CACHE:                              # reused across jobs in a worker
        _CACHE["d"], _CACHE["flows"] = load_raw()
    try:
        run_one(_CACHE["d"], _CACHE["flows"], name, flows_mode, use_ratio, freq,
                force=force, method=method)
        return name, "ok"
    except Exception as exc:
        return name, f"FAILED: {type(exc).__name__}: {exc}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-run even if cached")
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only SWEEP entries whose name starts with any of these")
    ap.add_argument("--report", action="store_true",
                    help="print NOx tables for cached runs and exit")
    ap.add_argument("--jobs", type=int, default=1,
                    help="run this many configs in parallel processes "
                         "(pair with OMP_NUM_THREADS=1)")
    ap.add_argument("--method", choices=sorted(RUNS_BY_METHOD), default=METHOD,
                    help="pcmciplus (causal sufficiency assumed) or lpcmci "
                         "(allows latent confounders, much slower)")
    args = ap.parse_args(argv)

    if args.report:
        report(method=args.method)
        return 0

    todo = SWEEP
    if args.only:
        todo = [s for s in SWEEP if any(s[0].startswith(p) for p in args.only)]
        if not todo:
            print(f"nothing in SWEEP matches {args.only}", file=sys.stderr)
            return 1

    print(f"{len(todo)} run(s), {args.method}, horizon {HORIZON}, "
          f"jobs {args.jobs}\n")
    t0 = time.time()

    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        jobs = [(*s, args.force, args.method) for s in todo]
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for name, status in pool.map(_worker, jobs):
                if status != "ok":
                    print(f"[{name}] {status}")
    else:
        d, flows = load_raw()
        for name, flows_mode, use_ratio, freq in todo:
            try:
                run_one(d, flows, name, flows_mode, use_ratio, freq,
                        force=args.force, method=args.method)
            except Exception as exc:                   # keep the sweep going
                print(f"[{name}] FAILED: {type(exc).__name__}: {exc}")

    print(f"\ndone in {time.time() - t0:.0f}s. "
          f"`python {Path(__file__).name} --method {args.method} --report` "
          f"to read them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
