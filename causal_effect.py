"""Turn a cached PCMCI+ graph into a number: how much does NOx move per move in X?

    python causal_effect.py --run <npz> --list                 # variables in that run
    python causal_effect.py --run <npz> --x ratio_main         # the estimate
    python causal_effect.py --run <npz> --x oil_1 pull         # several causes
    python causal_effect.py --run <npz> --x ratio_main --boot 200    # error bars

That is the whole interface. Everything else that could have been a flag is a
constant at the top of this file, because none of it turned out to be a choice
worth making per run.

The graph tells us *which* arrows exist; tigramite's CausalEffects turns that
into an adjustment set and a regression, so the coefficient is a causal slope
(back-door blocked) rather than a correlation.

WHAT IS ESTIMATED
-----------------
The sweep analyses first differences of (log) flows, so a step in the *level*
of X is a single impulse in the differenced series. So we intervene jointly on
all lags of X:

    do( dX_{t}=delta, dX_{t-1}=..=dX_{t-tau_max}=0 )   ->   E[ dY_t ]

evaluated once per lag. Because the frame is stationary, "delta at lag tau,
zero elsewhere" traces the impulse response psi(tau) = d dY_{t} / d dX_{t-tau},
and its cumulative sum is the level response:

    NOx(t+h) - NOx(t)   for a permanent step in X held from t onwards.

Intervening on every lag at once (rather than one lag at a time) is what makes
the cumulative sum honest: if X is autocorrelated, a single-lag intervention
would also flow forward through X's own future and be counted again at the next
lag. But intervening on all lags at once is also the strictest thing to ask of
the data, and sometimes no adjustment set can deliver it. When that happens the
script falls back to one lag at a time on its own, says so, and then tells you
whether X has self-edges -- because if it does not, the cumulative column is
still safe to read.

UNITS
-----
Flows enter the sweep as log(flow), so a step of +1 in the differenced log is a
factor e. Everything is reported per +1% of X (and, in the summary line, per
+10%), in the native unit of NOx. Non-log variables (pull, temperatures,
weather) are reported per +1 native unit.

CAVEATS
-------
* PCMCI+ returns a CPDAG. Contemporaneous 'o-o' edges are oriented by
  tigramite's Markov-equivalence-class member picker -- one arbitrary but
  valid choice. Conflicting 'x-x' edges are KEPT as '<->' (confounded), which
  is conservative and can make an effect non-identifiable; deleting them
  instead lets their association leak into the estimate. Edges listed in
  ORIENT are set by hand before any of that.
* Everything is assumed linear and the effect assumed constant over the window.
* tigramite drops any sample whose lag window touches a NaN, and the window is
  ~2x the horizon here, so the effective sample size is printed -- read it.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from tigramite import data_processing as pp
from tigramite.causal_effects import CausalEffects
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

import pcmci_sweep as ps

LOG_PREFIX = ("oil_", "oxy_", "ratio_")     # built from log(flow) in the sweep

# Composite temperatures the sweep has used at one time or another. A cached run
# records its variable *names* but not the recipe, so when an old run asks for a
# composite that pcmci_sweep.COMPOSITES no longer defines, we look it up here.
# Keep in sync with (or ahead of) COMPOSITES in pcmci_sweep.py.
KNOWN_COMPOSITES = {
    "arch_3_4": ["ARCH #3", "ARCH #4"],
    "bt_temp":  ["TEMP #08", "TEMP #09", "TEMP #10", "MELTER BT #11"],
}

# Fixed choices. These were command line flags once; none of them turned out to
# be a decision worth making per run, so they live here where they can be found
# and changed, instead of on the command line where they had to be understood.
DELTA    = 1.0        # intervention size in std units; linear model, so it cancels
STEP_PCT = 1.0        # report per +1% of X
CONF     = 0.95       # bootstrap confidence level for the printed table
SEED     = 42

EFFECTS_DIR = Path("effects/assertv2")   # saved estimates, one file per (run, x, y)

# Contemporaneous edges oriented by hand. PCMCI+ marks an edge 'x-x' when its
# orientation rules contradict each other, and a single such edge can be what
# makes an effect non-identifiable -- ratio_3 x-x NOx at lag 0 is the sole reason
# ratio_2's effect cannot be estimated in the 10min run.
#
# Asserting a direction is a claim about the furnace, not a finding from the
# data: the ratio is a manipulated input and NOx an emission outcome, and the
# analysis window ends before auto-control begins, so within-bin feedback from
# NOx to the ratio should not exist. Every assertion is printed at run time and
# recorded in the saved result file, so it travels with the numbers.
ORIENT = [
    ("ratio_3", "TMS_nox_clean"),
]

# filled in by apply_orientations with the assertions that actually applied to
# this run -- a main run has no ratio_3, so its results must not claim one
APPLIED_ORIENT = []

_T0 = time.time()


def log(msg, indent=4):
    """Timestamped progress line. Some stages here run for minutes in silence."""
    print(f"{' ' * indent}[{time.time() - _T0:6.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# rebuild the exact frame the graph was learnt on
# --------------------------------------------------------------------------- #
def load_run(path):
    z   = np.load(path, allow_pickle=False)
    cfg = json.loads(str(z["config"]))
    vn  = [str(s) for s in z["var_names"]]
    return z["graph"], vn, cfg


def restore_recipe(cfg, vn):
    """Point pcmci_sweep at the config THIS run was made with, not today's.

    The sweep evolves -- composites get switched off, knobs get retuned -- so a
    cached graph and a freshly built frame drift apart. The npz records the
    cleaning knobs directly, and its variable names imply the rest: whatever is
    left after the flows, pull, weather and the target are the temperatures, in
    order, and any of those that isn't a raw parquet column is a composite.
    """
    for k in ("NOX_PAD", "NOX_MIN_VALID", "TEMP_DROP_ZERO"):
        if k in cfg and cfg[k] != getattr(ps, k):
            print(f"using {k}={cfg[k]!r} from the cached run "
                  f"(pcmci_sweep has {getattr(ps, k)!r})")
            setattr(ps, k, cfg[k])

    # flows="main" reads OIL MAIN / OXY MAIN, which are not in today's FLOWC
    # (it lists the per-burner columns only), so load_raw would never fetch them.
    if cfg.get("flows") == "main":
        add = [c for c in ps.MAINC if c not in ps.FLOWC]
        if add:
            ps.FLOWC = ps.FLOWC + add
            print(f"flows='main': added {add} to the columns read from the parquet")

    # Only keep the non-temperature extras this run actually had.
    ps.MISC    = [c for c in ps.MISC if c in vn]
    ps.WEATHER = [c for c in ps.WEATHER if c in vn]

    known = set(ps.MISC) | set(ps.WEATHER) | {ps.TARGET_CLEAN}
    temps = [v for v in vn if v not in known and not v.startswith(LOG_PREFIX)]

    composites = {}
    for t in temps:
        if t in ps.COMPOSITES:
            composites[t] = ps.COMPOSITES[t]
        elif t in KNOWN_COMPOSITES:
            composites[t] = KNOWN_COMPOSITES[t]
        # else: a raw parquet column, carried through as-is

    ps.TEMPC      = temps
    ps.COMPOSITES = composites
    ps.COMP_SRC   = [c for m in composites.values() for c in m]
    ps.NONFLOW    = ps.TEMPC + ps.MISC + ps.WEATHER   # recomputed: set at import
    if composites:
        print("composites restored from the cached run: "
              + ", ".join(f"{k}={v}" for k, v in composites.items()))


def rebuild_frame(cfg, vn):
    """Re-derive the analysis frame from the parquet, same knobs as the sweep."""
    for k in ("START", "END"):
        if cfg[k] != getattr(ps, k):
            raise SystemExit(f"{k} in the cached run ({cfg[k]}) != pcmci_sweep.{k} "
                             f"({getattr(ps, k)}) -- the frame would not match the graph.")
    restore_recipe(cfg, vn)
    log("reading the parquet ...", indent=0)
    d, flows = ps.load_raw()
    R = ps.resample(d, flows, cfg["FREQ"])
    C = ps.build_frame(R, cfg["flows"], cfg["ratio"])
    log(f"frame rebuilt: {C.shape[0]} rows x {C.shape[1]} vars", indent=0)
    if list(C.columns) != vn:
        raise SystemExit(
            f"rebuilt columns differ from the cached run:\n"
            f"  cached : {vn}\n  rebuilt: {list(C.columns)}\n"
            f"If the cached run used a composite this script does not know the "
            f"members of, add it to KNOWN_COMPOSITES at the top of this file.")
    return C


def standardize(C):
    """Same standardisation as the sweep; returns (dataframe, sd per column)."""
    X  = C.to_numpy(float)
    sd = np.nanstd(X, axis=0)
    Z  = (X - np.nanmean(X, axis=0)) / sd
    Z[~np.isfinite(Z)] = ps.MISSING
    frame = pp.DataFrame(Z, var_names=list(C.columns),
                         datatime={0: np.arange(len(Z))},
                         missing_flag=ps.MISSING)
    return frame, sd


# --------------------------------------------------------------------------- #
# CPDAG -> one DAG
# --------------------------------------------------------------------------- #
def apply_orientations(graph, vn, orient):
    """Orient lag-0 edges by hand, per the ORIENT list at the top of this file.

    Runs before the conflict handling, so an asserted edge is no longer 'x-x'
    and will not be turned into '<->'. Prints what it overrode: '(was x-x)' is
    the intended use, '(was -->)' means you contradicted the data, and
    '(no edge)' means you invented an adjacency PCMCI+ tested and rejected.
    """
    g = graph.copy()
    APPLIED_ORIENT.clear()
    for a, b in orient:
        missing = [nm for nm in (a, b) if nm not in vn]
        if missing:
            print(f"ORIENT: {missing} not in this run -- skipped")
            continue
        i, j = vn.index(a), vn.index(b)
        was = g[i, j, 0] or "(no edge)"
        g[i, j, 0], g[j, i, 0] = "-->", "<--"
        APPLIED_ORIENT.append(f"{a}->{b}")
        print(f"ASSERTED {a} --> {b} at lag 0 (was {was}) -- domain assumption, "
              f"not a finding")
    return g


def cpdag_to_dag(graph, frame):
    """Pick one member of the Markov equivalence class.

    PCMCI+ marks a contemporaneous edge 'x-x' when its orientation rules
    contradict each other. Deleting those is tempting and wrong: the adjacency
    is real, only its direction is unknown, and a deleted edge is one the
    adjustment set will never account for -- the association leaks into the
    estimate instead. So we keep them as '<->', which says "these two are
    confounded by something we cannot orient". That is the conservative reading,
    and it makes the graph an ADMG rather than a DAG.
    """
    g = graph.copy()
    conflict = (g == "x-x")
    n_conflict = int(conflict.sum() // 2)
    g[conflict] = ""                     # _get_dag_from_cpdag cannot take 'x-x'
    n_undir = int((g == "o-o").sum() // 2)

    pcmci = PCMCI(dataframe=frame, cond_ind_test=ParCorr(), verbosity=0)
    dag = pcmci._get_dag_from_cpdag(cpdag_graph=g,
                                    variable_order=list(range(g.shape[0])))
    if n_undir:
        print(f"oriented {n_undir} undirected (o-o) contemporaneous edge(s) "
              f"-- one valid choice out of several")
    if n_conflict:
        dag[conflict] = "<->"
        print(f"kept {n_conflict} conflicting (x-x) edge(s) as '<->' "
              f"(unorientable -> treated as confounded)")

    left = set(np.unique(dag)) - {"", "-->", "<--", "<->"}
    if left:
        raise SystemExit(f"graph still contains unsupported marks {left}.")
    return dag


def stationary_to_tsg(graph, model_tau_max):
    """Roll the stationary graph out into an explicit time series DAG.

    We do this instead of handing CausalEffects graph_type='stationary_dag',
    because that path always runs a latent projection -- three path searches for
    every pair of time series nodes, O(N^2 tau^2) of them -- even when there are
    no hidden variables to project out. At 16 variables and 28 lags that is
    ~600k graph traversals in Python and it takes forever. With graph_type
    'tsg_dag' the projection is skipped entirely.

    The expansion itself is just stationarity: an edge i --> j at lag tau means
    (i, -(t+tau)) --> (j, -t) at every t. tigramite has this same loop in
    graphs.py, commented out and marked "TO BE REVISED!".
    """
    N, _, nlag = graph.shape
    stat = nlag - 1
    tsg  = np.full((N, N, model_tau_max + 1, model_tau_max + 1), "", dtype="<U3")
    for i, j, tau in zip(*np.where(graph != "")):
        mark = graph[i, j, tau]
        if mark == "<--":
            if tau > 0:
                raise SystemExit(f"graph[{i},{j},{tau}] = '<--' points backwards "
                                 f"in time; the stationary convention stores "
                                 f"lagged links as '-->' only.")
            continue                     # contemporaneous mirror of a '-->'
        for tj in range(model_tau_max + 1):
            ti = tj + tau
            if ti > model_tau_max:
                break
            if mark == "<->":
                tsg[i, j, ti, tj] = tsg[j, i, tj, ti] = "<->"
            else:
                tsg[i, j, ti, tj] = "-->"
                tsg[j, i, tj, ti] = "<--"
    n_bi = int((tsg == "<->").sum() // 2)
    kind = "ADMG" if n_bi else "DAG"
    print(f"rolled out to a {N} x {model_tau_max + 1} time series {kind} "
          f"({int((tsg == '-->').sum())} directed"
          + (f", {n_bi} bidirected" if n_bi else "")
          + f" edges), stationary tau_max was {stat}", flush=True)
    return tsg


# --------------------------------------------------------------------------- #
# the estimate
# --------------------------------------------------------------------------- #
def effects_joint(dag, frame, vn, xname, yname, lags, boot, blocklength):
    """All lags intervened at once. Returns None if that is not identifiable."""
    xi, yi = vn.index(xname), vn.index(yname)
    nlag   = len(lags)
    X  = [(xi, -t) for t in lags]
    # row 0 = baseline (no intervention), row t+1 = delta at lag t only
    iv = np.vstack([np.zeros(nlag), np.eye(nlag) * DELTA])
    pred, draws = _one_fit(dag, frame, X, [(yi, 0)], iv, boot, blocklength,
                           f"{xname} at lags {lags[0]}..{lags[-1]} -> {yname}",
                           strict=False)
    return None if pred is None else _contrast(pred, draws)


# _contrast returns the per-draw contrasts as well as the percentiles, so the
# raw bootstrap survives into the saved file. Changing the confidence level is
# then a plotting decision, not a reason to refit for another four minutes.


def effects_single(dag, frame, vn, xname, yname, lags, boot, blocklength):
    """One lag at a time. Lags that are not identifiable come back as NaN."""
    xi, yi = vn.index(xname), vn.index(yname)
    nlag   = len(lags)
    psi = np.zeros(nlag)
    lo  = np.full(nlag, np.nan)
    hi  = np.full(nlag, np.nan)
    all_draws = np.full((boot, nlag), np.nan) if boot else None
    for k, t in enumerate(lags):
        log(f"--- lag {t} ({k + 1}/{nlag}) ---")
        iv = np.array([[0.0], [DELTA]])
        pred, draws = _one_fit(dag, frame, [(xi, -t)], [(yi, 0)], iv, boot,
                               blocklength, f"{xname} at lag {t} -> {yname}",
                               strict=False)
        if pred is None:                  # not identifiable at this lag
            psi[k] = np.nan
            continue
        e, l, h, d = _contrast(pred, draws)
        psi[k] = e[0]
        if l is not None:
            lo[k], hi[k] = l[0], h[0]
            all_draws[:, k] = d[:, 0]
    return (psi, (None if boot == 0 else lo), (None if boot == 0 else hi),
            all_draws)


def _contrast(pred, draws):
    """Intervened rows minus the baseline row, with CIs from the same draws.

    The baseline and the intervened predictions move together from bootstrap
    sample to bootstrap sample, so the difference must be taken *within* each
    draw before the percentiles -- taking percentiles first would fold the
    baseline's own spread into the interval and widen it for no reason.
    """
    eff = pred[1:] - pred[0]
    if draws is None:
        return eff, None, None, None
    d  = draws[:, 1:] - draws[:, :1]
    q  = 100 * (1 - CONF) / 2
    lo, hi = np.percentile(d, [q, 100 - q], axis=0)
    return eff, lo, hi, d


def _one_fit(dag, frame, X, Y, iv, boot, blocklength, label, strict=True):
    """Fit one total-effect model and predict. Returns (pred, bootstrap draws)."""
    gtype = "tsg_admg" if (dag == "<->").any() else "tsg_dag"
    ce = CausalEffects(dag, graph_type=gtype, X=X, Y=Y,
                       hidden_variables=None, verbosity=0)
    if ce.no_causal_path:
        log("no causal path from X to Y -- effect is exactly 0")
        return np.zeros(len(iv)), None

    log("searching for the adjustment set ...")
    oset = ce.get_optimal_set()
    if oset is False:
        log(f"NOT IDENTIFIABLE: {label}")
        if not strict:
            return None, None
        raise SystemExit("no adjustment set makes this effect estimable.")
    log(f"adjustment set: {len(oset)} nodes (+{len(X)} X)")

    n = _n_samples(ce, frame, X, Y, oset)
    npar = len(X) + len(oset)
    log(f"{n} samples survive the NaN lag windows, for {npar} coefficients")
    if n <= npar:
        log("WARNING underdetermined -- the numbers would be meaningless. "
            "Try a coarser run (30min instead of 10min).")
    elif n < 10 * npar:
        log(f"WARNING only {n / npar:.1f} samples per coefficient -- estimates "
            f"are indicative only.")

    fit_args = dict(dataframe=frame, estimator=LinearRegression(),
                    adjustment_set=list(oset))
    log("fitting ...")
    ce.fit_total_effect(**fit_args)
    pred = np.asarray(ce.predict_total_effect(intervention_data=iv)).ravel()
    log("fitted")

    if boot == 0:
        return pred, None

    # Rolled by hand rather than via fit_bootstrap_of, which runs all N draws
    # in one silent call and deep-copies the dataset for each of them.
    draws = np.zeros((boot, len(iv)))
    every = max(1, boot // 20)
    t0    = time.time()
    try:
        for b in range(boot):
            frame.bootstrap = dict(boot_blocklength=blocklength,
                                   random_state=np.random.default_rng(SEED * boot + b))
            ce.fit_total_effect(**fit_args)
            draws[b] = np.asarray(
                ce.predict_total_effect(intervention_data=iv)).ravel()
            if b == 0 or (b + 1) % every == 0:
                rate = (time.time() - t0) / (b + 1)
                log(f"bootstrap {b + 1}/{boot} "
                    f"(~{rate * (boot - b - 1):.0f}s left)")
    finally:
        frame.bootstrap = None
    return pred, draws


def _n_samples(ce, frame, X, Y, Z):
    """Rows that survive: tigramite drops any row whose lag window hits a NaN."""
    out = frame.construct_array(X=list(X), Y=list(Y), Z=[], extraZ=list(Z),
                                tau_max=ce.tau_max, cut_off="tau_max",
                                remove_overlaps=True)
    return int(np.asarray(out[0]).shape[-1])


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
def show(xname, yname, psi, lo, hi, sd, vn, lags, mins, mode, selfdep=False):
    """Convert standardised effects to native units and print the table."""
    xi, yi = vn.index(xname), vn.index(yname)
    scale  = sd[yi] / sd[xi] / DELTA          # std effect -> d(NOx) per unit dX
    islog  = xname.startswith(LOG_PREFIX)
    bump   = np.log1p(STEP_PCT / 100) if islog else 1.0
    unit   = f"+{STEP_PCT:g}% {xname}" if islog else f"+1 unit {xname}"

    psi_u = np.asarray(psi) * scale * bump
    cum   = np.nancumsum(psi_u)           # a skipped lag contributes nothing
    n_skip = int(np.isnan(psi_u).sum())
    lo_u  = None if lo is None else np.asarray(lo) * scale * bump
    hi_u  = None if hi is None else np.asarray(hi) * scale * bump

    print(f"\n  {'lag':>4} {'min':>6} | {'d(NOx) per ' + unit:>26} | cumulative NOx")
    print(f"  {'-' * 62}")
    for k, (t, p, c) in enumerate(zip(lags, psi_u, cum)):
        if np.isnan(p):
            print(f"  {t:>4} {t * mins:>6} | {'not identifiable':>26} | "
                  f"{c:>+10.4f}")
            continue
        ci, star = "", ""
        if lo_u is not None and np.isfinite(lo_u[k]):
            ci = f"  [{lo_u[k]:+.4f}, {hi_u[k]:+.4f}]"
            star = "  *" if lo_u[k] * hi_u[k] > 0 else ""
        print(f"  {t:>4} {t * mins:>6} | {p:>+26.4f}{ci} | {c:>+10.4f}{star}")

    tot = cum[-1]
    print(f"\n  total (level) effect of a permanent {unit}: "
          f"{tot:+.3f} NOx after {lags[-1] * mins / 60:.1f} h")
    if n_skip:
        print(f"  ...but {n_skip} lag(s) were not identifiable and count as 0 "
              f"in that total, so read it as a lower bound on the magnitude.")
    if islog:
        print(f"  ... and of a permanent +{10 * STEP_PCT:g}% {xname}: "
              f"{tot * np.log1p(10 * STEP_PCT / 100) / bump:+.3f} NOx")
    if lo is not None:
        print(f"  brackets are {CONF:.0%} bootstrap CIs; * marks a lag whose own "
              f"CI excludes 0")
    if mode == "single" and selfdep:
        print(f"  NOTE one lag at a time, and {xname} depends on its own past, "
              f"so each row's intervention also flows through {xname}'s future "
              f"and the cumulative column double counts. Read the per-lag column.")
    elif mode == "single":
        print(f"  ({xname} has no self-edges in the graph, so nothing propagates "
              f"through its own future and the cumulative column is safe.)")


def _slug(s):
    """Filenames from variable names like 'ARCH #1' or 'OIL MAIN_cleaned'."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)


def save_result(runpath, cfg, xname, yname, psi, lo, hi, sd, vn, lags, mins,
                mode, selfdep, boot, draws=None):
    """Write one estimate to effects/ so a notebook can plot it without refitting.

    Stored in native NOx units (per STEP_PCT% of X), which is what you plot,
    plus the standardised values and the scale factor so nothing is lost.

    `status` distinguishes the two kinds of blank row the printed table shows
    differently: a lag with no causal path is a genuine zero, a non-identifiable
    lag is a refusal to answer. A plot must not draw those the same way.
    """
    xi, yi = vn.index(xname), vn.index(yname)
    islog  = xname.startswith(LOG_PREFIX)
    scale  = sd[yi] / sd[xi] / DELTA * (np.log1p(STEP_PCT / 100) if islog else 1.0)

    psi = np.asarray(psi, float)
    nan = np.full(len(psi), np.nan)
    status = np.full(len(psi), "ok", dtype="<U16")
    status[psi == 0.0]    = "no_causal_path"
    status[np.isnan(psi)] = "not_identifiable"

    meta = dict(run=str(runpath), name=cfg["name"], freq=cfg["FREQ"],
                tau_max=cfg["TAU_MAX"], flows=cfg["flows"], ratio=cfg["ratio"],
                x=xname, y=yname, mode=mode, minutes_per_lag=mins,
                step_pct=STEP_PCT, conf=CONF, boot=int(boot),
                selfdep=bool(selfdep), islog=bool(islog), scale=float(scale),
                # only the assertions that actually applied to THIS run
                orient=list(APPLIED_ORIENT))

    EFFECTS_DIR.mkdir(exist_ok=True)
    path = EFFECTS_DIR / f"{Path(runpath).stem}__{_slug(xname)}__{_slug(yname)}.npz"
    np.savez_compressed(
        path,
        lags=np.asarray(lags),
        minutes=np.asarray(lags) * mins,
        psi=psi * scale,
        lo=(nan if lo is None else np.asarray(lo, float) * scale),
        hi=(nan if hi is None else np.asarray(hi, float) * scale),
        psi_std=psi,
        status=status,
        # every bootstrap draw, in native units: lets a notebook recompute the
        # interval at any confidence level without refitting anything
        draws=(np.zeros((0, len(psi))) if draws is None
               else np.asarray(draws, float) * scale),
        meta=json.dumps(meta))
    print(f"  saved -> {path}")
    return path


def load_effect(path):
    """Read one saved estimate back. Returns a dict, meta already parsed."""
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files if k != "meta"}
    d["meta"] = json.loads(str(z["meta"]))
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="path to a cached .npz from pcmci_sweep")
    ap.add_argument("--x", nargs="*", default=[], help="cause variable(s)")
    ap.add_argument("--y", default=ps.TARGET_CLEAN, help="effect variable")
    ap.add_argument("--list", action="store_true", help="print the run's variables and exit")
    ap.add_argument("--boot", type=int, default=0,
                    help="bootstrap samples for error bars (try 200; slower)")
    ap.add_argument("--maxlag", type=int, default=None,
                    help="stop at this lag instead of the run's tau_max. The "
                         "response lands in lag 1 and the tail is noise, so this "
                         "mostly buys speed -- with --boot it is one fit per lag")
    args = ap.parse_args(argv)

    graph, vn, cfg = load_run(Path(args.run))
    tau_max = cfg["TAU_MAX"]
    mins    = int(pd.Timedelta(cfg["FREQ"]) / pd.Timedelta("1min"))
    print(f"{cfg['name']}  |  {cfg['FREQ']}  tau_max={tau_max} ({cfg['HORIZON']})  "
          f"flows={cfg['flows']}  ratio={cfg['ratio']}")
    if args.list:
        print("\nvariables:\n  " + "\n  ".join(vn))
        return 0
    if not args.x:
        raise SystemExit("nothing to do: pass --x VAR [VAR ...] (or --list)")
    for name in args.x + [args.y]:
        if name not in vn:
            raise SystemExit(f"{name!r} not in this run. --list to see the variables.")

    top  = tau_max if args.maxlag is None else min(args.maxlag, tau_max)
    lags = list(range(top + 1))

    C = rebuild_frame(cfg, vn)
    frame, sd = standardize(C)
    dag = cpdag_to_dag(apply_orientations(graph, vn, ORIENT), frame)
    # tigramite's own conservative bound: deep enough to hold every parent of X,
    # Y and the mediators between them. It scales with the deepest X lag, so
    # --maxlag shrinks the window by exactly as much as it shrinks the question
    # -- the truncation stays valid.
    tsg = stationary_to_tsg(dag, top + tau_max)

    for k, xname in enumerate(args.x, 1):
        print(f"\n{'=' * 74}\n[{k}/{len(args.x)}] {xname}  ->  {args.y}   "
              f"(lags 0..{top}, {top * mins} min)\n{'=' * 74}", flush=True)

        # Intervening on every lag at once is the question you actually want --
        # "turn it up and leave it there". It is also the strictest demand on
        # the data, so when it cannot be answered, fall back to asking one lag
        # at a time rather than making the user pick a mode they cannot assess.
        mode = "joint"
        res  = effects_joint(tsg, frame, vn, xname, args.y, lags, args.boot,
                             tau_max)
        if res is None:
            log("joint effect not identifiable -- falling back to one lag "
                "at a time")
            mode = "single"
            res  = effects_single(tsg, frame, vn, xname, args.y, lags,
                                  args.boot, tau_max)
        psi, lo, hi, draws = res

        xi = vn.index(xname)
        selfdep = bool((graph[xi, xi, 1:] != "").any())
        if np.any(np.nan_to_num(psi)):
            show(xname, args.y, psi, lo, hi, sd, vn, lags, mins, mode,
                 selfdep=selfdep)
        else:
            print("  no causal path in the graph -- effect is zero by construction")
        # saved either way: "this variable has no path to NOx" is a result, and a
        # plot that silently omits it would read as "not measured" instead.
        save_result(args.run, cfg, xname, args.y, psi, lo, hi, sd, vn, lags,
                    mins, mode, selfdep, args.boot, draws)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
