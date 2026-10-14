"""
Robust small-N statistics for the EEG creativity study.

Motivation
----------
With ~18 matched subjects and only a handful of aha epochs each (often 0-10,
and unbalanced between conditions), the classical paired t-test + Pearson +
"trending p<0.1" pipeline is not defensible:

  * The paired t-test assumes the *differences* are normal. Testing that at
    this N is pointless (Shapiro-Wilk has no power), so we do not test it and
    we do not rely on it -- we use exact nonparametric inference instead.
  * Wilcoxon signed-rank tests make no normality assumption for paired
    condition contrasts.
  * For multiple channel/window tests, scripts report FDR-adjusted Wilcoxon
    p-values.
  * Unequal aha-trial counts make subject means unequally reliable, and a
    baseline-free absolute TFR turns a trial-count difference into a fake power
    effect. A mixed model on single-trial amplitudes,
    amplitude ~ condition + (1 | subject), weights subjects by their evidence
    and is the honest primary test.
  * Correlations use Kendall's tau with a bootstrap CI (rank-based, robust in
    small samples), and we report the effect size with its interval rather
    than a bare p-value.

This module holds the shared machinery; the analysis scripts call into it.
"""

import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats

try:
    import statsmodels.formula.api as smf
    _HAS_STATSMODELS = True
except Exception:                       # pragma: no cover
    _HAS_STATSMODELS = False

_SEED = 0


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def fdr_bh(p_values):
    """Benjamini-Hochberg FDR-adjusted p-values."""
    p = np.asarray(p_values, float)
    out = np.full_like(p, np.nan, dtype=float)
    good = np.isfinite(p)
    if not good.any():
        return out

    pv = p[good]
    m = pv.size
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(m)
    adjusted[order] = np.clip(ranked, 0, 1)
    out[good] = adjusted
    return out


def shapiro_normality(values, alpha=0.05):
    """
    Shapiro-Wilk normality diagnostic. Reporting only; do not switch tests
    automatically based on this small-N check.
    """
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    out = {
        "n": int(x.size),
        "w": np.nan,
        "p": np.nan,
        "normal_at_alpha": None,
        "note": "",
    }
    if x.size < 3:
        out["note"] = "n<3"
        return out
    if np.all(x == x[0]):
        out["note"] = "constant"
        return out
    res = stats.shapiro(x)
    out.update(
        w=float(res.statistic),
        p=float(res.pvalue),
        normal_at_alpha=bool(res.pvalue >= alpha),
        note="normal-ish" if res.pvalue >= alpha else "non-normal",
    )
    return out


def exact_wilcoxon(control, treatment):
    """
    Exact Wilcoxon signed-rank test (falls back to normal approx if ties/zeros
    force it). Returns dict: n, stat, p, method.
    """
    c = np.asarray(control, float)
    t = np.asarray(treatment, float)
    mask = np.isfinite(c) & np.isfinite(t)
    c, t = c[mask], t[mask]
    n = c.size
    out = {"n": n, "stat": np.nan, "p": np.nan, "method": None}
    if n < 1 or np.all(t - c == 0):
        return out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = stats.wilcoxon(t, c, zero_method="wilcox", method="exact")
            method = "exact"
        except ValueError:
            res = stats.wilcoxon(t, c, zero_method="wilcox", method="approx")
            method = "approx"
    out.update(stat=float(res.statistic), p=float(res.pvalue), method=method)
    return out


def wilcoxon_family(control_matrix, treatment_matrix, labels):
    """
    Wilcoxon signed-rank tests across a family of paired contrasts, with FDR.

    Returns DataFrame: label, n, mean_diff, wilcoxon_w, p_unc, p_fdr, method.
    """
    C = np.asarray(control_matrix, float)
    T = np.asarray(treatment_matrix, float)
    if C.ndim == 1:
        C = C[:, None]
    if T.ndim == 1:
        T = T[:, None]

    rows = []
    for j, label in enumerate(labels):
        c = C[:, j]
        t = T[:, j]
        mask = np.isfinite(c) & np.isfinite(t)
        d = t[mask] - c[mask]
        wilc = exact_wilcoxon(c, t)
        rows.append({
            "label": label,
            "n": wilc["n"],
            "mean_diff": float(np.mean(d)) if d.size else np.nan,
            "wilcoxon_w": wilc["stat"],
            "p_unc": wilc["p"],
            "method": wilc["method"],
        })

    out = pd.DataFrame(rows)
    out["p_fdr"] = fdr_bh(out["p_unc"])
    return out


def paired_ttest(control, treatment):
    """
    Paired (dependent) t-test on treatment - control.

    Same interface as exact_wilcoxon: aligns finite pairs and returns
    dict: n, stat (t), p (two-sided), method.
    """
    c = np.asarray(control, float)
    t = np.asarray(treatment, float)
    mask = np.isfinite(c) & np.isfinite(t)
    c, t = c[mask], t[mask]
    n = c.size
    out = {"n": n, "stat": np.nan, "p": np.nan, "method": "paired-t"}
    if n < 2 or np.all(t - c == 0):
        return out
    res = stats.ttest_rel(t, c)
    out.update(stat=float(res.statistic), p=float(res.pvalue))
    return out


# --------------------------------------------------------------------------- #
# Network-Based Statistic (Zalesky, Fornito & Bullmore, 2010)
# --------------------------------------------------------------------------- #
def _edge_components(sup_cols, edges, n_nodes):
    """Connected components among the suprathreshold edges (union-find).

    Returns a list of edge-column-index lists, one per component that has at
    least one edge. `edges[k]` is the (i, j) node pair of column k.
    """
    parent = list(range(n_nodes))

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:          # path compression
            parent[x], x = root, parent[x]
        return root

    for k in sup_cols:
        i, j = edges[k]
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    comp = {}
    for k in sup_cols:
        i, _ = edges[k]
        comp.setdefault(find(i), []).append(int(k))
    return list(comp.values())


def nbs_paired(diff_matrix, edges, n_nodes, thresh=2.5, n_perm=5000,
               tail="both", seed=_SEED):
    """
    Within-subject Network-Based Statistic on paired differences.

    The NBS finds connected sub-networks of edges whose condition effect is
    stronger than a primary threshold, and assigns each a family-wise-error
    p-value from a permutation null on the *size of the largest component*.
    Because the design is within-subject (each subject has a control and a
    treatment network), the exchangeability under H0 is a per-subject sign flip
    of the paired difference -- the same logic as paired_signflip, lifted to a
    whole network.

    Parameters
    ----------
    diff_matrix : (n_subj, n_edges) paired differences (treatment - control),
                  one column per edge.
    edges       : sequence of (i, j) node-index pairs, one per column.
    n_nodes     : number of nodes (channels).
    thresh      : primary edge threshold on the one-sample t statistic. Only
                  edges beyond it can join a component. Higher -> sparser,
                  more focal networks (the NBS result depends on this choice).
    n_perm      : number of sign-flip permutations for the null.
    tail        : "pos" (treatment > control), "neg" (treatment < control), or
                  "both" (|t|; a component may then mix directions).

    Returns
    -------
    dict: edge_t (n_edges t-values), components (list of {edges, size, sum_t,
          p_fwer} sorted largest-first), null_max (permuted max sizes), thresh,
          tail, n_perm, n_subj.
    """
    D = np.asarray(diff_matrix, float)
    n_subj, n_edges = D.shape
    edges = [tuple(int(x) for x in e) for e in edges]

    def tstat(M):
        m = M.mean(axis=0)
        s = M.std(axis=0, ddof=1)
        t = m / (s / np.sqrt(M.shape[0]) + 1e-30)
        return np.nan_to_num(t, nan=0.0)

    def supra(t):
        stat = t if tail == "pos" else (-t if tail == "neg" else np.abs(t))
        return np.where(stat >= thresh)[0]

    t_obs = tstat(D)
    obs = _edge_components(supra(t_obs), edges, n_nodes)

    rng = np.random.RandomState(seed)
    null_max = np.empty(n_perm)
    for p in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=(n_subj, 1))
        comps = _edge_components(supra(tstat(signs * D)), edges, n_nodes)
        null_max[p] = max((len(c) for c in comps), default=0)

    components = []
    for c in sorted(obs, key=len, reverse=True):
        size = len(c)
        p_fwer = (1 + int(np.sum(null_max >= size))) / (n_perm + 1)
        components.append({
            "edges": c,
            "size": size,
            "sum_t": float(np.sum(np.abs(t_obs[c]))),
            "p_fwer": float(p_fwer),
        })

    return {"edge_t": t_obs, "components": components, "null_max": null_max,
            "thresh": thresh, "tail": tail, "n_perm": n_perm, "n_subj": n_subj}


# --------------------------------------------------------------------------- #
# Correlation with bootstrap CI
# --------------------------------------------------------------------------- #
def kendall_bootstrap(x, y, n_boot=10000, ci=95, seed=_SEED):
    """
    Kendall's tau with a percentile bootstrap CI.

    Returns dict: n, tau, p, ci_low, ci_high (of tau).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = x.size
    out = {"n": n, "tau": np.nan, "p": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    if n < 3:
        return out

    tau, p = stats.kendalltau(x, y)
    rng = np.random.RandomState(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        # A resample with no variation in x or y yields nan; keep it, filter later.
        boot[b] = stats.kendalltau(x[idx], y[idx])[0]
    boot = boot[np.isfinite(boot)]
    lo, hi = np.percentile(boot, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    out.update(tau=float(tau), p=float(p), ci_low=float(lo), ci_high=float(hi))
    return out


def spearman_bootstrap(x, y, n_boot=10000, ci=95, seed=_SEED):
    """
    Spearman's rho with a percentile bootstrap CI.

    Returns dict: n, rho, p, ci_low, ci_high (of rho).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = x.size
    out = {"n": n, "rho": np.nan, "p": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    if n < 3:
        return out

    rho, p = stats.spearmanr(x, y)
    rng = np.random.RandomState(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        boot[b] = stats.spearmanr(x[idx], y[idx])[0]
    boot = boot[np.isfinite(boot)]
    lo, hi = np.percentile(boot, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    out.update(rho=float(rho), p=float(p), ci_low=float(lo), ci_high=float(hi))
    return out


# --------------------------------------------------------------------------- #
# Mixed model on single-trial amplitudes
# --------------------------------------------------------------------------- #
def mixedlm_condition(long_df, standardize=True):
    """
    Fit  amplitude ~ condition + (1 | subject) + (1 | subject:session)  on
    single-trial data, where `session` = subject x condition (each subject has
    one control and one treatment session here).

    Why the extra session term matters -- PSEUDOREPLICATION.  `condition` is a
    property of the whole session, not of the individual trial/window: every
    trial within a session carries the same condition label.  A model with only
    (1 | subject) therefore estimates the condition SE from the hundreds/
    thousands of *within-session* trials as if they were independent evidence
    for the condition effect, and returns a spuriously tiny p (e.g. p=0.0000
    from 3000 windows / 18 subjects).  The nested session random intercept moves
    the condition test onto session-level variance (effectively ~N sessions),
    which is the honest unit of replication and agrees with the subject-level
    Wilcoxon tests.  It still uses single trials, so trial-count
    imbalance is handled (subjects/sessions are weighted by their evidence) -- it
    just does not manufacture significance from pseudoreplicates.

    Parameters
    ----------
    long_df : DataFrame with columns 'subject', 'condition' (0/1), 'amplitude'.
        An optional 'session' column overrides the default subject-x-condition
        session id (use it when a subject truly has several sessions per
        condition).
    standardize : z-score amplitude before fitting (numerical stability for the
        very small V^2 scale of TFR/PSD power; the condition test is unchanged
        and the coefficient is then in standard-deviation units).

    Returns
    -------
    dict: coef, se, p, ci_low, ci_high, n_obs, n_sub, n_trials_c, n_trials_t,
          standardized (bool), nested (bool: session term included), error.
    """
    out = {"coef": np.nan, "se": np.nan, "p": np.nan,
           "ci_low": np.nan, "ci_high": np.nan, "n_obs": 0, "n_sub": 0,
           "n_trials_c": 0, "n_trials_t": 0,
           "standardized": standardize, "nested": False, "error": None}
    if not _HAS_STATSMODELS:
        out["error"] = "statsmodels not installed"
        return out

    d = long_df.dropna(subset=["amplitude", "condition", "subject"]).copy()
    out["n_obs"] = len(d)
    out["n_sub"] = d["subject"].nunique()
    out["n_trials_c"] = int((d["condition"] == 0).sum())
    out["n_trials_t"] = int((d["condition"] == 1).sum())
    if out["n_sub"] < 3 or len(d) < 6 or d["condition"].nunique() < 2:
        out["error"] = "insufficient data for mixed model"
        return out

    if "session" not in d.columns:
        d["session"] = d["subject"].astype(str) + "_" + d["condition"].astype(int).astype(str)

    if standardize:
        sd = d["amplitude"].std()
        if sd > 0:
            d["amplitude"] = (d["amplitude"] - d["amplitude"].mean()) / sd

    def _extract(mf, nested):
        ci = mf.conf_int().loc["condition"]
        out.update(coef=float(mf.params["condition"]), se=float(mf.bse["condition"]),
                   p=float(mf.pvalues["condition"]),
                   ci_low=float(ci[0]), ci_high=float(ci[1]), nested=nested)

    # Primary model: nested session random intercept guards against
    # pseudoreplication. Only meaningful when sessions carry >1 trial on
    # average; otherwise the session term is confounded with residual and we
    # fall back to the subject-only model.
    trials_per_session = len(d) / max(d["session"].nunique(), 1)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if trials_per_session >= 1.5:
                md = smf.mixedlm("amplitude ~ condition", d, groups=d["subject"],
                                 vc_formula={"session": "0 + C(session)"})
                mf = md.fit(reml=True, method="lbfgs")
                _extract(mf, nested=True)
            else:
                md = smf.mixedlm("amplitude ~ condition", d, groups=d["subject"])
                mf = md.fit(reml=True, method="lbfgs")
                _extract(mf, nested=False)
    except Exception:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                md = smf.mixedlm("amplitude ~ condition", d, groups=d["subject"])
                mf = md.fit(reml=True, method="lbfgs")
            _extract(mf, nested=False)
        except Exception as e:
            out["error"] = str(e)
    return out


# --------------------------------------------------------------------------- #
# Pretty printing
# --------------------------------------------------------------------------- #
def format_contrast(name, wilc, mm=None, corrected_p=None):
    """One-line-ish summary for a single paired contrast."""
    fdr = "" if corrected_p is None else f"  |  FDR p = {corrected_p:.4f}"
    lines = [f"[{name}]  N = {wilc['n']} paired"]
    lines.append(
        f"    Wilcoxon signed-rank:  W = {wilc['stat']:.1f},  p = {wilc['p']:.4f}{fdr}  ({wilc['method']})"
    )
    if mm is not None:
        if mm["error"]:
            lines.append(f"    mixed model:     n/a ({mm['error']})")
        else:
            lines.append(
                f"    mixed model:     b = {mm['coef']:+.3f} [{mm['ci_low']:+.3f}, {mm['ci_high']:+.3f}]"
                f"{' SD' if mm['standardized'] else ''},  p = {mm['p']:.4f}"
                f"  ({mm['n_obs']} trials: {mm['n_trials_c']} C / {mm['n_trials_t']} T, {mm['n_sub']} subj)")
    return "\n".join(lines)
