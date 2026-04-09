"""
run_me_pairwise.py
==================
Runs only the Cross-Subreddit Mixed-Effects model from §4.7, and extends
the analysis to recover all pairwise comparisons between non-reference
subreddits via contrast testing on the fitted model's covariance matrix.

The standard run_mixed_effects() in regressions.py reports each subreddit
relative to the reference (r/Parenting, the first alphabetically). This
script refits the identical model, preserves res.cov_params() for each
metric, then computes

    contrast_{s,s'} = δ_s − δ_{s'}
    SE = sqrt( Var(δ_s) + Var(δ_{s'}) − 2·Cov(δ_s, δ_{s'}) )
    z  = contrast / SE

for all C(K,2) pairs of non-reference subreddits, where δ_s and δ_{s'} are
the subreddit dummy coefficients (θ_s in equation (17)). With 3 non-reference
subreddits (college, retirement, teenagers) and 6 metrics this yields 18
pairwise contrasts, all corrected jointly via Benjamini–Hochberg FDR.

Outputs (written to OUTPUT_DIR)
--------------------------------
me_results.csv              — subreddit vs. reference results (identical schema
                              to mixed_results.csv from run_arcticshift_regressions.py)
me_pairwise_contrasts.csv   — all pairwise contrast results
me_beta1.csv                — β₁ (log_freq_month) and control coefficients per metric

Author: Nicholas Vickery, Princeton ORFE '26
"""

import itertools
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM

from regressions import (
    LEXICAL_METRICS,
    CONTROLS,
    INVERSE_METRICS,
    _bh_adjust,
    _require_columns,
)

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:.6f}".format)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CSV_PATH = "/scratch/network/nv9344/Thesis/Thesis-Data/ArcticShift/lexical_df_combined.csv"
OUTPUT_DIR = "/scratch/network/nv9344/Thesis/Visualizations"

METRICS       = LEXICAL_METRICS
ALPHA         = 0.05
APPLY_BH      = True

FREQ_COL      = "log_freq_month"
SPEAKER_COL   = "speaker_id"
SUBREDDIT_COL = "subreddit"
TIME_COL      = "year_month"
MIN_OBS       = 50

COLS_TO_LOAD = [
    "utterance_id", SPEAKER_COL, SUBREDDIT_COL,
    "timestamp", TIME_COL,
    "num_utterances_by_speaker", "num_utterances_by_speaker_month",
    FREQ_COL,
    "post_depth", "score", "num_direct_replies", "controversiality", "edited",
    "mtld_score", "mattr_score", "yules_k", "zipf_score", "aoa_score", "nawl_ratio",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    ts = time.strftime("%H:%M:%S")
    line = "=" * 72
    print(f"\n{line}", flush=True)
    print(f"  [{ts}]  {title}", flush=True)
    print(line, flush=True)


def _elapsed(t0: float) -> str:
    secs = time.time() - t0
    mins, secs = divmod(int(secs), 60)
    return f"{mins}m {secs:02d}s"


def _save_csv(df: pd.DataFrame, name: str) -> None:
    path = os.path.join(OUTPUT_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"  → saved {path}", flush=True)


def _null_ref_row(metric, sub, ref_sub, n_obs, n_users, conclusion):
    """Placeholder row for reference comparisons when fitting fails."""
    return dict(
        metric=metric, subreddit=sub, reference_subreddit=ref_sub,
        n_obs=n_obs, n_users=n_users,
        beta_1=np.nan, se_beta_1=np.nan,
        delta=np.nan, se_delta=np.nan,
        z_stat=np.nan, p_value=np.nan,
        ci_lower=np.nan, ci_upper=np.nan,
        log_likelihood=np.nan,
        significant=np.nan, conclusion=conclusion,
    )


def _null_pair_row(metric, s_a, s_b, ref_sub, n_obs, conclusion):
    """Placeholder row for pairwise contrasts when fitting fails."""
    return dict(
        metric=metric, subreddit_a=s_a, subreddit_b=s_b,
        reference_subreddit=ref_sub, n_obs=n_obs,
        delta_a=np.nan, delta_b=np.nan,
        contrast=np.nan, se_contrast=np.nan,
        z_stat=np.nan, p_value=np.nan,
        significant=np.nan, conclusion=conclusion,
    )


def _ref_conclusion(row: pd.Series) -> str:
    """Quality-direction conclusion for a subreddit vs. reference comparison."""
    if pd.isna(row.get("significant")):
        return row.get("conclusion", "") or "insufficient data"
    if not row["significant"]:
        return "no significant difference"
    inverse = row["metric"] in INVERSE_METRICS
    higher_quality = (row["delta"] > 0) ^ inverse   # XOR flips sense for inverse metrics
    direction = "higher" if higher_quality else "lower"
    return f"{direction} quality than {row['reference_subreddit']}"


def _contrast_conclusion(row: pd.Series) -> str:
    """Quality-direction conclusion for a pairwise contrast (subreddit_a vs subreddit_b)."""
    if pd.isna(row.get("significant")):
        return row.get("conclusion", "") or "insufficient data"
    if not row["significant"]:
        return "no significant difference"
    inverse = row["metric"] in INVERSE_METRICS
    # contrast = δ_a − δ_b; positive means subreddit_a has a higher raw dummy coef
    higher_quality_a = (row["contrast"] > 0) ^ inverse
    winner = row["subreddit_a"] if higher_quality_a else row["subreddit_b"]
    loser  = row["subreddit_b"] if higher_quality_a else row["subreddit_a"]
    return f"{winner} higher quality than {loser}"


# ---------------------------------------------------------------------------
# Core fitting function
# ---------------------------------------------------------------------------

def run_me_with_contrasts(
    df: pd.DataFrame,
    metrics=None,
    controls=None,
    alpha: float = ALPHA,
    apply_bh: bool = APPLY_BH,
):
    """Fit the ME model and return (ref_df, pairwise_df, beta1_df).

    The model is identical to run_mixed_effects() in regressions.py:

        y_ust = Σ_s θ_s · 1[subreddit=s] + β₁·F_ut + β₂·X_ust + a_u + γ_t + ε_ust

    After fitting, the full covariance matrix res.cov_params() is preserved
    and used to compute SE for all pairwise contrasts between non-reference
    subreddits via the delta method:

        SE(δ_s − δ_{s'}) = sqrt( Var(δ_s) + Var(δ_{s'}) − 2·Cov(δ_s, δ_{s'}) )

    Parameters
    ----------
    df        : utterance-level DataFrame (see COLS_TO_LOAD).
    metrics   : metric columns to model; defaults to LEXICAL_METRICS.
    controls  : control columns (β₂·X_ust); defaults to CONTROLS.
    alpha     : significance level (default 0.05).
    apply_bh  : apply BH FDR correction within each output table.

    Returns
    -------
    ref_df      : pd.DataFrame — non-reference subreddits vs. r/Parenting.
    pairwise_df : pd.DataFrame — all pairwise contrasts between non-reference subs.
    beta1_df    : pd.DataFrame — β₁ (log_freq_month) per metric.
    """
    if metrics is None:
        metrics = LEXICAL_METRICS
    if controls is None:
        controls = CONTROLS

    avail_controls = [c for c in controls if c in df.columns]
    _require_columns(
        df,
        [SPEAKER_COL, SUBREDDIT_COL, TIME_COL, FREQ_COL] + list(metrics),
        "run_me_with_contrasts",
    )

    subreddits_all = sorted(df[SUBREDDIT_COL].dropna().unique())
    reference_sub  = subreddits_all[0]
    non_ref_subs   = subreddits_all[1:]
    pairs          = list(itertools.combinations(non_ref_subs, 2))

    print(f"  Reference subreddit : {reference_sub}", flush=True)
    print(f"  Comparison subs     : {non_ref_subs}", flush=True)
    print(f"  Pairwise pairs      : {pairs}", flush=True)

    ref_records      = []
    pairwise_records = []
    beta1_records    = []

    for metric in metrics:
        print(f"\n  Fitting: {metric} ...", flush=True)
        cols = [SPEAKER_COL, SUBREDDIT_COL, TIME_COL, FREQ_COL, metric] + avail_controls
        mdf = df[cols].dropna().copy().reset_index(drop=True)
        n_obs   = len(mdf)
        n_users = mdf[SPEAKER_COL].nunique()
        print(f"    n_obs={n_obs:,}  n_users={n_users:,}", flush=True)

        if n_obs < MIN_OBS:
            print(f"    Skipping: insufficient data", flush=True)
            for sub in non_ref_subs:
                ref_records.append(_null_ref_row(metric, sub, reference_sub, n_obs, n_users, "insufficient data"))
            for s_a, s_b in pairs:
                pairwise_records.append(_null_pair_row(metric, s_a, s_b, reference_sub, n_obs, "insufficient data"))
            continue

        # -- Subreddit dummies (θ_s): drop reference category --
        sub_dummies = pd.get_dummies(mdf[SUBREDDIT_COL], prefix="sub", drop_first=False).astype(float)
        ref_col = f"sub_{reference_sub}"
        if ref_col in sub_dummies.columns:
            sub_dummies = sub_dummies.drop(columns=[ref_col])
        sub_dummy_cols = sub_dummies.columns.tolist()

        # -- Time dummies (γ_t): drop first period to avoid dummy trap --
        time_dummies = pd.get_dummies(mdf[TIME_COL], prefix="time", drop_first=True).astype(float)
        time_dummy_cols = time_dummies.columns.tolist()

        mdf = pd.concat([mdf, sub_dummies, time_dummies], axis=1)

        # exog order: subreddit dummies | freq (β₁·F_ut) | controls (β₂·X_ust) | time dummies (γ_t)
        exog_cols = sub_dummy_cols + [FREQ_COL] + avail_controls + time_dummy_cols
        exog = sm.add_constant(mdf[exog_cols], has_constant="add")

        # -- Fit via REML (fallback to NM if L-BFGS fails) --
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = MixedLM(
                endog=mdf[metric].values.astype(float),
                exog=exog,
                groups=mdf[SPEAKER_COL].values,
            )
            try:
                res = mod.fit(reml=True, method="lbfgs")
                print(f"    Converged (REML/L-BFGS)", flush=True)
            except Exception:
                try:
                    res = mod.fit(reml=False, method="nm")
                    print(f"    Converged (ML/NM fallback)", flush=True)
                except Exception as e:
                    print(f"    Convergence failure: {e}", flush=True)
                    for sub in non_ref_subs:
                        ref_records.append(_null_ref_row(metric, sub, reference_sub, n_obs, n_users, "convergence failure"))
                    for s_a, s_b in pairs:
                        pairwise_records.append(_null_pair_row(metric, s_a, s_b, reference_sub, n_obs, "convergence failure"))
                    continue

        # -- Preserve full covariance matrix --
        cov     = res.cov_params()   # pd.DataFrame indexed by parameter names
        log_lik = round(float(res.llf), 4) if hasattr(res, "llf") else np.nan
        ci      = res.conf_int(alpha=alpha)

        # -- β₁: log_freq_month coefficient (same for all subreddit rows) --
        if FREQ_COL in res.params.index:
            beta_1    = round(float(res.params[FREQ_COL]), 6)
            se_beta_1 = round(float(res.bse[FREQ_COL]), 6)
            z_beta1   = round(float(res.tvalues[FREQ_COL]), 4)
            p_beta1   = round(float(res.pvalues[FREQ_COL]), 6)
        else:
            beta_1 = se_beta_1 = z_beta1 = p_beta1 = np.nan

        # -- β₂ control coefficients --
        ctrl_beta = {
            f"beta2_{c}": round(float(res.params[c]), 6) if c in res.params.index else np.nan
            for c in avail_controls
        }
        ctrl_se = {
            f"se_beta2_{c}": round(float(res.bse[c]), 6) if c in res.bse.index else np.nan
            for c in avail_controls
        }

        beta1_records.append(dict(
            metric=metric, n_obs=n_obs, n_users=n_users,
            beta_1=beta_1, se_beta_1=se_beta_1,
            z_beta1=z_beta1, p_beta1=p_beta1,
            log_likelihood=log_lik,
            **ctrl_beta, **ctrl_se,
        ))

        # -- Reference comparisons: δ_s vs. reference subreddit --
        for sub in non_ref_subs:
            param = f"sub_{sub}"
            if param not in res.params.index:
                ref_records.append(_null_ref_row(metric, sub, reference_sub, n_obs, n_users, "not estimated"))
                continue
            ref_records.append(dict(
                metric=metric, subreddit=sub,
                reference_subreddit=reference_sub,
                n_obs=n_obs, n_users=n_users,
                beta_1=beta_1, se_beta_1=se_beta_1,
                delta=round(float(res.params[param]), 6),
                se_delta=round(float(res.bse[param]), 6),
                z_stat=round(float(res.tvalues[param]), 4),
                p_value=round(float(res.pvalues[param]), 6),
                ci_lower=round(float(ci.loc[param, 0]), 6),
                ci_upper=round(float(ci.loc[param, 1]), 6),
                log_likelihood=log_lik,
                significant=np.nan, conclusion="",
                **ctrl_beta, **ctrl_se,
            ))

        # -- Pairwise contrasts between non-reference subreddits --
        for s_a, s_b in pairs:
            param_a = f"sub_{s_a}"
            param_b = f"sub_{s_b}"

            if param_a not in cov.index or param_b not in cov.index:
                pairwise_records.append(_null_pair_row(metric, s_a, s_b, reference_sub, n_obs, "not estimated"))
                continue

            delta_a  = float(res.params[param_a])
            delta_b  = float(res.params[param_b])
            contrast = delta_a - delta_b

            # Delta method: SE(δ_a − δ_b) = sqrt(Var_a + Var_b − 2·Cov_ab)
            var_a       = float(cov.loc[param_a, param_a])
            var_b       = float(cov.loc[param_b, param_b])
            cov_ab      = float(cov.loc[param_a, param_b])
            se_contrast = np.sqrt(max(var_a + var_b - 2 * cov_ab, 0.0))

            z = contrast / se_contrast if se_contrast > 0 else np.nan
            p = float(2 * stats.norm.sf(abs(z))) if not np.isnan(z) else np.nan

            pairwise_records.append(dict(
                metric=metric,
                subreddit_a=s_a, subreddit_b=s_b,
                reference_subreddit=reference_sub,
                n_obs=n_obs,
                delta_a=round(delta_a, 6),
                delta_b=round(delta_b, 6),
                contrast=round(contrast, 6),
                se_contrast=round(se_contrast, 6),
                z_stat=round(z, 4) if not np.isnan(z) else np.nan,
                p_value=round(p, 6) if not np.isnan(p) else np.nan,
                significant=np.nan, conclusion="",
            ))

    # -------------------------------------------------------------------------
    # Assemble DataFrames and apply BH correction
    # -------------------------------------------------------------------------
    ref_df      = pd.DataFrame(ref_records)
    pairwise_df = pd.DataFrame(pairwise_records)
    beta1_df    = pd.DataFrame(beta1_records)

    # BH correction — reference comparisons (same 18 tests as original script)
    if not ref_df.empty:
        ref_df["significant"] = ref_df["significant"].astype(object)
        valid = ref_df["p_value"].notna()
        if apply_bh and valid.any():
            adj = _bh_adjust(ref_df.loc[valid, "p_value"].values)
            ref_df.loc[valid, "p_value_bh"] = np.round(adj, 6)
            ref_df.loc[valid, "significant"] = ref_df.loc[valid, "p_value_bh"] < alpha
        else:
            ref_df.loc[valid, "significant"] = ref_df.loc[valid, "p_value"] < alpha
        ref_df["conclusion"] = ref_df.apply(_ref_conclusion, axis=1)

    # BH correction — pairwise contrasts (18 tests: 3 pairs × 6 metrics)
    if not pairwise_df.empty:
        pairwise_df["significant"] = pairwise_df["significant"].astype(object)
        valid = pairwise_df["p_value"].notna()
        if apply_bh and valid.any():
            adj = _bh_adjust(pairwise_df.loc[valid, "p_value"].values)
            pairwise_df.loc[valid, "p_value_bh"] = np.round(adj, 6)
            pairwise_df.loc[valid, "significant"] = pairwise_df.loc[valid, "p_value_bh"] < alpha
        else:
            pairwise_df.loc[valid, "significant"] = pairwise_df.loc[valid, "p_value"] < alpha
        pairwise_df["conclusion"] = pairwise_df.apply(_contrast_conclusion, axis=1)

    return ref_df, pairwise_df, beta1_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_banner("Loading data")
t0_total = time.time()
t0 = time.time()

df = pd.read_csv(CSV_PATH, usecols=COLS_TO_LOAD, low_memory=False)
print(f"  Read CSV in {_elapsed(t0)}", flush=True)
print(f"  {len(df):,} utterances across {df[SUBREDDIT_COL].nunique()} subreddits", flush=True)
print(df.groupby(SUBREDDIT_COL)[TIME_COL].nunique().rename("n_months").to_frame().T.to_string(), flush=True)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Fit ME model and compute contrasts
# ---------------------------------------------------------------------------

_banner("Cross-Subreddit Mixed-Effects + Pairwise Contrasts")
t0 = time.time()

ref_results, pairwise_results, beta1_results = run_me_with_contrasts(df)

print(f"\n  Fitting complete in {_elapsed(t0)}", flush=True)


# ---------------------------------------------------------------------------
# Results: subreddit vs. reference (r/Parenting)
# ---------------------------------------------------------------------------

_banner("Results: each subreddit vs. reference (r/Parenting)")

ref_sub = ref_results["reference_subreddit"].iloc[0] if len(ref_results) else "N/A"
print(f"Reference subreddit: {ref_sub}\n", flush=True)

display_cols_ref = [
    "metric", "subreddit", "beta_1", "se_beta_1",
    "delta", "se_delta", "z_stat", "p_value", "p_value_bh",
    "significant", "conclusion",
]
print(
    ref_results[[c for c in display_cols_ref if c in ref_results.columns]]
    .sort_values(["metric", "subreddit"])
    .to_string(index=False),
    flush=True,
)
_save_csv(ref_results, "me_results")


# ---------------------------------------------------------------------------
# β₁ summary
# ---------------------------------------------------------------------------

_banner("β₁ (log post frequency) and control coefficients")

display_cols_b1 = ["metric", "n_obs", "n_users", "beta_1", "se_beta_1", "z_beta1", "p_beta1"]
ctrl_cols = [c for c in beta1_results.columns if c.startswith("beta2_") or c.startswith("se_beta2_")]
print(
    beta1_results[[c for c in display_cols_b1 + ctrl_cols if c in beta1_results.columns]]
    .to_string(index=False),
    flush=True,
)
_save_csv(beta1_results, "me_beta1")


# ---------------------------------------------------------------------------
# Pairwise contrasts between non-reference subreddits
# ---------------------------------------------------------------------------

_banner("Pairwise contrasts between non-reference subreddits")

n_pairs = pairwise_results["p_value"].notna().sum()
n_sig   = (pairwise_results["significant"] == True).sum()
print(
    f"Total contrasts: {len(pairwise_results)}  |  "
    f"Testable: {n_pairs}  |  Significant after BH: {n_sig}\n",
    flush=True,
)

display_cols_pair = [
    "metric", "subreddit_a", "subreddit_b",
    "delta_a", "delta_b", "contrast", "se_contrast",
    "z_stat", "p_value", "p_value_bh", "significant", "conclusion",
]
print(
    pairwise_results[[c for c in display_cols_pair if c in pairwise_results.columns]]
    .sort_values(["metric", "subreddit_a", "subreddit_b"])
    .to_string(index=False),
    flush=True,
)
_save_csv(pairwise_results, "me_pairwise_contrasts")


# ---------------------------------------------------------------------------
# Significant pairwise contrasts only
# ---------------------------------------------------------------------------

_banner("Significant pairwise contrasts (BH-corrected)")

sig_pairs = pairwise_results[pairwise_results["significant"] == True].copy()
if len(sig_pairs):
    print(
        sig_pairs[[c for c in display_cols_pair if c in sig_pairs.columns]]
        .sort_values(["metric", "subreddit_a", "subreddit_b"])
        .to_string(index=False),
        flush=True,
    )
else:
    print("  No significant pairwise contrasts after BH correction.", flush=True)


_banner(f"Done — total elapsed: {_elapsed(t0_total)}")
sys.exit(0)
