"""
run_arcticshift_regressions.py
==============================
Script version of arcticshift.ipynb.

Runs all six regression models from Chapter 4 of the methodology on the
combined ArcticShift lexical CSV produced by combine_lexical_csvs.py.
Results are printed to stdout and saved as CSVs under OUTPUT_DIR.
Designed to be run on Adroit via run_arcticshift_regressions.slurm.

Author: Nicholas Vickery, Princeton ORFE '26
"""

import os
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — required on headless cluster nodes
import matplotlib.pyplot as plt
import pandas as pd

from regressions import (
    run_baseline_ols,
    run_first_diff_ols,
    run_ar_ols,
    run_cross_user_wls,
    run_fixed_effects_panel,
    run_mixed_effects,
    LEXICAL_METRICS,
)
from trend_analysis import plot_ols_trend_grid

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:.6f}".format)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CSV_PATH = "/scratch/network/nv9344/Thesis/Thesis-Data/ArcticShift/lexical_df_combined.csv"

# All figures and result CSVs are written here.
OUTPUT_DIR = "/scratch/network/nv9344/Thesis/Visualizations"

METRICS  = LEXICAL_METRICS   # all six, or pass a subset to any function below
ALPHA    = 0.05              # significance level
APPLY_BH = True              # Benjamini-Hochberg FDR correction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    """Print a clearly visible section header with a timestamp."""
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


def _print_ctrl_coefs(results: pd.DataFrame, model_label: str) -> None:
    """Print β₂ control coefficients from WLS, FE, or mixed-effects results."""
    ctrl_names = ["post_depth", "edited", "score", "num_direct_replies", "controversiality"]
    b2_cols  = [f"beta2_{c}"    for c in ctrl_names if f"beta2_{c}"    in results.columns]
    se_cols  = [f"se_beta2_{c}" for c in ctrl_names if f"se_beta2_{c}" in results.columns]
    avail    = [c for c in ctrl_names if f"beta2_{c}" in results.columns]

    if not b2_cols:
        print("β₂ columns not found — re-run after updating regressions.py.", flush=True)
        return

    print(f"\n=== {model_label} — β₂ Control Coefficients ===\n", flush=True)

    # Deduplicate if the β₂ is the same across subreddit rows (FE, mixed-effects)
    dedup_col = "metric" if "subreddit" in results.columns else None
    display_df = (
        results[["metric"] + b2_cols + se_cols]
        .drop_duplicates(subset="metric")
        .rename(columns={f"beta2_{c}": c for c in avail}
                        | {f"se_beta2_{c}": f"se_{c}" for c in avail})
        .reset_index(drop=True)
    ) if dedup_col == "metric" else (
        results[["subreddit", "metric"] + b2_cols + se_cols]
        .rename(columns={f"beta2_{c}": c for c in avail}
                        | {f"se_beta2_{c}": f"se_{c}" for c in avail})
        .sort_values(["metric", "subreddit"])
    )
    print(display_df.to_string(index=False), flush=True)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

_banner("Loading data")
t0_total = time.time()

COLS_TO_LOAD = [
    "utterance_id", "speaker_id", "subreddit",
    "timestamp", "year_month",
    "num_utterances_by_speaker", "num_utterances_by_speaker_month",
    "log_freq_month",
    "post_depth", "score", "num_direct_replies", "controversiality", "edited",
    "mtld_score", "mattr_score", "yules_k", "zipf_score", "aoa_score", "nawl_ratio",
]

t0 = time.time()
df = pd.read_csv(CSV_PATH, usecols=COLS_TO_LOAD, low_memory=False)
print(f"  Read CSV in {_elapsed(t0)}", flush=True)
print(f"  Loaded {len(df):,} utterances across {df['subreddit'].nunique()} subreddit(s).", flush=True)
print(df[["subreddit", "year_month"]].groupby("subreddit")["year_month"].nunique()
        .rename("n_months").to_frame().T.to_string(), flush=True)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Baseline OLS Regression
# ---------------------------------------------------------------------------
# Model: y_t = β₀ + β₁·t + ε_t
# One regression per (subreddit × metric), fitted on monthly-aggregated data
# with Newey-West HAC standard errors. β₁ is the estimated change per month.

_banner("Baseline OLS Regression (§4.4.6)")
t0 = time.time()

ols_results = run_baseline_ols(
    df,
    metrics=METRICS,
    alpha=ALPHA,
    apply_bh=APPLY_BH,
)

print("\n=== Baseline OLS — conclusion counts ===", flush=True)
print(ols_results["conclusion"].value_counts().to_string(), flush=True)
print(flush=True)

sig = ols_results[ols_results["significant"] == True].copy()
print(f"Significant (subreddit × metric) pairs: {len(sig)} / {len(ols_results)}", flush=True)
if len(sig):
    print(sig[["subreddit", "metric", "beta_1", "se_beta_1", "p_value", "p_value_bh", "conclusion"]]
          .sort_values(["metric", "subreddit"])
          .to_string(index=False), flush=True)

print(f"\n  Completed in {_elapsed(t0)}", flush=True)
_save_csv(ols_results, "ols_results")

# OLS trend grid plots (diversity + sophistication)
agg = df.groupby(["subreddit", "year_month"])[METRICS].mean().reset_index()
plot_ols_trend_grid(
    agg,
    ols_results,
    save_path=os.path.join(OUTPUT_DIR, "ols_trend_grid.png"),
)


# ---------------------------------------------------------------------------
# 2. First-Differenced OLS Regression
# ---------------------------------------------------------------------------
# Model: Δy_t = β₀ + ε_t
# Drift model on first-differenced monthly series with HAC standard errors.
# A significant β₀ (drift) with the same sign as β₁ from baseline OLS
# provides convergent evidence of a genuine trend.

_banner("First-Differenced OLS Regression (§4.4.7)")
t0 = time.time()

fd_results = run_first_diff_ols(
    df,
    metrics=METRICS,
    alpha=ALPHA,
    apply_bh=APPLY_BH,
)

print("\n=== First-Differenced OLS — conclusion counts ===", flush=True)
print(fd_results["conclusion"].value_counts().to_string(), flush=True)
print(flush=True)

sig_fd = fd_results[fd_results["significant"] == True].copy()
print(f"Significant pairs: {len(sig_fd)} / {len(fd_results)}", flush=True)
if len(sig_fd):
    print(sig_fd[["subreddit", "metric", "drift", "se_drift", "p_value", "p_value_bh", "conclusion"]]
          .sort_values(["metric", "subreddit"])
          .to_string(index=False), flush=True)

# Convergence check: compare signs with baseline OLS
merged = ols_results[["subreddit", "metric", "beta_1", "significant"]].rename(
    columns={"beta_1": "ols_beta1", "significant": "ols_sig"}
).merge(
    fd_results[["subreddit", "metric", "drift", "significant"]].rename(
        columns={"drift": "fd_drift", "significant": "fd_sig"}
    ),
    on=["subreddit", "metric"],
)
merged["signs_agree"] = (
    merged["ols_beta1"].apply(lambda x: 1 if x > 0 else -1) ==
    merged["fd_drift"].apply(lambda x: 1 if x > 0 else -1)
)
print("\n=== OLS ↔ First-Diff sign convergence ===", flush=True)
print(merged[["subreddit", "metric", "ols_beta1", "ols_sig", "fd_drift", "fd_sig", "signs_agree"]]
      .sort_values(["metric", "subreddit"])
      .to_string(index=False), flush=True)

print(f"\n  Completed in {_elapsed(t0)}", flush=True)
_save_csv(fd_results, "fd_results")


# ---------------------------------------------------------------------------
# 3. AR OLS Regression
# ---------------------------------------------------------------------------
# Model: y_t = β₀ + β₁·t + φ·y_{t-1} + ε_t
# Adds an AR(1) term to absorb autocorrelation, with HAC standard errors.
# β₁ captures the trend net of persistence; φ captures the AR pull.

_banner("AR OLS Regression (§4.4.8)")
t0 = time.time()

ar_results = run_ar_ols(
    df,
    metrics=METRICS,
    alpha=ALPHA,
    apply_bh=APPLY_BH,
)

print("\n=== AR OLS — conclusion counts ===", flush=True)
print(ar_results["conclusion"].value_counts().to_string(), flush=True)
print(flush=True)

sig_ar = ar_results[ar_results["significant"] == True].copy()
print(f"Significant pairs: {len(sig_ar)} / {len(ar_results)}", flush=True)
if len(sig_ar):
    print(sig_ar[["subreddit", "metric", "beta_1", "se_beta_1", "phi",
                  "p_value_beta1", "p_value_bh", "conclusion"]]
          .sort_values(["metric", "subreddit"])
          .to_string(index=False), flush=True)

# Three-model sign convergence
convergence = merged.merge(
    ar_results[["subreddit", "metric", "beta_1", "significant"]].rename(
        columns={"beta_1": "ar_beta1", "significant": "ar_sig"}
    ),
    on=["subreddit", "metric"],
)
convergence["all_agree"] = (
    convergence["signs_agree"] &
    (convergence["ols_beta1"].apply(lambda x: 1 if x > 0 else -1) ==
     convergence["ar_beta1"].apply(lambda x: 1 if x > 0 else -1))
)
print("\n=== Three-model sign convergence (OLS, FD, AR) ===", flush=True)
print(convergence[["subreddit", "metric", "ols_sig", "fd_sig", "ar_sig", "all_agree"]]
      .sort_values(["all_agree", "metric", "subreddit"], ascending=[False, True, True])
      .to_string(index=False), flush=True)

print(f"\n  Completed in {_elapsed(t0)}", flush=True)
_save_csv(ar_results, "ar_results")


# ---------------------------------------------------------------------------
# 4. Cross-User WLS Regression
# ---------------------------------------------------------------------------
# Model: ȳ_u = β₀ + β₁·F̄_u + β₂·X̄_u + ε_u
# User-level means, weighted by total post count n_u. β₁ captures the
# relationship between posting frequency and lexical quality across users.

_banner("Cross-User WLS Regression (§4.5)")
t0 = time.time()

wls_results = run_cross_user_wls(
    df,
    metrics=METRICS,
    alpha=ALPHA,
    apply_bh=APPLY_BH,
)

print("\n=== Cross-User WLS — conclusion counts ===", flush=True)
print(wls_results["conclusion"].value_counts().to_string(), flush=True)
print(flush=True)

sig_wls = wls_results[wls_results["significant"] == True].copy()
print(f"Significant pairs: {len(sig_wls)} / {len(wls_results)}", flush=True)
if len(sig_wls):
    print(sig_wls[["subreddit", "metric", "n_users", "beta_1", "se_beta_1",
                   "p_value", "p_value_bh", "conclusion"]]
          .sort_values(["metric", "subreddit"])
          .to_string(index=False), flush=True)

_print_ctrl_coefs(wls_results, "Cross-User WLS")

print(f"\n  Completed in {_elapsed(t0)}", flush=True)
_save_csv(wls_results, "wls_results")


# ---------------------------------------------------------------------------
# 5. Fixed Effects Panel Regression
# ---------------------------------------------------------------------------
# Model: y_ust = β₁·F_ut + β₂·X_ust + α_u + γ_t + δ_s + ε_ust
# User (α_u), time (γ_t), and subreddit (δ_s) fixed effects across all
# communities jointly. β₁ is the within-user effect of log monthly posting
# frequency on lexical quality.

_banner("Fixed Effects Panel Regression (§4.6)")
t0 = time.time()

fe_results = run_fixed_effects_panel(
    df,
    metrics=METRICS,
    alpha=ALPHA,
    apply_bh=APPLY_BH,
)

print("\n=== Fixed Effects Panel — conclusion counts ===", flush=True)
print(fe_results["conclusion"].value_counts().to_string(), flush=True)
print(flush=True)

sig_fe = fe_results[fe_results["significant"] == True].copy()
print(f"Significant metrics: {len(sig_fe)} / {len(fe_results)}", flush=True)
if len(sig_fe):
    print(sig_fe[["metric", "n_obs", "n_users", "n_periods",
                  "beta_1", "se_beta_1", "p_value", "p_value_bh", "conclusion"]]
          .to_string(index=False), flush=True)

_print_ctrl_coefs(fe_results, "Fixed Effects Panel")

print(f"\n  Completed in {_elapsed(t0)}", flush=True)
_save_csv(fe_results, "fe_results")


# ---------------------------------------------------------------------------
# 6. Cross-Subreddit Mixed-Effects Model
# ---------------------------------------------------------------------------
# Model: y_ust = Σ_s θ_s·1[subreddit=s] + β₁·F_ut + β₂·X_ust + a_u + γ_t + ε_ust
# Subreddit fixed effects (θ_s) with user random intercepts (a_u), a frequency
# term (β₁·F_ut), time fixed effects (γ_t), and post-level controls (β₂·X_ust),
# fitted via REML. θ_s is the conditional mean difference in lexical quality
# for subreddit s relative to the reference (first alphabetically).

_banner("Cross-Subreddit Mixed-Effects Model (§4.7)")
t0 = time.time()

mixed_results = run_mixed_effects(
    df,
    metrics=METRICS,
    alpha=ALPHA,
    apply_bh=APPLY_BH,
)

print("\n=== Mixed Effects — conclusion counts ===", flush=True)
print(mixed_results["conclusion"].value_counts().to_string(), flush=True)
print(flush=True)

ref = mixed_results["reference_subreddit"].iloc[0] if len(mixed_results) else "N/A"
print(f"Reference subreddit: {ref}", flush=True)
print(flush=True)

sig_mixed = mixed_results[mixed_results["significant"] == True].copy()
print(f"Significant (metric × subreddit) pairs: {len(sig_mixed)} / {len(mixed_results)}", flush=True)
if len(sig_mixed):
    print(sig_mixed[["metric", "subreddit", "reference_subreddit",
                     "delta", "se_delta", "p_value", "p_value_bh", "conclusion"]]
          .sort_values(["metric", "subreddit"])
          .to_string(index=False), flush=True)

_print_ctrl_coefs(mixed_results, "Mixed-Effects")

print(f"\n  Completed in {_elapsed(t0)}", flush=True)
_save_csv(mixed_results, "mixed_results")


# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

_banner(f"All models complete — total elapsed: {_elapsed(t0_total)}")
sys.exit(0)
