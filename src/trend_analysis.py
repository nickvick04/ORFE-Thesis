# ----------------------------------------------------------------------------------------
# Trend Analysis Pipeline: utterance-level lexical_master.csv → speaker-month panel
#
# Steps
# -----
# 1. Load lexical_master.csv (raw_text excluded), parse timestamps, create t_month
# 2. Filter invalid observations (rows missing all metric values)
# 3. Aggregate to (speaker_id × subreddit × year_month) panel
# 4. Filter sparse speakers (min_obs monthly observations required)
# 5. Pooled OLS: quality ~ t_month, no fixed effects, SEs clustered by speaker
# 6. Panel FE OLS: quality ~ t_month + speaker fixed effects, SEs clustered by speaker
# 7. Per-subreddit panel FE: repeat step 6 within each subreddit
#
# Outputs (all written to --output-dir)
# ----------------------------------------
# results_pooled_ols.csv        — beta, SE, t-stat, p-value per metric
# results_panel_fe.csv          — same, with speaker fixed effects
# results_per_subreddit_fe.csv  — per-subreddit FE results (metric × subreddit)
# beta_comparison.csv           — pooled vs. FE betas side by side (decomposition table)
# acf_diagnostics/              — ACF plots of monthly mean residuals per metric
#
# Code Author: Nicholas Vickery, Princeton ORFE '26
# ----------------------------------------------------------------------------------------

import argparse
import gc
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')   # non-interactive backend for Adroit (no display)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf

# linearmodels is required for the panel FE estimator
# Install with: pip install linearmodels --user
try:
    from linearmodels import PanelOLS
    HAS_LINEARMODELS = True
except ImportError:
    HAS_LINEARMODELS = False
    warnings.warn(
        "linearmodels is not installed. Steps 6 and 7 (panel FE) will be skipped.\n"
        "Install with: pip install linearmodels --user",
        ImportWarning,
        stacklevel=2,
    )

# ----------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------

LEXICAL_METRICS = [
    'mtld_score',
    'mattr_score',
    'yules_k',
    'zipf_score',
    'aoa_score',
    'nawl_ratio',
]

METRIC_LABELS = {
    'mtld_score':  'MTLD',
    'mattr_score': 'MATTR',
    'yules_k':     "Yule's K",
    'zipf_score':  'Zipf Score',
    'aoa_score':   'Age of Acquisition',
    'nawl_ratio':  'NAWL Ratio',
}

# Columns needed from lexical_master.csv — raw_text intentionally excluded
COLS_TO_LOAD = [
    'speaker_id',
    'timestamp',
    'num_utterances_by_speaker_month',
    'mtld_score',
    'mattr_score',
    'yules_k',
    'zipf_score',
    'aoa_score',
    'nawl_ratio',
    'source_variation',
    'subreddit',
]

# Minimum number of monthly observations a speaker must have to be included
DEFAULT_MIN_OBS = 3

# Minimum number of unique speakers a subreddit must have before running a
# per-subreddit FE regression (guards against degenerate models)
DEFAULT_MIN_SPEAKERS_PER_SUB = 50

# Drop utterance row if fewer than this many metric columns are non-NaN.
# Rows below this threshold correspond to utterances too short or malformed
# for the lexical pipeline to score reliably.
MIN_METRICS_NONNAN = 2

# ----------------------------------------------------------------------------------------
# Step 1 — Load and prepare
# ----------------------------------------------------------------------------------------

def load_and_prepare(input_path: str) -> pd.DataFrame:
    """Load lexical_master.csv, parse timestamps, and create a numeric time index.

    Returns a DataFrame with an integer 't_month' column representing months
    elapsed since the earliest timestamp in the dataset (t_month = 0 at the
    first observed calendar month). This is the regressor used in all OLS models.

    raw_text is not loaded; only metric and metadata columns are read.
    """
    print(f"[Step 1] Loading data from: {input_path}")
    df = pd.read_csv(input_path, usecols=COLS_TO_LOAD, low_memory=False)
    print(f"         Loaded {len(df):,} utterances.")

    # Parse timestamp and drop rows with unparseable dates
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    n_bad_ts = df['timestamp'].isna().sum()
    if n_bad_ts > 0:
        print(f"         Dropping {n_bad_ts:,} rows with unparseable timestamps.")
    df = df.dropna(subset=['timestamp'])

    # year_month as Period — used as the groupby key in aggregation
    df['year_month'] = df['timestamp'].dt.to_period('M')

    # t_month: integer months since the earliest record — the OLS regressor.
    # Multiplying beta by 12 gives an annualised effect size.
    month_int = df['timestamp'].dt.year * 12 + df['timestamp'].dt.month
    origin = month_int.min()
    df['t_month'] = (month_int - origin).astype(int)

    t_min  = df['timestamp'].min().strftime('%Y-%m')
    t_max  = df['timestamp'].max().strftime('%Y-%m')
    t_span = df['t_month'].max()
    print(f"         Time range : {t_min} → {t_max}  ({t_span + 1} months, "
          f"t_month 0–{t_span})")
    print(f"         Subreddits : {sorted(df['subreddit'].unique())}")

    return df


# ----------------------------------------------------------------------------------------
# Step 2 — Filter invalid observations
# ----------------------------------------------------------------------------------------

def filter_invalid(df: pd.DataFrame) -> pd.DataFrame:
    """Drop utterance rows that are unlikely to yield reliable metric estimates.

    Because raw_text is excluded, token-length filtering is not directly possible.
    Rows are instead dropped when they have fewer than MIN_METRICS_NONNAN non-NaN
    metric values — these correspond to utterances too short or malformed for the
    lexical pipeline to score.
    """
    print(f"\n[Step 2] Filtering invalid observations...")
    n_before = len(df)

    metric_nonnan = df[LEXICAL_METRICS].notna().sum(axis=1)
    df = df[metric_nonnan >= MIN_METRICS_NONNAN].copy()

    n_dropped = n_before - len(df)
    print(f"         Dropped {n_dropped:,} rows ({100 * n_dropped / n_before:.1f}%).  "
          f"Retained {len(df):,}.")
    return df


# ----------------------------------------------------------------------------------------
# Step 3 — Aggregate to (speaker × subreddit × year_month) panel
# ----------------------------------------------------------------------------------------

def aggregate_to_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate utterance-level data to a (speaker_id, subreddit, year_month) panel.

    Each row represents one speaker's mean lexical quality in one subreddit in
    one calendar month. Utterance counts per cell ('n_utt_sm') are stored for
    use as regression weights in downstream models: speaker-months backed by more
    utterances are more reliable estimates of true quality and receive higher weight.
    """
    print(f"\n[Step 3] Aggregating to (speaker × subreddit × year_month) panel...")

    agg_dict = {m: 'mean' for m in LEXICAL_METRICS}
    agg_dict['t_month']                        = 'first'
    agg_dict['num_utterances_by_speaker_month'] = 'sum'
    agg_dict['source_variation']               = 'first'

    panel = (
        df.groupby(['speaker_id', 'subreddit', 'year_month'], sort=False)
          .agg(agg_dict)
          .reset_index()
          .rename(columns={'num_utterances_by_speaker_month': 'n_utt_sm'})
    )

    print(f"         Panel rows      : {len(panel):,}")
    print(f"         Unique speakers : {panel['speaker_id'].nunique():,}")
    print(f"         Subreddits      : {panel['subreddit'].nunique()}")
    return panel


def _aggregate_cross_subreddit(panel: pd.DataFrame) -> pd.DataFrame:
    """Further aggregate the subreddit-level panel to (speaker_id, year_month).

    Used in steps 5 and 6 (pooled/FE across all subreddits). A speaker posting
    in multiple subreddits in the same month gets one row whose metric values
    are utterance-count-weighted means across those subreddits.
    """
    records = []
    for (spk, ym), grp in panel.groupby(['speaker_id', 'year_month'], sort=False):
        row = {'speaker_id': spk, 'year_month': ym,
               't_month': int(grp['t_month'].iloc[0]),
               'n_utt_sm': grp['n_utt_sm'].sum()}
        w = grp['n_utt_sm'].values.astype(float)
        for m in LEXICAL_METRICS:
            v = grp[m].values.astype(float)
            mask = ~np.isnan(v)
            if mask.sum() == 0 or w[mask].sum() == 0:
                row[m] = np.nan
            else:
                row[m] = float(np.average(v[mask], weights=w[mask]))
        records.append(row)
    return pd.DataFrame(records)


# ----------------------------------------------------------------------------------------
# Step 4 — Filter sparse speakers
# ----------------------------------------------------------------------------------------

def filter_sparse_speakers(panel: pd.DataFrame,
                            min_obs: int = DEFAULT_MIN_OBS) -> pd.DataFrame:
    """Keep only speakers with at least min_obs distinct monthly observations.

    Observation count is computed over the subreddit-level panel (a speaker who
    posts in two subreddits in the same month counts as 2 observations). This
    ensures meaningful within-speaker time variation exists for the FE estimator.
    """
    print(f"\n[Step 4] Filtering sparse speakers (min_obs = {min_obs})...")
    n_before = panel['speaker_id'].nunique()

    obs_per_speaker = panel.groupby('speaker_id').size()
    valid = obs_per_speaker[obs_per_speaker >= min_obs].index
    panel = panel[panel['speaker_id'].isin(valid)].copy()

    n_retained = panel['speaker_id'].nunique()
    print(f"         Retained {n_retained:,} / {n_before:,} speakers "
          f"({100 * n_retained / n_before:.1f}%)  |  "
          f"panel rows: {len(panel):,}")
    return panel


# ----------------------------------------------------------------------------------------
# Helpers — extract regression results into tidy dicts
# ----------------------------------------------------------------------------------------

def _extract_ols_results(fit, metric: str, model_type: str,
                          subreddit: str = 'all') -> dict:
    """Pull beta, SE, t-stat, p-value for the t_month coefficient (statsmodels)."""
    try:
        key = 't_month' if 't_month' in fit.params.index else fit.params.index[-1]
        return {
            'metric':        metric,
            'model':         model_type,
            'subreddit':     subreddit,
            'beta':          round(float(fit.params[key]),        8),
            'beta_annual':   round(float(fit.params[key]) * 12,   6),
            'se':            round(float(fit.bse[key]),            8),
            't_stat':        round(float(fit.tvalues[key]),        4),
            'p_value':       round(float(fit.pvalues[key]),        6),
            'n_obs':         int(fit.nobs),
        }
    except Exception as e:
        warnings.warn(f"Could not extract results for {metric} ({model_type}): {e}")
        return {'metric': metric, 'model': model_type, 'subreddit': subreddit,
                'beta': np.nan, 'beta_annual': np.nan, 'se': np.nan,
                't_stat': np.nan, 'p_value': np.nan, 'n_obs': np.nan}


def _extract_linearmodels_results(fit, metric: str, model_type: str,
                                   subreddit: str = 'all') -> dict:
    """Pull beta, SE, t-stat, p-value for the t_month coefficient (linearmodels)."""
    try:
        return {
            'metric':        metric,
            'model':         model_type,
            'subreddit':     subreddit,
            'beta':          round(float(fit.params['t_month']),        8),
            'beta_annual':   round(float(fit.params['t_month']) * 12,   6),
            'se':            round(float(fit.std_errors['t_month']),     8),
            't_stat':        round(float(fit.tstats['t_month']),         4),
            'p_value':       round(float(fit.pvalues['t_month']),        6),
            'n_obs':         int(fit.nobs),
        }
    except Exception as e:
        warnings.warn(f"Could not extract results for {metric} ({model_type}): {e}")
        return {'metric': metric, 'model': model_type, 'subreddit': subreddit,
                'beta': np.nan, 'beta_annual': np.nan, 'se': np.nan,
                't_stat': np.nan, 'p_value': np.nan, 'n_obs': np.nan}


# ----------------------------------------------------------------------------------------
# Step 5 — Pooled OLS (no fixed effects)
# ----------------------------------------------------------------------------------------

def run_pooled_ols(cross_panel: pd.DataFrame,
                   metrics: list = LEXICAL_METRICS) -> tuple:
    """Step 5: WLS(metric ~ t_month) with no fixed effects, SEs clustered by speaker.

    Uses the cross-subreddit aggregated panel so each (speaker, month) is one
    observation regardless of how many subreddits they posted in that month.

    Weighted by n_utt_sm so speaker-months backed by more utterances carry more
    influence. SEs are clustered by speaker to handle within-person correlation
    across months.

    Returns
    -------
    results_df : pd.DataFrame  tidy table of regression statistics per metric
    residuals  : dict          {metric: pd.Series(index=t_month)}
                               monthly mean residuals for ACF diagnostics
    """
    print(f"\n[Step 5] Running pooled OLS (no fixed effects)...")
    records   = []
    residuals = {}

    for metric in metrics:
        sub = cross_panel[['speaker_id', 't_month', metric, 'n_utt_sm']].dropna(subset=[metric])
        if len(sub) < 100:
            print(f"         Skipping {metric}: fewer than 100 valid observations.")
            continue

        y = sub[metric].values
        X = sm.add_constant(sub['t_month'].values, has_constant='add')
        w = np.where(sub['n_utt_sm'].values > 0, sub['n_utt_sm'].values, 1.0).astype(float)

        fit = sm.WLS(y, X, weights=w).fit(
            cov_type='cluster',
            cov_kwds={'groups': sub['speaker_id'].values},
        )

        records.append(_extract_ols_results(fit, metric, 'pooled_ols'))

        # Monthly mean residuals — used downstream for ACF diagnostic
        resid_s = pd.Series(fit.resid, index=sub['t_month'].values)
        residuals[metric] = resid_s.groupby(level=0).mean().sort_index()

        beta_key = 't_month' if 't_month' in fit.params.index else fit.params.index[-1]
        print(f"         {METRIC_LABELS[metric]:25s}  "
              f"beta/mo={float(fit.params[beta_key]):+.6f}  "
              f"(ann.={float(fit.params[beta_key])*12:+.5f})  "
              f"p={float(fit.pvalues[beta_key]):.4f}  n={int(fit.nobs):,}")

    return pd.DataFrame(records), residuals


# ----------------------------------------------------------------------------------------
# Step 6 — Panel FE OLS (speaker fixed effects)
# ----------------------------------------------------------------------------------------

def run_panel_fe(cross_panel: pd.DataFrame,
                 metrics: list = LEXICAL_METRICS) -> pd.DataFrame:
    """Step 6: PanelOLS(metric ~ t_month, entity_effects=True) clustered by speaker.

    Speaker fixed effects absorb all time-invariant individual characteristics
    (baseline writing quality, vocabulary size, etc.). Beta on t_month captures
    within-person change over time only — the purely behavioural component.

    Comparing pooled beta (step 5) and FE beta (step 6) decomposes the aggregate
    trend:
        beta_pooled - beta_FE  ≈  compositional effect (changing user mix)
        beta_FE               ≈  behavioural effect (same speakers writing differently)
    """
    if not HAS_LINEARMODELS:
        print("\n[Step 6] Skipped — linearmodels not installed.")
        return pd.DataFrame()

    print(f"\n[Step 6] Running panel FE OLS (speaker fixed effects)...")
    records = []

    for metric in metrics:
        sub = cross_panel[['speaker_id', 't_month', metric, 'n_utt_sm']].dropna(subset=[metric])

        # Drop speakers with only one time observation (no within variation for FE)
        counts = sub.groupby('speaker_id')['t_month'].nunique()
        valid  = counts[counts > 1].index
        sub    = sub[sub['speaker_id'].isin(valid)].copy()

        if sub['speaker_id'].nunique() < 100:
            print(f"         Skipping {metric}: fewer than 100 speakers with >= 2 obs.")
            continue

        # linearmodels requires a MultiIndex of (entity, time)
        sub = sub.set_index(['speaker_id', 't_month'])
        y = sub[[metric]]
        X = sub[['t_month']].copy()
        w = np.where(sub['n_utt_sm'].values > 0, sub['n_utt_sm'].values, 1.0).astype(float)

        try:
            model = PanelOLS(y, X, entity_effects=True, weights=w)
            fit   = model.fit(cov_type='clustered', cluster_entity=True)
            records.append(_extract_linearmodels_results(fit, metric, 'panel_fe'))

            print(f"         {METRIC_LABELS[metric]:25s}  "
                  f"beta/mo={float(fit.params['t_month']):+.6f}  "
                  f"(ann.={float(fit.params['t_month'])*12:+.5f})  "
                  f"p={float(fit.pvalues['t_month']):.4f}  n={int(fit.nobs):,}")
        except Exception as e:
            warnings.warn(f"Panel FE failed for {metric}: {e}")

    return pd.DataFrame(records)


# ----------------------------------------------------------------------------------------
# Step 7 — Per-subreddit panel FE
# ----------------------------------------------------------------------------------------

def run_per_subreddit_fe(panel: pd.DataFrame,
                          metrics: list = LEXICAL_METRICS,
                          min_speakers: int = DEFAULT_MIN_SPEAKERS_PER_SUB) -> pd.DataFrame:
    """Step 7: Panel FE regression run separately within each subreddit.

    Each subreddit yields its own beta estimate, enabling comparison of trends
    across communities and variation categories (Age / Culture / Topic).

    Each (speaker_id, year_month) pair is guaranteed to be unique within a single
    subreddit after step 3's aggregation, satisfying the PanelOLS index requirement.
    """
    if not HAS_LINEARMODELS:
        print("\n[Step 7] Skipped — linearmodels not installed.")
        return pd.DataFrame()

    print(f"\n[Step 7] Running per-subreddit panel FE OLS...")
    records    = []
    subreddits = sorted(panel['subreddit'].unique())

    for sub_name in subreddits:
        sub_df     = panel[panel['subreddit'] == sub_name].copy()
        n_speakers = sub_df['speaker_id'].nunique()

        if n_speakers < min_speakers:
            print(f"         Skipping {sub_name}: {n_speakers} speakers < {min_speakers}.")
            continue

        print(f"         {sub_name}  ({n_speakers:,} speakers)")

        for metric in metrics:
            sub_m = sub_df[['speaker_id', 't_month', metric, 'n_utt_sm']].dropna(subset=[metric])

            # Keep only speakers with > 1 observation within this subreddit
            counts = sub_m.groupby('speaker_id')['t_month'].nunique()
            valid  = counts[counts > 1].index
            sub_m  = sub_m[sub_m['speaker_id'].isin(valid)].copy()

            if sub_m['speaker_id'].nunique() < min_speakers:
                continue

            sub_m = sub_m.set_index(['speaker_id', 't_month'])
            y = sub_m[[metric]]
            X = sub_m[['t_month']]
            w = np.where(sub_m['n_utt_sm'].values > 0,
                         sub_m['n_utt_sm'].values, 1.0).astype(float)

            try:
                model = PanelOLS(y, X, entity_effects=True, weights=w)
                fit   = model.fit(cov_type='clustered', cluster_entity=True)
                records.append(
                    _extract_linearmodels_results(fit, metric, 'subreddit_fe', sub_name)
                )
            except Exception as e:
                warnings.warn(f"Subreddit FE failed for {sub_name}/{metric}: {e}")

    return pd.DataFrame(records)


# ----------------------------------------------------------------------------------------
# ACF Diagnostics
# ----------------------------------------------------------------------------------------

def plot_acf_diagnostics(residuals: dict, output_dir: Path) -> None:
    """Plot and save ACF of monthly mean OLS residuals for each metric.

    The pooled OLS residuals (speaker-month level) are aggregated to their
    calendar-month mean and the autocorrelation function is plotted. Significant
    spikes at low lags confirm that residuals are autocorrelated, validating the
    use of clustered SEs. Spikes at lag 12 indicate seasonal structure that would
    require month-of-year fixed effects.
    """
    acf_dir = output_dir / 'acf_diagnostics'
    acf_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[ACF Diagnostics] Saving plots to: {acf_dir}")

    for metric, resid_series in residuals.items():
        if resid_series.empty or len(resid_series) < 20:
            continue

        n_lags = min(40, len(resid_series) // 2 - 1)
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_acf(resid_series.values, lags=n_lags, ax=ax, alpha=0.05)
        ax.set_title(
            f'ACF of Monthly Mean Residuals — {METRIC_LABELS.get(metric, metric)}\n'
            f'(Pooled WLS residuals aggregated to calendar month)',
            fontsize=11,
        )
        ax.set_xlabel('Lag (months)')
        ax.set_ylabel('Autocorrelation')
        ax.axhline(0, color='black', linewidth=0.8)
        fig.tight_layout()
        path = acf_dir / f'acf_{metric}.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"         Saved: {path.name}")


# ----------------------------------------------------------------------------------------
# Beta comparison / decomposition table
# ----------------------------------------------------------------------------------------

def build_beta_comparison(pooled_df: pd.DataFrame,
                           fe_df: pd.DataFrame) -> pd.DataFrame:
    """Merge pooled and FE results into a side-by-side decomposition table.

    The compositional component is estimated as (beta_pooled - beta_FE):
    the portion of the aggregate trend explained by changes in who is posting
    rather than how existing speakers write over time.
    """
    if pooled_df.empty or fe_df.empty:
        return pd.DataFrame()

    p = pooled_df[['metric', 'beta', 'beta_annual', 'se', 'p_value', 'n_obs']].copy()
    f = fe_df[['metric',     'beta', 'beta_annual', 'se', 'p_value', 'n_obs']].copy()
    p.columns = ['metric'] + [f'pooled_{c}' for c in p.columns[1:]]
    f.columns = ['metric'] + [f'fe_{c}'     for c in f.columns[1:]]

    comp = p.merge(f, on='metric', how='outer')
    comp['compositional_beta']        = comp['pooled_beta']        - comp['fe_beta']
    comp['compositional_beta_annual'] = comp['pooled_beta_annual'] - comp['fe_beta_annual']
    return comp


# ----------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Trend analysis: utterance-level → speaker-month panel → OLS / Panel FE'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Path to lexical_master.csv')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory for all output files (created if absent)')
    parser.add_argument('--min-obs', type=int, default=DEFAULT_MIN_OBS,
                        help=f'Min monthly observations per speaker (default: {DEFAULT_MIN_OBS})')
    parser.add_argument('--min-speakers', type=int, default=DEFAULT_MIN_SPEAKERS_PER_SUB,
                        help=(f'Min speakers per subreddit for per-subreddit FE '
                              f'(default: {DEFAULT_MIN_SPEAKERS_PER_SUB})'))
    parser.add_argument('--skip-fe', action='store_true',
                        help='Skip panel FE steps (6 & 7) for a faster pooled-OLS-only run')
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # ------------------------------------------------------------------
    # Steps 1–4: load, filter, aggregate, prune
    # ------------------------------------------------------------------
    df    = load_and_prepare(args.input)
    df    = filter_invalid(df)
    panel = aggregate_to_panel(df)
    del df; gc.collect()

    panel = filter_sparse_speakers(panel, min_obs=args.min_obs)

    # Cross-subreddit aggregation for the pooled and pooled-FE models
    print("\n[Prep] Building cross-subreddit panel for steps 5 & 6...")
    cross_panel = _aggregate_cross_subreddit(panel)
    print(f"       {len(cross_panel):,} rows  |  "
          f"{cross_panel['speaker_id'].nunique():,} speakers")

    # ------------------------------------------------------------------
    # Step 5: Pooled OLS + ACF diagnostics
    # ------------------------------------------------------------------
    pooled_df, residuals = run_pooled_ols(cross_panel)

    if residuals:
        plot_acf_diagnostics(residuals, output_dir)

    # ------------------------------------------------------------------
    # Steps 6–7: Panel FE (skipped with --skip-fe or missing linearmodels)
    # ------------------------------------------------------------------
    fe_df     = pd.DataFrame()
    sub_fe_df = pd.DataFrame()

    if not args.skip_fe:
        fe_df     = run_panel_fe(cross_panel)
        sub_fe_df = run_per_subreddit_fe(panel, min_speakers=args.min_speakers)

    # ------------------------------------------------------------------
    # Decomposition table and outputs
    # ------------------------------------------------------------------
    comp_df = build_beta_comparison(pooled_df, fe_df)

    print(f"\n[Output] Writing results to {output_dir}...")

    def _save(df: pd.DataFrame, name: str):
        if df.empty:
            print(f"         Skipping {name} (empty).")
            return
        path = output_dir / name
        df.to_csv(path, index=False)
        print(f"         Saved: {name}  ({len(df)} rows)")

    _save(pooled_df,  'results_pooled_ols.csv')
    _save(fe_df,      'results_panel_fe.csv')
    _save(sub_fe_df,  'results_per_subreddit_fe.csv')
    _save(comp_df,    'beta_comparison.csv')

    print("\nDone.")


if __name__ == '__main__':
    main()
