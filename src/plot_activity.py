"""
plot_activity.py
================
Produces two time-series plots for the ArcticShift Data section:

  1. Monthly Active Users  — unique speakers per (subreddit, year_month)
  2. Monthly Post Volume   — total raw posts per (subreddit, year_month),
                             recovered from num_utterances_by_speaker_month
                             since the pipeline keeps only 1 post per speaker
                             per month in the processed CSV.

Both figures are saved as high-resolution PNG and vector PDF.

Usage
-----
    python plot_activity.py

    # Override paths or output directory
    python plot_activity.py \\
        --csv     /scratch/network/nv9344/Thesis/Thesis-Data/ArcticShift/lexical_df_combined.csv \\
        --out_dir /scratch/network/nv9344/Thesis/Visualizations

Author: Nicholas Vickery, Princeton ORFE '26
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CSV     = "/scratch/network/nv9344/Thesis/Thesis-Data/ArcticShift/lexical_df_combined.csv"
DEFAULT_OUT_DIR = "/scratch/network/nv9344/Thesis/Visualizations"

# Display labels for subreddit CSV values
SUBREDDIT_LABELS: dict[str, str] = {
    "teenagers": "r/teenagers",
    "college":   "r/college",
    "Parenting": "r/parenting",
    "retirement":"r/retirement",
}

# Plot order (youngest → oldest community, top to bottom in legend)
SUBREDDIT_ORDER = ["teenagers", "college", "Parenting", "retirement"]

# Colorblind-friendly palette (Wong 2011), assigned youngest → oldest
COLORS: dict[str, str] = {
    "teenagers": "#0072B2",   # blue
    "college":   "#E69F00",   # amber
    "Parenting": "#009E73",   # teal-green
    "retirement":"#D55E00",   # vermillion
}

# Drop months that are likely incomplete (the data extraction date).
# Set to None to keep all months.
DROP_AFTER = "2026-03"   # keep through March 2026; April is only ~6 days of data

# Figure dimensions (inches) — sized for a two-column thesis layout
FIG_W, FIG_H = 9, 4.5

DPI = 300


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_monthly(csv_path: str) -> pd.DataFrame:
    """
    Load the combined CSV and aggregate to (subreddit, year_month) level.

    Returns
    -------
    pd.DataFrame with columns:
        subreddit, date, active_users, post_volume
    """
    df = pd.read_csv(
        csv_path,
        usecols=["speaker_id", "subreddit", "year_month",
                 "num_utterances_by_speaker_month"],
        low_memory=False,
    )

    monthly = (
        df.groupby(["subreddit", "year_month"])
          .agg(
              active_users=("speaker_id",                      "nunique"),
              post_volume =("num_utterances_by_speaker_month", "sum"),
          )
          .reset_index()
    )
    monthly["date"] = pd.to_datetime(monthly["year_month"])

    if DROP_AFTER is not None:
        cutoff = pd.to_datetime(DROP_AFTER)
        monthly = monthly[monthly["date"] <= cutoff]

    return monthly.sort_values(["subreddit", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Shared plot helpers
# ---------------------------------------------------------------------------

def _apply_shared_style(ax: plt.Axes, title: str, ylabel: str) -> None:
    """Apply consistent axis formatting to both plots."""
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)

    # x-axis: major ticks every year, minor every 6 months
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[7]))

    # y-axis: comma-separated thousands
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"
    ))

    ax.tick_params(axis="both", labelsize=10)
    ax.tick_params(axis="x", which="minor", length=3)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_xlim(pd.Timestamp("2014-10-01"), pd.Timestamp("2026-05-01"))


def _add_legend(ax: plt.Axes) -> None:
    ax.legend(
        frameon=True,
        framealpha=0.9,
        edgecolor="lightgrey",
        fontsize=10,
        loc="upper left",
    )


# ---------------------------------------------------------------------------
# Plot 1: Monthly Active Users
# ---------------------------------------------------------------------------

def plot_active_users(monthly: pd.DataFrame, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    for sub in SUBREDDIT_ORDER:
        g = monthly[monthly["subreddit"] == sub]
        ax.plot(
            g["date"], g["active_users"],
            label=SUBREDDIT_LABELS[sub],
            color=COLORS[sub],
            linewidth=1.8,
            alpha=0.9,
        )

    _apply_shared_style(
        ax,
        title="Monthly Active Users by Subreddit",
        ylabel="Unique Speakers",
    )
    _add_legend(ax)

    fig.tight_layout()
    _save(fig, out_dir, "active_users_over_time")
    plt.close(fig)
    print("Saved: active_users_over_time.png / .pdf")


# ---------------------------------------------------------------------------
# Plot 2: Monthly Post Volume
# ---------------------------------------------------------------------------

def plot_post_volume(monthly: pd.DataFrame, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    for sub in SUBREDDIT_ORDER:
        g = monthly[monthly["subreddit"] == sub]
        ax.plot(
            g["date"], g["post_volume"],
            label=SUBREDDIT_LABELS[sub],
            color=COLORS[sub],
            linewidth=1.8,
            alpha=0.9,
        )

    _apply_shared_style(
        ax,
        title="Monthly Post Volume by Subreddit",
        ylabel="Total Comments",
    )
    _add_legend(ax)

    fig.tight_layout()
    _save(fig, out_dir, "post_volume_over_time")
    plt.close(fig)
    print("Saved: post_volume_over_time.png / .pdf")


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, out_dir: str, stem: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(path, dpi=DPI, bbox_inches="tight")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot monthly active users and post volume for ArcticShift subreddits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv",     default=DEFAULT_CSV,
                   help="Path to lexical_df_combined.csv")
    p.add_argument("--out_dir", default=DEFAULT_OUT_DIR,
                   help="Directory to write PNG and PDF files")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading data from: {args.csv}")
    monthly = load_monthly(args.csv)

    print(f"Months per subreddit:")
    for sub, g in monthly.groupby("subreddit"):
        print(f"  {SUBREDDIT_LABELS.get(sub, sub):<18} "
              f"{g['date'].min().strftime('%b %Y')} – "
              f"{g['date'].max().strftime('%b %Y')}  "
              f"({len(g)} months)")
    print()

    print(f"Writing figures to: {args.out_dir}")
    plot_active_users(monthly, args.out_dir)
    plot_post_volume( monthly, args.out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
