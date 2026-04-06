"""
convokit_plot_activity.py
==========================
Produces two time-series plots for the Convokit Data section:

  1. Monthly Active Users  — unique speakers per (subreddit, year_month)
  2. Monthly Post Volume   — total raw posts per (subreddit, year_month),
                             recovered from num_utterances_by_speaker_month

Each figure has three panels side-by-side, one per variation group
(Age, Topic, Culture), with one line per subreddit (3 lines per panel).
This layout keeps the 9 subreddits readable and grouped by research theme.

Both figures are saved as high-resolution PNG and vector PDF.

Usage
-----
    python convokit_plot_activity.py

    python convokit_plot_activity.py \\
        --csv     /scratch/network/nv9344/Thesis/Thesis-Data/lexical_master.csv \\
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

DEFAULT_CSV     = "/scratch/network/nv9344/Thesis/Thesis-Data/lexical_master.csv"
DEFAULT_OUT_DIR = "/scratch/network/nv9344/Thesis/Visualizations"

# Variation groups and their subreddits (in plot order, top-to-bottom in legend).
# Keys match the normalised source_variation label in the CSV
# ("Age-Variation" → "age", stripped by clean_and_prepare; but raw CSV keeps
#  "Age-Variation" so we strip it ourselves below).
VARIATION_GROUPS: dict[str, list[str]] = {
    "Age-Variation":     ["college", "parent", "teenagers"],
    "Topic-Variation":   ["relationships", "science", "worldnews"],
    "Culture-Variation": ["books", "movies", "religion"],
}

VARIATION_TITLES: dict[str, str] = {
    "Age-Variation":     "Age Variation",
    "Topic-Variation":   "Topic Variation",
    "Culture-Variation": "Culture Variation",
}

# Display labels for subreddits in the legend
SUBREDDIT_LABELS: dict[str, str] = {
    "college":       "r/college",
    "parent":        "r/Parenting",
    "teenagers":     "r/teenagers",
    "relationships": "r/relationships",
    "science":       "r/science",
    "worldnews":     "r/worldnews",
    "books":         "r/books",
    "movies":        "r/movies",
    "religion":      "r/religion",
}

# Three colorblind-distinguishable colors per variation group (Wong 2011 + extensions)
COLORS: dict[str, list[str]] = {
    "Age-Variation":     ["#0072B2", "#56B4E9", "#004C80"],   # blue family
    "Topic-Variation":   ["#009E73", "#56C9A0", "#005740"],   # teal-green family
    "Culture-Variation": ["#D55E00", "#E69F00", "#8B1A00"],   # red-orange family
}

# Line styles to further distinguish subreddits within each group
LINE_STYLES = ["-", "--", ":"]

# Drop months likely to be incomplete (data extraction cutoff)
DROP_AFTER = "2024-12"   # lexical_master covers through ~end of 2024

FIG_W, FIG_H = 14, 4.5   # wider to accommodate 3 panels
DPI = 300


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_monthly(csv_path: str) -> pd.DataFrame:
    """
    Load lexical_master.csv and aggregate to (subreddit, year_month) level.

    The raw CSV has:
      - timestamp   : datetime string  ("2010-07-12 02:19:03")
      - subreddit   : "subreddit-college", etc.
      - source_variation : "Age-Variation", etc.
      - num_utterances_by_speaker_month : raw monthly post count per speaker

    Returns a DataFrame with columns:
        variation, subreddit, date, active_users, post_volume
    """
    print("Reading CSV … (this may take several minutes on a large file)",
          flush=True)

    df = pd.read_csv(
        csv_path,
        usecols=["speaker_id", "subreddit", "source_variation",
                 "timestamp", "num_utterances_by_speaker_month"],
        low_memory=False,
    )

    # Normalise labels to match VARIATION_GROUPS keys and SUBREDDIT_LABELS
    # "subreddit-college" → "college"
    df["subreddit"] = (
        df["subreddit"].astype(str).str.strip().str.split("-", n=1).str[-1]
    )
    # "Age-Variation" stays as-is (already matches VARIATION_GROUPS keys)
    df["source_variation"] = df["source_variation"].astype(str).str.strip()

    # Derive year_month from timestamp
    df["date"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    monthly = (
        df.groupby(["source_variation", "subreddit", "year_month"])
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

    return monthly.sort_values(["source_variation", "subreddit", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Shared plot helpers
# ---------------------------------------------------------------------------

def _apply_panel_style(ax: plt.Axes, title: str, ylabel: str,
                       x_min: pd.Timestamp, x_max: pd.Timestamp) -> None:
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)

    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.YearLocator())

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"
    ))

    ax.tick_params(axis="both", labelsize=9)
    ax.tick_params(axis="x", which="minor", length=3)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_xlim(x_min, x_max)


def _add_panel_legend(ax: plt.Axes) -> None:
    ax.legend(frameon=True, framealpha=0.9, edgecolor="lightgrey",
              fontsize=9, loc="upper left")


def _save(fig: plt.Figure, out_dir: str, stem: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{stem}.{ext}"),
                    dpi=DPI, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Plot builder (shared logic for active users and post volume)
# ---------------------------------------------------------------------------

def _build_figure(monthly: pd.DataFrame, metric: str,
                  fig_title: str, ylabel: str,
                  out_dir: str, stem: str) -> None:
    variations = list(VARIATION_GROUPS.keys())
    n_panels   = len(variations)

    fig, axes = plt.subplots(1, n_panels, figsize=(FIG_W, FIG_H),
                             sharey=False)
    fig.suptitle(fig_title, fontsize=13, fontweight="bold", y=1.01)

    x_min = monthly["date"].min() - pd.DateOffset(months=2)
    x_max = monthly["date"].max() + pd.DateOffset(months=2)

    for ax, variation in zip(axes, variations):
        subreddits = VARIATION_GROUPS[variation]
        colors     = COLORS[variation]
        panel_data = monthly[monthly["source_variation"] == variation]

        for sub, color, ls in zip(subreddits, colors, LINE_STYLES):
            g = panel_data[panel_data["subreddit"] == sub]
            if g.empty:
                continue
            ax.plot(
                g["date"], g[metric],
                label=SUBREDDIT_LABELS.get(sub, sub),
                color=color,
                linewidth=1.8,
                linestyle=ls,
                alpha=0.9,
            )

        _apply_panel_style(ax, VARIATION_TITLES[variation], ylabel, x_min, x_max)
        _add_panel_legend(ax)

    fig.tight_layout()
    _save(fig, out_dir, stem)
    plt.close(fig)
    print(f"Saved: {stem}.png / .pdf")


# ---------------------------------------------------------------------------
# Public plot functions
# ---------------------------------------------------------------------------

def plot_active_users(monthly: pd.DataFrame, out_dir: str) -> None:
    _build_figure(
        monthly,
        metric="active_users",
        fig_title="Monthly Active Users by Subreddit",
        ylabel="Unique Speakers",
        out_dir=out_dir,
        stem="convokit_active_users_over_time",
    )


def plot_post_volume(monthly: pd.DataFrame, out_dir: str) -> None:
    _build_figure(
        monthly,
        metric="post_volume",
        fig_title="Monthly Post Volume by Subreddit",
        ylabel="Total Comments",
        out_dir=out_dir,
        stem="convokit_post_volume_over_time",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot monthly active users and post volume for Convokit subreddits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv",     default=DEFAULT_CSV,
                   help="Path to lexical_master.csv")
    p.add_argument("--out_dir", default=DEFAULT_OUT_DIR,
                   help="Directory to write PNG and PDF files")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading data from : {args.csv}")
    monthly = load_monthly(args.csv)

    print(f"\nMonths per subreddit:")
    for (var, sub), g in monthly.groupby(["source_variation", "subreddit"]):
        print(f"  {VARIATION_TITLES.get(var, var):<20} "
              f"{SUBREDDIT_LABELS.get(sub, sub):<18} "
              f"{g['date'].min().strftime('%b %Y')} – "
              f"{g['date'].max().strftime('%b %Y')}  "
              f"({len(g)} months)")

    print(f"\nWriting figures to: {args.out_dir}")
    plot_active_users(monthly, args.out_dir)
    plot_post_volume( monthly, args.out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
