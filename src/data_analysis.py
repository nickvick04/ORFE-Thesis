# ----------------------------------------------------------------------------------------
# This code is designed to analyze post-cluster computation CSV files
# Code Author: Nicholas Vickery, Princeton ORFE '26
# ----------------------------------------------------------------------------------------

from pathlib import Path
from typing import List

import pandas as pd

def combine_pipeline_csvs_to_lexical_master(
    convokit_root: str | Path,
    output_filename: str = "lexical_master.csv",
    variations_only: bool = True,
    skip_read_errors: bool = False,
) -> Path:
    """Combine pipeline-produced CSV files under Convokit variations into one master CSV.

    Expected structure:
    Convokit/
      Age-Variation/
        subreddit_a.csv
        subreddit_b.csv
      Topic-Variation/
        ...
    """
    root = Path(convokit_root).expanduser().resolve()

    if not root.exists() or not root.is_dir():
        alias_hint = ""
        if root.exists() and root.is_file():
            alias_hint = " (path is a file; if this is a macOS Alias, use the real folder path)"
        raise FileNotFoundError(f"Convokit root not found: {root}{alias_hint}")

    def _is_pipeline_csv(path: Path) -> bool:
        name = path.name
        return (
            name.endswith("_df.csv")
            or "_df_shard-" in name
            or name.endswith("_lexical_df.csv")
        )

    def _subreddit_from_filename(path: Path) -> str:
        name = path.name
        suffixes = ("_lexical_df.csv", "_df.csv")
        for suffix in suffixes:
            if name.endswith(suffix):
                return name[: -len(suffix)]
        shard_marker = "_df_shard-"
        if shard_marker in name and name.endswith(".csv"):
            return name.split(shard_marker, 1)[0]
        return path.stem

    variation_dirs = [p for p in root.iterdir() if p.is_dir()]
    if variations_only:
        variation_dirs = [p for p in variation_dirs if "variation" in p.name.lower()]

    csv_files: List[Path] = []
    for variation_dir in sorted(variation_dirs):
        variation_csvs = [
            p for p in sorted(variation_dir.glob("*.csv"))
            if p.name != output_filename and _is_pipeline_csv(p)
        ]
        csv_files.extend(variation_csvs)

    if not csv_files:
        raise FileNotFoundError(
            f"No pipeline CSV files found under selected folders in: {root}"
        )

    dfs = []
    for i, csv_path in enumerate(csv_files, start=1):
        print(f"[{i}/{len(csv_files)}] Reading {csv_path}")
        try:
            df = pd.read_csv(csv_path)
            df["source_variation"] = csv_path.parent.name
            df["subreddit"] = _subreddit_from_filename(csv_path)
            dfs.append(df)
        except (TimeoutError, OSError, pd.errors.ParserError) as exc:
            if skip_read_errors:
                print(f"Skipping unreadable file: {csv_path}\nReason: {exc}")
                continue
            raise RuntimeError(
                f"Failed to read CSV: {csv_path}\n"
                "If this is in a cloud-synced folder, ensure the file is downloaded locally "
                "or rerun with skip_read_errors=True."
            ) from exc

    if not dfs:
        raise RuntimeError("No readable CSV files were loaded.")

    master_df = pd.concat(dfs, ignore_index=True)
    output_path = root / output_filename
    master_df.to_csv(output_path, index=False)

    print(f"Combined {len(csv_files)} CSV files into:\n{output_path}")
    print(f"Total rows: {len(master_df)}")
    return output_path
