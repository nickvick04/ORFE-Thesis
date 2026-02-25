# ----------------------------------------------------------------------------------------
# This code is designed to analyze post-cluster computation CSV files
# Code Author: Nicholas Vickery, Princeton ORFE '26
# ----------------------------------------------------------------------------------------

from pathlib import Path
from typing import List

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONVOKIT_ROOT = SCRIPT_DIR.parent.parent / "Thesis-Data" / "Convokit"


def combine_pipeline_csvs_to_lexical_master(convokit_root: str | Path = DEFAULT_CONVOKIT_ROOT,
    output_filename: str = "lexical_master.csv",) -> Path:
    """Combine pipeline-produced CSV files under Convokit variations into one master CSV.

    Expected structure:
    Convokit/
      Age-Variation/
        subreddit_a.csv
        subreddit_b.csv
      Topic-Variation/
        ...
    """
    root = Path(convokit_root).resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Convokit root not found: {root}")

    csv_files: List[Path] = []
    for variation_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        variation_csvs = [
            p for p in sorted(variation_dir.glob("*.csv"))
            if p.name != output_filename
        ]
        csv_files.extend(variation_csvs)

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under variation folders in: {root}")

    dfs = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        df["source_variation"] = csv_path.parent.name
        df["source_file"] = csv_path.name
        dfs.append(df)

    master_df = pd.concat(dfs, ignore_index=True)
    output_path = root / output_filename
    master_df.to_csv(output_path, index=False)

    print(f"Combined {len(csv_files)} CSV files into:\n{output_path}")
    print(f"Total rows: {len(master_df)}")
    return output_path
