from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_CSV = PROJECT_ROOT / "combined_coded_responses.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"


def default_output_dir(script_filename: str) -> Path:
    return DEFAULT_OUTPUT_ROOT / Path(script_filename).stem


def build_parser(description: str, script_filename: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--csv",
        default=str(DEFAULT_DATA_CSV),
        help="Path to combined_coded_responses.csv",
    )
    parser.add_argument(
        "--outdir",
        default=str(default_output_dir(script_filename)),
        help="Directory for CSV and plot outputs",
    )
    return parser


def ensure_outdir(path: str | Path) -> Path:
    outdir = Path(path)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def save_csv(df: pd.DataFrame, path: str | Path, *, index: bool = False) -> Path:
    outpath = Path(path)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outpath, index=index)
    return outpath
