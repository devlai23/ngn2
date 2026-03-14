"""
research question 2 part b

Young people are tech-savvy and know how to change their feeds and algorithms to avoid news when overwhelmed. 

tested with q8 + chi squared test + cramers v
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

from common import build_parser, ensure_outdir, save_csv

AGE_COL = "age"
Q8_FEED_MIX_COL = "Q8_m_4"

def clean_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.replace("__NA__", np.nan), errors="coerce")

def q8_selected_binary(s: pd.Series) -> pd.Series:
    x = clean_numeric(s)
    return x.map({1: 1, 2: 0})

def add_age_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[AGE_COL] = clean_numeric(out[AGE_COL])

    bins = [0, 17, 29, 39, 49, 64, 120]
    labels = ["<18", "18–29", "30–39", "40–49", "50–64", "65+"]
    out["age_bin"] = pd.cut(out[AGE_COL], bins=bins, labels=labels, include_lowest=True)

    out["age_bin"] = out["age_bin"].astype(
        pd.CategoricalDtype(categories=labels, ordered=True)
    )
    return out

def plot_selected_rate(row_pct: pd.DataFrame, out_png: str):
    if 1 not in row_pct.columns:
        raise ValueError("row_pct missing column 1 (selected). Check coding.")

    y = row_pct[1].astype(float)

    plt.figure(figsize=(10, 5))
    plt.bar(row_pct.index.astype(str), y.values, color="#fe8543")
    plt.ylabel("% selected (within age group)")
    plt.xlabel("Age group")
    plt.title("Q8: Try to change the mix of content in your social media feeds")
    plt.ylim(0, max(35, np.nanmax(y.values) * 1.15))
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Saved plot: {out_png}")

def main():
    parser = build_parser(__doc__.strip().splitlines()[0], __file__)
    parser.add_argument("--age_col", default=AGE_COL)
    parser.add_argument("--q8_col", default=Q8_FEED_MIX_COL)
    args = parser.parse_args()

    outdir = ensure_outdir(args.outdir)
    df = pd.read_csv(args.csv)
    df = add_age_bins(df)

    df["feed_mix_selected"] = q8_selected_binary(df[args.q8_col])

    tmp = df.dropna(subset=["age_bin", "feed_mix_selected"]).copy()
    tmp["feed_mix_selected"] = tmp["feed_mix_selected"].astype(int)

    ct = pd.crosstab(tmp["age_bin"], tmp["feed_mix_selected"])
    ct = ct.reindex(columns=[0, 1], fill_value=0)
    save_csv(ct.reset_index(), outdir / "q8_feed_mix_by_age_counts.csv")
    print("Contingency table (counts):")
    print(ct)

    row_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    save_csv(row_pct.reset_index(), outdir / "q8_feed_mix_by_age_rowpct.csv")
    print("\nRow % (within age group):")
    print(row_pct.round(1))

    plot_selected_rate(row_pct, str(outdir / "q8_feed_mix_by_age.png"))

    chi2, p, dof, expected = chi2_contingency(ct)
    print("\nChi-square test of independence")
    print(f"chi2 = {chi2:.4f}")
    print(f"dof  = {dof}")
    print(f"p    = {p:.4e}")

    expected_df = pd.DataFrame(expected, index=ct.index, columns=ct.columns)
    save_csv(expected_df.reset_index(), outdir / "q8_feed_mix_by_age_expected.csv")
    print("\nExpected counts:")
    print(expected_df.round(2))

    low_expected = int((expected < 5).sum())
    total_cells = expected.size
    print(f"\nCells with expected count < 5: {low_expected}/{total_cells}")

    n = ct.to_numpy().sum()
    r, k = ct.shape
    cramers_v = np.sqrt(chi2 / (n * (min(r - 1, k - 1))))
    save_csv(pd.DataFrame([{
        "chi2": float(chi2),
        "dof": int(dof),
        "p": float(p),
        "cramers_v": float(cramers_v),
        "low_expected_cells": low_expected,
        "total_cells": int(total_cells),
    }]), outdir / "q8_feed_mix_by_age_test.csv")
    print(f"Cramer's V = {cramers_v:.4f}")
    print(f"\nSaved outputs to: {Path(outdir)}")


if __name__ == "__main__":
    main()
